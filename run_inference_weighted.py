"""
SAM 3D Objects Weighted Inference Script

This script extends the standard inference with attention-based weighted fusion.
Instead of simple averaging across views, it uses attention entropy to determine
per-latent fusion weights.

Key features:
    - Per-latent weighting based on attention entropy
    - Configurable weighting parameters (alpha, layer, step)
    - Optional visualization of weights and entropy
    - Extensible architecture for adding new confidence factors
    - Support for external pointmaps from DA3 (Depth Anything 3)
    - GLB merge visualization (SAM3D output + DA3 scene)

Usage:
    # Basic weighted inference (default: use entropy weighting)
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3
    
    # Disable weighting (simple average, like original)
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 --no_weighting
    
    # With visualization
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 --visualize_weights
    
    # Custom weighting parameters
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
        --entropy_alpha 3.0 --attention_layer 6 --attention_step 0
    
    # Use external pointmaps from DA3 (Depth Anything 3) and merge GLB for comparison
    # First run: python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example
    # Then:
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
        --da3_output ./da3_outputs/example/da3_output.npz --merge_da3_glb
"""
import sys
import argparse
from pathlib import Path
import subprocess
import tempfile
import shutil
from typing import List, Optional
from datetime import datetime
import numpy as np
import torch
from loguru import logger

# 导入推理代码
sys.path.append("notebook")
from inference import Inference
from load_images_and_masks import load_images_and_masks_from_path

from sam3d_objects.utils.cross_attention_logger import CrossAttentionLogger
from sam3d_objects.utils.latent_weighting import WeightingConfig, LatentWeightManager
from pytorch3d.transforms import Transform3d, quaternion_to_matrix


def merge_glb_with_da3_aligned(
    sam3d_glb_path: Path, 
    da3_output_dir: Path,
    sam3d_pose: dict,
    output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    将 SAM3D 重建的物体与 DA3 的完整场景 GLB 对齐合并。
    
    ## 核心发现
    
    DA3 的 scene.glb 在 metadata 中保存了对齐矩阵 `hf_alignment`！
    这个矩阵 A = T_center @ M @ w2c0，包含了：
    - w2c0: 第一帧相机的 world-to-camera
    - M: CV -> glTF 坐标系变换
    - T_center: 居中平移
    
    我们可以直接读取这个矩阵，用于对齐 SAM3D 物体。
    
    ## 变换策略
    
    SAM3D 物体变换链：
    1. canonical (Z-up) -> Y-up 旋转
    2. 应用 SAM3D pose -> PyTorch3D 相机空间
    3. PyTorch3D -> CV 相机空间: (-x, -y, z) -> (x, y, z)
    4. 应用 DA3 的对齐矩阵 A（从 scene.glb metadata 读取）
    
    Args:
        sam3d_glb_path: SAM3D 输出的 GLB 文件路径 (canonical space)
        da3_output_dir: DA3 输出目录，包含 scene.glb
        sam3d_pose: SAM3D 的 pose 参数 {'scale', 'rotation', 'translation'}
        output_path: 输出路径
    
    Returns:
        对齐后的 GLB 文件路径
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed, cannot merge GLB files")
        return None
    
    if not sam3d_glb_path.exists():
        logger.warning(f"SAM3D GLB not found: {sam3d_glb_path}")
        return None
    
    # 查找 DA3 的 scene.glb
    da3_scene_glb = da3_output_dir / "scene.glb"
    da3_npz = da3_output_dir / "da3_output.npz"
    
    if not da3_scene_glb.exists():
        logger.warning(f"DA3 scene.glb not found: {da3_scene_glb}")
        logger.warning("Please run DA3 with visualization enabled")
        return None
    
    if output_path is None:
        output_path = sam3d_glb_path.parent / f"{sam3d_glb_path.stem}_merged_scene.glb"
    
    try:
        # 加载 SAM3D GLB
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # 加载 DA3 scene.glb
        da3_scene = trimesh.load(str(da3_scene_glb))
        
        # 尝试从 DA3 scene 的 metadata 中读取对齐矩阵
        alignment_matrix = None
        if hasattr(da3_scene, 'metadata') and da3_scene.metadata is not None:
            alignment_matrix = da3_scene.metadata.get('hf_alignment', None)
        
        if alignment_matrix is None:
            logger.warning("DA3 scene.glb does not contain alignment matrix (hf_alignment)")
            logger.warning("Falling back to computing alignment from extrinsics")
            
            # 回退：从 npz 中读取 extrinsics 计算对齐矩阵
            if not da3_npz.exists():
                logger.warning(f"DA3 da3_output.npz not found: {da3_npz}")
                return None
            
            da3_data = np.load(da3_npz)
            da3_extrinsics = da3_data["extrinsics"]
            
            # 获取第一帧的 w2c
            w2c0 = da3_extrinsics[0]
            if w2c0.shape == (3, 4):
                w2c0_44 = np.eye(4, dtype=np.float64)
                w2c0_44[:3, :4] = w2c0
                w2c0 = w2c0_44
            
            # CV -> glTF 坐标系变换
            M_cv_to_gltf = np.eye(4, dtype=np.float64)
            M_cv_to_gltf[1, 1] = -1.0
            M_cv_to_gltf[2, 2] = -1.0
            
            # 计算对齐矩阵（不含居中，需要从 scene 点云计算）
            A_no_center = M_cv_to_gltf @ w2c0
            
            # 从 DA3 scene 获取点云中心
            da3_points = []
            if isinstance(da3_scene, trimesh.Scene):
                for geom in da3_scene.geometry.values():
                    if hasattr(geom, 'vertices'):
                        da3_points.append(geom.vertices)
            elif hasattr(da3_scene, 'vertices'):
                da3_points.append(da3_scene.vertices)
            
            if da3_points:
                all_pts = np.vstack(da3_points)
                # DA3 scene 已经居中了，所以我们需要找到它的中心
                # 但由于已经居中，中心应该接近 0
                # 我们需要反推原始的居中偏移
                # 这比较复杂，暂时假设居中偏移为 0
                alignment_matrix = A_no_center
                logger.warning("Using alignment without centering (may be slightly off)")
        
        logger.info(f"[Merge Scene] Alignment matrix:\n{alignment_matrix}")
        
        # 提取 SAM3D pose 参数
        scale = sam3d_pose.get('scale', np.array([1.0, 1.0, 1.0]))
        rotation_quat = sam3d_pose.get('rotation', np.array([1.0, 0.0, 0.0, 0.0]))  # wxyz
        translation = sam3d_pose.get('translation', np.array([0.0, 0.0, 0.0]))
        
        if len(scale.shape) > 1:
            scale = scale.flatten()
        if len(rotation_quat.shape) > 1:
            rotation_quat = rotation_quat.flatten()
        if len(translation.shape) > 1:
            translation = translation.flatten()
        
        logger.info(f"[Merge Scene] SAM3D pose:")
        logger.info(f"  scale: {scale}")
        logger.info(f"  rotation (wxyz): {rotation_quat}")
        logger.info(f"  translation: {translation}")
        
        # ========================================
        # SAM3D 物体变换
        # ========================================
        
        # Z-up to Y-up 旋转矩阵
        z_up_to_y_up_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float32)
        
        # 构建 PyTorch3D 空间的 pose 变换
        quat_tensor = torch.tensor(rotation_quat, dtype=torch.float32).unsqueeze(0)
        R_sam3d = quaternion_to_matrix(quat_tensor)
        scale_tensor = torch.tensor(scale, dtype=torch.float32).reshape(1, -1)
        if scale_tensor.shape[-1] == 1:
            scale_tensor = scale_tensor.repeat(1, 3)
        translation_tensor = torch.tensor(translation, dtype=torch.float32).reshape(1, 3)
        pose_transform = (
            Transform3d(dtype=torch.float32)
            .scale(scale_tensor)
            .rotate(R_sam3d)
            .translate(translation_tensor)
        )
        
        # PyTorch3D 到 CV 相机空间的变换
        p3d_to_cv = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        
        def transform_sam3d_to_da3_space(vertices):
            """
            将 SAM3D canonical space 的顶点变换到 DA3 场景空间 (glTF)
            """
            # Step 1: Z-up to Y-up
            v_rotated = vertices @ z_up_to_y_up_matrix.T
            
            # Step 2: 应用 SAM3D pose -> PyTorch3D 空间
            pts_local = torch.from_numpy(v_rotated).float().unsqueeze(0)
            pts_p3d = pose_transform.transform_points(pts_local).squeeze(0).numpy()
            
            # Step 3: PyTorch3D -> CV 相机空间
            pts_cv = pts_p3d @ p3d_to_cv.T
            
            # Step 4: 应用 DA3 对齐矩阵
            pts_final = trimesh.transform_points(pts_cv, alignment_matrix)
            
            return pts_final
        
        # ========================================
        # 创建合并场景
        # ========================================
        
        merged_scene = trimesh.Scene()
        
        # 添加 DA3 场景（保持原样，因为它已经在正确的坐标系中）
        if isinstance(da3_scene, trimesh.Scene):
            for name, geom in da3_scene.geometry.items():
                merged_scene.add_geometry(geom.copy(), node_name=f"da3_{name}")
        else:
            merged_scene.add_geometry(da3_scene.copy(), node_name="da3_scene")
        
        # 变换并添加 SAM3D 物体
        sam3d_vertices_final = None
        if isinstance(sam3d_scene, trimesh.Scene):
            for name, geom in sam3d_scene.geometry.items():
                if hasattr(geom, 'vertices'):
                    geom_copy = geom.copy()
                    geom_copy.vertices = transform_sam3d_to_da3_space(geom_copy.vertices)
                    merged_scene.add_geometry(geom_copy, node_name=f"sam3d_{name}")
                    if sam3d_vertices_final is None:
                        sam3d_vertices_final = geom_copy.vertices
                else:
                    merged_scene.add_geometry(geom, node_name=f"sam3d_{name}")
        else:
            if hasattr(sam3d_scene, 'vertices'):
                sam3d_scene_copy = sam3d_scene.copy()
                sam3d_scene_copy.vertices = transform_sam3d_to_da3_space(sam3d_scene_copy.vertices)
                sam3d_vertices_final = sam3d_scene_copy.vertices
                merged_scene.add_geometry(sam3d_scene_copy, node_name="sam3d_object")
            else:
                merged_scene.add_geometry(sam3d_scene.copy(), node_name="sam3d_object")
        
        # 打印对齐信息
        if sam3d_vertices_final is not None:
            logger.info(f"[Merge Scene] SAM3D object in DA3 space:")
            logger.info(f"  X: [{sam3d_vertices_final[:, 0].min():.4f}, {sam3d_vertices_final[:, 0].max():.4f}]")
            logger.info(f"  Y: [{sam3d_vertices_final[:, 1].min():.4f}, {sam3d_vertices_final[:, 1].max():.4f}]")
            logger.info(f"  Z: [{sam3d_vertices_final[:, 2].min():.4f}, {sam3d_vertices_final[:, 2].max():.4f}]")
        
        # 打印 DA3 场景范围
        da3_pts = []
        if isinstance(da3_scene, trimesh.Scene):
            for geom in da3_scene.geometry.values():
                if hasattr(geom, 'vertices'):
                    da3_pts.append(geom.vertices)
        if da3_pts:
            da3_all = np.vstack(da3_pts)
            logger.info(f"[Merge Scene] DA3 scene bounds:")
            logger.info(f"  X: [{da3_all[:, 0].min():.4f}, {da3_all[:, 0].max():.4f}]")
            logger.info(f"  Y: [{da3_all[:, 1].min():.4f}, {da3_all[:, 1].max():.4f}]")
            logger.info(f"  Z: [{da3_all[:, 2].min():.4f}, {da3_all[:, 2].max():.4f}]")
        
        # 导出
        merged_scene.export(str(output_path))
        logger.info(f"[Merge Scene] Saved merged GLB: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Failed to merge GLB files: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_multiview_pose_consistency(
    sam3d_glb_path: Path,
    all_view_poses_decoded: list,
    da3_extrinsics: np.ndarray,
    da3_scene_glb_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    可视化多视角 pose 一致性：将每个视角预测的物体都放到世界坐标系中。
    
    如果所有视角的预测一致，这些物体应该重叠在一起。
    如果不一致，可以直观地看到哪些视角的预测偏离了。
    
    Args:
        sam3d_glb_path: SAM3D 输出的 GLB 文件路径 (canonical space)
        all_view_poses_decoded: 所有视角解码后的 pose 列表
        da3_extrinsics: DA3 的相机外参 (N, 3, 4) or (N, 4, 4), world-to-camera
        da3_scene_glb_path: DA3 的 scene.glb 路径（可选，用于添加场景背景）
        output_path: 输出路径
    
    Returns:
        可视化 GLB 文件路径
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed, cannot create visualization")
        return None
    
    if not sam3d_glb_path.exists():
        logger.warning(f"SAM3D GLB not found: {sam3d_glb_path}")
        return None
    
    if output_path is None:
        output_path = sam3d_glb_path.parent / "multiview_pose_consistency.glb"
    
    try:
        # 加载 SAM3D GLB (canonical space)
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # 提取 canonical 顶点
        canonical_vertices = None
        canonical_faces = None
        if isinstance(sam3d_scene, trimesh.Scene):
            for name, geom in sam3d_scene.geometry.items():
                if hasattr(geom, 'vertices'):
                    canonical_vertices = geom.vertices.copy()
                    if hasattr(geom, 'faces'):
                        canonical_faces = geom.faces.copy()
                    break
        elif hasattr(sam3d_scene, 'vertices'):
            canonical_vertices = sam3d_scene.vertices.copy()
            if hasattr(sam3d_scene, 'faces'):
                canonical_faces = sam3d_scene.faces.copy()
        
        if canonical_vertices is None:
            logger.warning("No vertices found in SAM3D GLB")
            return None
        
        logger.info(f"[MultiView Viz] Canonical vertices: {canonical_vertices.shape}")
        logger.info(f"[MultiView Viz] Number of views: {len(all_view_poses_decoded)}")
        
        # Z-up to Y-up 旋转矩阵（与 merge_glb_with_da3_aligned 一致）
        z_up_to_y_up_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float32)
        
        # PyTorch3D 到 CV 相机空间的变换
        p3d_to_cv = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        
        # CV 到 glTF 坐标系变换
        M_cv_to_gltf = np.eye(4, dtype=np.float64)
        M_cv_to_gltf[1, 1] = -1.0
        M_cv_to_gltf[2, 2] = -1.0
        
        # 创建场景
        merged_scene = trimesh.Scene()
        
        # 如果有 DA3 scene，添加为背景
        alignment_matrix = None
        if da3_scene_glb_path is not None and da3_scene_glb_path.exists():
            da3_scene = trimesh.load(str(da3_scene_glb_path))
            
            # 获取对齐矩阵
            if hasattr(da3_scene, 'metadata') and da3_scene.metadata is not None:
                alignment_matrix = da3_scene.metadata.get('hf_alignment', None)
            
            # 添加 DA3 场景（半透明灰色）
            if isinstance(da3_scene, trimesh.Scene):
                for name, geom in da3_scene.geometry.items():
                    geom_copy = geom.copy()
                    if hasattr(geom_copy, 'visual'):
                        geom_copy.visual.face_colors = [128, 128, 128, 100]
                    merged_scene.add_geometry(geom_copy, node_name=f"da3_{name}")
        
        # 为每个视角创建变换后的物体
        colors_per_view = [
            [255, 0, 0, 200],     # View 0: 红
            [0, 255, 0, 200],     # View 1: 绿
            [0, 0, 255, 200],     # View 2: 蓝
            [255, 255, 0, 200],   # View 3: 黄
            [255, 0, 255, 200],   # View 4: 品红
            [0, 255, 255, 200],   # View 5: 青
            [255, 128, 0, 200],   # View 6: 橙
            [128, 0, 255, 200],   # View 7: 紫
        ]
        
        for view_idx, pose in enumerate(all_view_poses_decoded):
            # 提取 pose 参数
            translation = np.array(pose.get('translation', [[0, 0, 0]])).flatten()[:3]
            rotation_quat = np.array(pose.get('rotation', [[1, 0, 0, 0]])).flatten()[:4]
            scale = np.array(pose.get('scale', [[1, 1, 1]])).flatten()[:3]
            
            # 构建变换（与 merge_glb_with_da3_aligned 一致）
            quat_tensor = torch.tensor(rotation_quat, dtype=torch.float32).unsqueeze(0)
            R_sam3d = quaternion_to_matrix(quat_tensor)
            scale_tensor = torch.tensor(scale, dtype=torch.float32).reshape(1, -1)
            if scale_tensor.shape[-1] == 1:
                scale_tensor = scale_tensor.repeat(1, 3)
            translation_tensor = torch.tensor(translation, dtype=torch.float32).reshape(1, 3)
            pose_transform = (
                Transform3d(dtype=torch.float32)
                .scale(scale_tensor)
                .rotate(R_sam3d)
                .translate(translation_tensor)
            )
            
            # 变换顶点
            # Step 1: Z-up to Y-up
            v_rotated = canonical_vertices @ z_up_to_y_up_matrix.T
            
            # Step 2: 应用 SAM3D pose -> PyTorch3D 空间
            pts_local = torch.from_numpy(v_rotated).float().unsqueeze(0)
            pts_p3d = pose_transform.transform_points(pts_local).squeeze(0).numpy()
            
            # Step 3: PyTorch3D -> CV 相机空间
            pts_cv = pts_p3d @ p3d_to_cv.T
            
            # Step 4: View i 的相机空间 -> 世界坐标系
            w2c_i = da3_extrinsics[view_idx]
            if w2c_i.shape == (3, 4):
                w2c_i_44 = np.eye(4, dtype=np.float64)
                w2c_i_44[:3, :4] = w2c_i
                w2c_i = w2c_i_44
            c2w_i = np.linalg.inv(w2c_i)
            pts_world = trimesh.transform_points(pts_cv, c2w_i)
            
            # Step 5: 世界坐标系 -> glTF 坐标系
            pts_gltf = trimesh.transform_points(pts_world, M_cv_to_gltf)
            
            # Step 6: 如果有对齐矩阵，应用居中偏移
            if alignment_matrix is not None and view_idx == 0:
                # 使用 View 0 计算居中偏移
                # alignment_matrix 应用于 View 0 的 CV 空间点
                pts_aligned_v0 = trimesh.transform_points(pts_cv, alignment_matrix)
                center_offset = pts_aligned_v0.mean(axis=0) - pts_gltf.mean(axis=0)
            
            if alignment_matrix is not None:
                pts_final = pts_gltf + center_offset
            else:
                pts_final = pts_gltf
            
            # 过滤无效点
            valid = np.isfinite(pts_final).all(axis=1)
            pts_final = pts_final[valid]
            
            # 创建 mesh
            color = colors_per_view[view_idx % len(colors_per_view)]
            if canonical_faces is not None and valid.sum() == len(canonical_vertices):
                mesh = trimesh.Trimesh(
                    vertices=pts_final,
                    faces=canonical_faces,
                    process=False
                )
                mesh.visual.face_colors = color
            else:
                mesh = trimesh.PointCloud(pts_final, colors=np.tile(color, (len(pts_final), 1)))
            
            merged_scene.add_geometry(mesh, node_name=f"view{view_idx}_object")
            
            logger.info(f"  View {view_idx}: center = {pts_final.mean(axis=0)}, scale = {scale[0]:.4f}")
        
        # 导出
        merged_scene.export(str(output_path))
        logger.info(f"[MultiView Viz] Saved: {output_path}")
        logger.info(f"  Colors: View0=Red, View1=Green, View2=Blue, View3=Yellow, View4=Magenta, View5=Cyan, View6=Orange, View7=Purple")
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Failed to create multiview visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import SSIPointmapNormalizer
from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer


def overlay_sam3d_on_pointmap(
    sam3d_glb_path: Path,
    input_pointmap,
    sam3d_pose: dict,
    input_image = None,
    output_path: Optional[Path] = None,
    pointmap_scale: Optional[np.ndarray] = None,
    pointmap_shift: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """
    将 SAM3D 重建的物体叠加到输入的 pointmap 上。
    
    SAM3D 的 pose 参数 (scale, rotation, translation) 已经是真实世界尺度，
    并且是在 PyTorch3D 相机空间中。
    输入的 pointmap 也应该在 PyTorch3D 相机空间中。
    
    变换流程：
    SAM3D canonical (±0.5)
        ↓ scale * rotation + translation  (SAM3D pose，真实世界尺度，PyTorch3D 空间)
    PyTorch3D 相机空间 (真实世界尺度)
    
    Args:
        sam3d_glb_path: SAM3D 输出的 GLB 文件路径 (canonical space)
        input_pointmap: 输入的 pointmap, shape (3, H, W), 已经在 PyTorch3D 相机空间
        sam3d_pose: SAM3D 的 pose 参数 {'scale', 'rotation', 'translation'}
        input_image: 原始图像，用于给点云上色
        output_path: 输出路径
    
    Returns:
        叠加后的 GLB 文件路径
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh or scipy not installed, cannot create overlay GLB")
        return None
    
    if not sam3d_glb_path.exists():
        logger.warning(f"SAM3D GLB not found: {sam3d_glb_path}")
        return None
    
    if output_path is None:
        output_path = sam3d_glb_path.parent / f"{sam3d_glb_path.stem}_overlay.glb"
    
    try:
        # 加载 SAM3D GLB
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # 提取 SAM3D pose 参数 (已经在 PyTorch3D 相机空间，真实世界尺度)
        scale = sam3d_pose.get('scale', np.array([1.0, 1.0, 1.0]))
        rotation_quat = sam3d_pose.get('rotation', np.array([1.0, 0.0, 0.0, 0.0]))  # wxyz
        translation = sam3d_pose.get('translation', np.array([0.0, 0.0, 0.0]))
        
        if len(scale.shape) > 1:
            scale = scale.flatten()
        if len(rotation_quat.shape) > 1:
            rotation_quat = rotation_quat.flatten()
        if len(translation.shape) > 1:
            translation = translation.flatten()
        
        logger.info(f"[Overlay] SAM3D pose (PyTorch3D 相机空间):")
        logger.info(f"  scale: {scale} (物体尺寸，单位：米)")
        logger.info(f"  rotation (wxyz): {rotation_quat}")
        logger.info(f"  translation: {translation} (物体位置，单位：米)")
        
        # SAM3D 内部对 GLB 顶点做了 z-up -> y-up 旋转
        # 需与 layout_post_optimization_utils.get_mesh 完全一致
        # 变换矩阵：X = X, Y = -Z, Z = Y
        z_up_to_y_up_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float32)
        quat_tensor = torch.tensor(rotation_quat, dtype=torch.float32).unsqueeze(0)
        R_sam3d = quaternion_to_matrix(quat_tensor)
        scale_tensor = torch.tensor(scale, dtype=torch.float32).reshape(1, -1)
        if scale_tensor.shape[-1] == 1:
            scale_tensor = scale_tensor.repeat(1, 3)
        translation_tensor = torch.tensor(translation, dtype=torch.float32).reshape(1, 3)
        pose_transform = (
            Transform3d(dtype=torch.float32)
            .scale(scale_tensor)
            .rotate(R_sam3d)
            .translate(translation_tensor)
        )
        
        def transform_to_pytorch3d_camera(vertices):
            """
            将 SAM3D canonical space 的顶点变换到 PyTorch3D 相机空间。
            
            步骤：
            1. 将 canonical 顶点从 Z-up 旋转到 Y-up (SAM3D 内部处理)
            2. 应用 SAM3D 的 pose (scale, rotation, translation)
            """
            # 1. Z-up to Y-up rotation
            v_rotated = vertices @ z_up_to_y_up_matrix.T
            pts_local = torch.from_numpy(v_rotated).float().unsqueeze(0)
            pts_world = pose_transform.transform_points(pts_local).squeeze(0).numpy()
            return pts_world
        
        # 创建合并场景
        merged_scene = trimesh.Scene()
        
        # 变换并添加 SAM3D 物体
        if isinstance(sam3d_scene, trimesh.Scene):
            for name, geom in sam3d_scene.geometry.items():
                if hasattr(geom, 'vertices'):
                    geom_copy = geom.copy()
                    geom_copy.vertices = transform_to_pytorch3d_camera(geom_copy.vertices)
                    merged_scene.add_geometry(geom_copy, node_name=f"sam3d_{name}")
                else:
                    merged_scene.add_geometry(geom, node_name=f"sam3d_{name}")
        else:
            if hasattr(sam3d_scene, 'vertices'):
                sam3d_scene.vertices = transform_to_pytorch3d_camera(sam3d_scene.vertices)
            merged_scene.add_geometry(sam3d_scene, node_name="sam3d_object")
        
        # 从输入的 pointmap 创建点云 (已经在 PyTorch3D 相机空间)
        # input_pointmap shape: (3, H, W) 或 (1, 3, H, W)
        pm_np = input_pointmap
        if torch.is_tensor(pm_np):
            pm_tensor = pm_np.detach().cpu()
        else:
            pm_tensor = torch.from_numpy(pm_np).float()
            
        # 去掉 batch 维度
        while pm_tensor.ndim > 3:
            pm_tensor = pm_tensor[0]
        
        # 转成 (3, H, W)
        if pm_tensor.ndim == 3 and pm_tensor.shape[0] != 3:
            pm_tensor = pm_tensor.permute(2, 0, 1)
        
        # 反归一化（如有需要）
        if pointmap_scale is not None and pointmap_shift is not None:
            normalizer = SSIPointmapNormalizer()
            scale_t = torch.as_tensor(pointmap_scale).float().view(-1)
            shift_t = torch.as_tensor(pointmap_shift).float().view(-1)
            pm_tensor = normalizer.denormalize(pm_tensor, scale_t, shift_t)
        
        pm_np = pm_tensor.permute(1, 2, 0).numpy()
        H, W = pm_np.shape[:2]
        
        # 获取颜色（从原始图像）
        colors = None
        if input_image is not None:
            from PIL import Image as PILImage
            if hasattr(input_image, 'convert'):
                # PIL Image
                img_np = np.array(input_image.convert("RGB"))
            else:
                # numpy array
                img_np = input_image
                if img_np.shape[-1] == 4:
                    img_np = img_np[..., :3]
            # Resize image to match pointmap resolution if needed
            if img_np.shape[:2] != (H, W):
                img_pil = PILImage.fromarray(img_np.astype(np.uint8))
                img_pil_resized = img_pil.resize((W, H), PILImage.BILINEAR)
                img_np = np.array(img_pil_resized)
            colors = img_np.reshape(-1, 3)
        
        # 过滤掉无效点 (NaN, Inf)
        valid_mask = np.all(np.isfinite(pm_np), axis=-1)
        pm_points = pm_np[valid_mask].reshape(-1, 3)
        
        if colors is not None:
            colors = colors.reshape(H, W, 3)[valid_mask].reshape(-1, 3)
        else:
            # 默认灰色
            colors = np.full((len(pm_points), 3), 128, dtype=np.uint8)
        
        # 下采样
        if len(pm_points) > 100000:
            step = len(pm_points) // 100000
            pm_points = pm_points[::step]
            colors = colors[::step]
        
        # 创建点云
        point_cloud = trimesh.points.PointCloud(vertices=pm_points, colors=colors)
        merged_scene.add_geometry(point_cloud, node_name="input_pointcloud")
        
        logger.info(f"[Overlay] Points in pointcloud: {len(pm_points)}")
        
        # 导出
        merged_scene.export(str(output_path))
        logger.info(f"✓ Overlay GLB saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Failed to create overlay GLB: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_image_names(image_names_str: Optional[str]) -> Optional[List[str]]:
    """Parse image names string."""
    if image_names_str is None or image_names_str == "":
        return None
    names = [x.strip() for x in image_names_str.split(",") if x.strip()]
    return names if names else None


def parse_attention_layers(layers_str: Optional[str]) -> Optional[List[int]]:
    """Parse attention layer indices from CLI string."""
    if layers_str is None:
        return None
    tokens = [token.strip() for token in layers_str.split(",") if token.strip()]
    if not tokens:
        return None
    indices: List[int] = []
    for token in tokens:
        try:
            indices.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid attention layer index: {token}") from exc
    return indices


def get_output_dir(
    input_path: Path, 
    mask_prompt: Optional[str] = None, 
    image_names: Optional[List[str]] = None,
    is_single_view: bool = False,
    use_weighting: bool = True,
    entropy_alpha: float = 30.0,
) -> Path:
    """Create output directory based on input path and parameters."""
    visualization_dir = Path("visualization")
    visualization_dir.mkdir(exist_ok=True)
    
    if mask_prompt:
        dir_name = mask_prompt
    else:
        dir_name = input_path.name if input_path.is_dir() else input_path.parent.name
    
    # Add weighting suffix with alpha value
    if use_weighting:
        # Format alpha: remove trailing zeros, e.g., 5.0 -> "5", 5.5 -> "5.5"
        alpha_str = f"{entropy_alpha:g}"
        suffix = f"_weighted_a{alpha_str}"
    else:
        suffix = "_avg"
    
    if is_single_view:
        if image_names and len(image_names) == 1:
            safe_name = image_names[0].replace("/", "_").replace("\\", "_")
            dir_name = f"{dir_name}_{safe_name}{suffix}"
        else:
            dir_name = f"{dir_name}_single{suffix}"
    elif image_names:
        if len(image_names) == 1:
            safe_name = image_names[0].replace("/", "_").replace("\\", "_")
            dir_name = f"{dir_name}_{safe_name}{suffix}"
        else:
            safe_names = [name.replace("/", "_").replace("\\", "_") for name in image_names]
            dir_name = f"{dir_name}_{'_'.join(safe_names[:3])}{suffix}"
            if len(safe_names) > 3:
                dir_name += f"_and_{len(safe_names)-3}_more"
    else:
        dir_name = f"{dir_name}_multiview{suffix}"
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{dir_name}_{timestamp}"
    
    output_dir = visualization_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    return output_dir


def run_weighted_inference(
    input_path: Path,
    mask_prompt: Optional[str] = None,
    image_names: Optional[List[str]] = None,
    seed: int = 42,
    stage1_steps: int = 50,
    stage2_steps: int = 25,
    decode_formats: List[str] = None,
    model_tag: str = "hf",
    # Weighting parameters
    use_weighting: bool = True,
    entropy_alpha: float = 30.0,
    attention_layer: int = 6,
    attention_step: int = 0,
    min_weight: float = 0.01,
    # Visualization
    visualize_weights: bool = False,
    save_attention: bool = False,
    attention_layers_to_save: Optional[List[int]] = None,
    save_coords: bool = True,  # Default True for weighted inference
    # Stage 2 init saving (for iteration stability analysis)
    save_stage2_init: bool = False,
    # External pointmap (from DA3 etc.)
    da3_output_path: Optional[str] = None,
    # GLB merge visualization
    merge_da3_glb: bool = False,
    # Overlay visualization
    overlay_pointmap: bool = False,
):
    """
    Run weighted inference.
    
    Args:
        input_path: Input path
        mask_prompt: Mask folder name
        image_names: List of image names
        seed: Random seed
        stage1_steps: Stage 1 inference steps
        stage2_steps: Stage 2 inference steps
        decode_formats: List of decode formats
        model_tag: Model tag
        use_weighting: Whether to use entropy-based weighting (default True)
        entropy_alpha: Gibbs temperature for entropy weighting
        attention_layer: Which layer to use for weight computation
        attention_step: Which step to use for weight computation
        min_weight: Minimum weight to prevent complete zeroing
        visualize_weights: Whether to save weight visualizations
        save_attention: Whether to save attention weights
        attention_layers_to_save: Which layers to save attention for
        save_coords: Whether to save 3D coordinates
        save_stage2_init: Whether to save Stage 2 initial latent for stability analysis
        da3_output_path: Path to DA3 output npz file (from run_da3.py)
            If provided, will use external pointmaps instead of internal depth model
    """
    config_path = f"checkpoints/{model_tag}/pipeline.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    
    logger.info(f"Loading model: {config_path}")
    inference = Inference(config_path, compile=False)
    
    if hasattr(inference._pipeline, 'rendering_engine'):
        if inference._pipeline.rendering_engine != "pytorch3d":
            logger.warning(f"Rendering engine is set to {inference._pipeline.rendering_engine}, changing to pytorch3d")
            inference._pipeline.rendering_engine = "pytorch3d"
    
    logger.info(f"Loading data: {input_path}")
    if mask_prompt:
        logger.info(f"Mask prompt: {mask_prompt}")
    
    view_images, view_masks = load_images_and_masks_from_path(
        input_path=input_path,
        mask_prompt=mask_prompt,
        image_names=image_names,
    )
    
    num_views = len(view_images)
    logger.info(f"Successfully loaded {num_views} views")
    
    # Load external pointmaps from DA3 if provided
    view_pointmaps = None
    da3_dir = None  # DA3 output directory (for GLB merge)
    da3_extrinsics = None  # Camera extrinsics for alignment
    da3_pointmaps = None  # Raw pointmaps for alignment visualization
    if da3_output_path is not None:
        da3_path = Path(da3_output_path)
        da3_dir = da3_path.parent  # Store the directory for potential GLB merge
        
        # Strict mode: if da3_output is specified, it MUST be used successfully
        # Otherwise, raise an error to help debug issues
        
        if not da3_path.exists():
            raise FileNotFoundError(
                f"DA3 output file not found: {da3_path}\n"
                f"Please run: python scripts/run_da3.py --image_dir <your_image_dir> --output_dir <output_dir>"
            )
        
        logger.info(f"Loading external pointmaps from DA3: {da3_path}")
        da3_data = np.load(da3_path)
        
        # Check if pointmaps_sam3d exists
        if "pointmaps_sam3d" not in da3_data:
            raise ValueError(
                f"No 'pointmaps_sam3d' found in DA3 output: {da3_path}\n"
                f"Available keys: {list(da3_data.keys())}\n"
                f"Please regenerate DA3 output with the latest run_da3.py script."
            )
        
        da3_pointmaps = da3_data["pointmaps_sam3d"]
        logger.info(f"  DA3 pointmaps shape: {da3_pointmaps.shape}")
        
        # Load extrinsics for alignment
        if "extrinsics" in da3_data:
            da3_extrinsics = da3_data["extrinsics"]
            logger.info(f"  DA3 extrinsics shape: {da3_extrinsics.shape}")
        
        # Check if number of pointmaps matches number of views
        if da3_pointmaps.shape[0] < num_views:
            raise ValueError(
                f"DA3 pointmap count mismatch!\n"
                f"  DA3 has {da3_pointmaps.shape[0]} pointmaps\n"
                f"  But inference needs {num_views} views\n"
                f"  DA3 output: {da3_path}\n"
                f"Please ensure DA3 was run on the SAME images you're using for inference.\n"
                f"Run: python scripts/run_da3.py --image_dir <correct_image_dir> --output_dir <output_dir>"
            )
        elif da3_pointmaps.shape[0] > num_views:
            # If DA3 has more pointmaps, use first N (this is acceptable)
            logger.warning(f"  DA3 has {da3_pointmaps.shape[0]} pointmaps but only {num_views} views, using first {num_views}")
            view_pointmaps = [da3_pointmaps[i] for i in range(num_views)]
        else:
            # Exact match
            view_pointmaps = [da3_pointmaps[i] for i in range(num_views)]
        
        logger.info(f"  Successfully loaded {num_views} external pointmaps from DA3")
    
    is_single_view = num_views == 1
    
    if is_single_view:
        logger.warning("Single view detected - weighting is not applicable, using standard inference")
        use_weighting = False
    
    output_dir = get_output_dir(input_path, mask_prompt, image_names, is_single_view, use_weighting, entropy_alpha)
    
    # Setup logging
    log_file = output_dir / "inference.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
    )
    
    decode_formats = decode_formats or ["gaussian", "mesh"]
    
    # Setup weighting config
    weighting_config = WeightingConfig(
        use_entropy=use_weighting,
        entropy_alpha=entropy_alpha,
        attention_layer=attention_layer,
        attention_step=attention_step,
        min_weight=min_weight,
    )
    
    logger.info(f"Weighting config: use_weighting={use_weighting}, alpha={entropy_alpha}, "
                f"layer={attention_layer}, step={attention_step}, min_weight={min_weight}")
    
    # Setup attention logger (only if explicitly requested for analysis)
    attention_logger: Optional[CrossAttentionLogger] = None
    if save_attention:
        # Only save attention when explicitly requested (for analysis purposes)
        layers_to_hook = attention_layers_to_save or [attention_layer]
        if attention_layer not in layers_to_hook:
            layers_to_hook.append(attention_layer)
        
        attention_dir = output_dir / "attention"
        attention_logger = CrossAttentionLogger(
            attention_dir,
            enabled_stages=["slat"],
            layer_indices=layers_to_hook,
            save_coords=save_coords,
        )
        attention_logger.attach_to_pipeline(inference._pipeline)
        logger.info(f"Cross-attention logging enabled → layers={layers_to_hook}, save_coords={save_coords}")
    
    # Note: Weighting uses in-memory AttentionCollector, not CrossAttentionLogger
    # The attention for weight computation is collected directly during warmup pass
    
    # Run inference
    if is_single_view:
        logger.info("Single-view inference mode")
        image = view_images[0]
        mask = view_masks[0] if view_masks else None
        result = inference._pipeline.run(
            image,
            mask,
            seed=seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
            decode_formats=decode_formats,
            attention_logger=attention_logger,
        )
        weight_manager = None
    else:
        logger.info(f"Multi-view inference mode ({'weighted' if use_weighting else 'average'})")
        if view_pointmaps is not None:
            logger.info(f"Using external pointmaps from DA3")
        result = inference._pipeline.run_multi_view(
            view_images=view_images,
            view_masks=view_masks,
            view_pointmaps=view_pointmaps,  # External pointmaps from DA3
            seed=seed,
            mode="multidiffusion",
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
            decode_formats=decode_formats,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
            attention_logger=attention_logger,
            # Pass weighting config for weighted fusion
            weighting_config=weighting_config if use_weighting else None,
            # Save Stage 2 init for stability analysis
            save_stage2_init=save_stage2_init,
            save_stage2_init_path=output_dir / "stage2_init.pt" if save_stage2_init else None,
        )
        weight_manager = result.get("weight_manager")
        
        # Log if stage2_init was saved
        if save_stage2_init and (output_dir / "stage2_init.pt").exists():
            logger.info(f"Stage 2 initial latent saved to: {output_dir / 'stage2_init.pt'}")
    
    # Save results
    saved_files = []
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Mode: {'Weighted' if use_weighting else 'Average'} fusion")
    print(f"Generated coordinates: {result['coords'].shape[0] if 'coords' in result else 'N/A'}")
    print(f"{'='*60}")
    
    glb_path = None
    if 'glb' in result and result['glb'] is not None:
        glb_path = output_dir / "result.glb"
        result['glb'].export(str(glb_path))
        saved_files.append("result.glb")
        print(f"✓ GLB file saved to: {glb_path}")
        
        # Merge with DA3 scene.glb if requested (with alignment)
        if merge_da3_glb and da3_dir is not None:
            # Prepare pose parameters for alignment
            # 注意：SAM3D 的 pose 参数已经是真实世界尺度了
            sam3d_pose = {}
            if 'scale' in result:
                sam3d_pose['scale'] = result['scale'].cpu().numpy() if torch.is_tensor(result['scale']) else result['scale']
            if 'rotation' in result:
                sam3d_pose['rotation'] = result['rotation'].cpu().numpy() if torch.is_tensor(result['rotation']) else result['rotation']
            if 'translation' in result:
                sam3d_pose['translation'] = result['translation'].cpu().numpy() if torch.is_tensor(result['translation']) else result['translation']
            
            if sam3d_pose:
                # Merge with DA3's complete scene.glb
                merged_path = merge_glb_with_da3_aligned(
                    glb_path, da3_dir, sam3d_pose
                )
                if merged_path:
                    saved_files.append(merged_path.name)
                    print(f"✓ Merged GLB with DA3 scene saved to: {merged_path}")
            else:
                logger.warning("Cannot align: missing SAM3D pose parameters")
        elif merge_da3_glb and da3_dir is None:
            logger.warning("--merge_da3_glb specified but no DA3 output directory available (need --da3_output)")
        
        # Overlay SAM3D result on input pointmap for pose visualization
        # 只叠加到实际使用的 pointmap 上
        if overlay_pointmap:
            sam3d_pose = {}
            if 'scale' in result:
                sam3d_pose['scale'] = result['scale'].cpu().numpy() if torch.is_tensor(result['scale']) else result['scale']
            if 'rotation' in result:
                sam3d_pose['rotation'] = result['rotation'].cpu().numpy() if torch.is_tensor(result['rotation']) else result['rotation']
            if 'translation' in result:
                sam3d_pose['translation'] = result['translation'].cpu().numpy() if torch.is_tensor(result['translation']) else result['translation']
            
            if sam3d_pose:
                pointmap_data = None
                pm_scale_np = None
                pm_shift_np = None
                
                if 'raw_view_pointmaps' in result and result['raw_view_pointmaps']:
                    pointmap_data = result['raw_view_pointmaps'][0]
                    logger.info("[Overlay] Using raw_view_pointmaps[0] (metric)")
                elif 'pointmap' in result:
                    pointmap_data = result['pointmap']
                    logger.info("[Overlay] Using result['pointmap'] (metric)")
                elif 'view_ss_input_dicts' in result and result['view_ss_input_dicts']:
                    internal_pm = result['view_ss_input_dicts'][0].get('pointmap')
                    if internal_pm is not None:
                        pointmap_data = internal_pm
                        logger.info("[Overlay] Using normalized pointmap from view_ss_input_dicts")
                    # 尝试从 per-view 输入中读取 scale/shift
                    pm_scale = result['view_ss_input_dicts'][0].get('pointmap_scale')
                    pm_shift = result['view_ss_input_dicts'][0].get('pointmap_shift')
                    if pm_scale is not None:
                        pm_scale_np = pm_scale.detach().cpu().numpy() if torch.is_tensor(pm_scale) else np.array(pm_scale)
                    if pm_shift is not None:
                        pm_shift_np = pm_shift.detach().cpu().numpy() if torch.is_tensor(pm_shift) else np.array(pm_shift)
                else:
                    logger.warning("Overlay: no pointmap source found")
                
                if pointmap_data is not None:
                    overlay_path = overlay_sam3d_on_pointmap(
                        glb_path,
                        pointmap_data,
                        sam3d_pose,
                        input_image=view_images[0] if view_images else None,
                        output_path=None,
                        pointmap_scale=pm_scale_np,
                        pointmap_shift=pm_shift_np,
                    )
                    if overlay_path:
                        saved_files.append(overlay_path.name)
                        print(f"✓ Overlay saved to: {overlay_path}")
                else:
                    logger.warning("Cannot create overlay: missing input pointmap")
    
    if 'gs' in result:
        output_path = output_dir / "result.ply"
        result['gs'].save_ply(str(output_path))
        saved_files.append("result.ply")
        print(f"✓ Gaussian Splatting (PLY) saved to: {output_path}")
    elif 'gaussian' in result:
        if isinstance(result['gaussian'], list) and len(result['gaussian']) > 0:
            output_path = output_dir / "result.ply"
            result['gaussian'][0].save_ply(str(output_path))
            saved_files.append("result.ply")
            print(f"✓ Gaussian Splatting (PLY) saved to: {output_path}")
    
    # Save pose and geometry parameters
    # These are important for converting from canonical space to metric/camera space
    # Reference: https://github.com/Stability-AI/stable-point-aware-3d/issues/XXX
    # - translation, rotation, scale: transform from canonical ([-0.5, 0.5]) to camera/metric space
    # - pointmap_scale: the scale factor used to normalize the pointmap (needed for real-world alignment)
    params = {}
    
    # Pose parameters
    if 'translation' in result:
        params['translation'] = result['translation'].cpu().numpy() if torch.is_tensor(result['translation']) else result['translation']
    if 'rotation' in result:
        params['rotation'] = result['rotation'].cpu().numpy() if torch.is_tensor(result['rotation']) else result['rotation']
    if 'scale' in result:
        params['scale'] = result['scale'].cpu().numpy() if torch.is_tensor(result['scale']) else result['scale']
    if 'downsample_factor' in result:
        params['downsample_factor'] = float(result['downsample_factor']) if torch.is_tensor(result['downsample_factor']) else result['downsample_factor']
    
    # Pointmap normalization parameters (for real-world alignment)
    if 'pointmap_scale' in result and result['pointmap_scale'] is not None:
        params['pointmap_scale'] = result['pointmap_scale'].cpu().numpy() if torch.is_tensor(result['pointmap_scale']) else result['pointmap_scale']
    if 'pointmap_shift' in result and result['pointmap_shift'] is not None:
        params['pointmap_shift'] = result['pointmap_shift'].cpu().numpy() if torch.is_tensor(result['pointmap_shift']) else result['pointmap_shift']
    
    # Geometry parameters
    if 'coords' in result:
        params['coords'] = result['coords'].cpu().numpy() if torch.is_tensor(result['coords']) else result['coords']
    
    if params:
        params_path = output_dir / "params.npz"
        np.savez(params_path, **params)
        saved_files.append("params.npz")
        print(f"✓ Parameters saved to: {params_path}")
    
    # 保存原始 pose（解码前）用于调试
    if 'all_view_poses_raw' in result:
        raw_poses = result['all_view_poses_raw']
        raw_poses_npz = {}
        for key, tensor in raw_poses.items():
            if torch.is_tensor(tensor):
                raw_poses_npz[key] = tensor.cpu().numpy()
            else:
                raw_poses_npz[key] = np.array(tensor)
        raw_poses_path = output_dir / "all_view_poses_raw.npz"
        np.savez(raw_poses_path, **raw_poses_npz)
        logger.info(f"Raw poses saved to: {raw_poses_path}")
        
        # 比较 View 0 的原始 pose 和 ss_return_dict 中的 pose
        logger.info("=" * 60)
        logger.info("Comparing View 0 raw pose vs final output (before decoding):")
        logger.info("=" * 60)
        for key in ['scale', 'translation', '6drotation', '6drotation_normalized', 'translation_scale']:
            if key in raw_poses:
                raw_v0 = raw_poses[key][0] if len(raw_poses[key].shape) > 1 else raw_poses[key]
                if torch.is_tensor(raw_v0):
                    raw_v0 = raw_v0.cpu().numpy()
                final_v = result.get(key, None)
                if final_v is not None:
                    if torch.is_tensor(final_v):
                        final_v = final_v.cpu().numpy()
                    logger.info(f"  {key}:")
                    logger.info(f"    View 0 raw:  {raw_v0.flatten()[:6]}")
                    logger.info(f"    Final (V0):  {final_v.flatten()[:6]}")
                    if np.allclose(raw_v0.flatten(), final_v.flatten(), atol=1e-5):
                        logger.info(f"    ✓ MATCH")
                    else:
                        logger.info(f"    ✗ MISMATCH! diff={np.abs(raw_v0.flatten() - final_v.flatten())[:6]}")
    
    # 保存所有视角的解码后 pose（如果有的话）
    if 'all_view_poses_decoded' in result:
        all_poses_decoded = result['all_view_poses_decoded']
        import json
        
        # 保存为 JSON 格式（方便查看）
        all_poses_json = {
            "num_views": len(all_poses_decoded),
            "note": "Each view decoded with its OWN pointmap_scale/shift. If predictions are correct, metric poses should be consistent.",
            "views": []
        }
        for view_idx, pose in enumerate(all_poses_decoded):
            view_data = {"view_idx": view_idx}
            for key, value in pose.items():
                if isinstance(value, np.ndarray):
                    view_data[key] = value.tolist()
                else:
                    view_data[key] = value
            all_poses_json["views"].append(view_data)
        
        all_poses_path = output_dir / "all_view_poses_decoded.json"
        with open(all_poses_path, 'w') as f:
            json.dump(all_poses_json, f, indent=2)
        saved_files.append("all_view_poses_decoded.json")
        print(f"✓ All view poses (decoded) saved to: {all_poses_path}")
        
        # 也保存为 npz 格式（方便程序读取）
        all_poses_npz = {}
        for view_idx, pose in enumerate(all_poses_decoded):
            for key, value in pose.items():
                npz_key = f"view{view_idx}_{key}"
                if isinstance(value, np.ndarray):
                    all_poses_npz[npz_key] = value
                elif isinstance(value, (list, tuple)):
                    all_poses_npz[npz_key] = np.array(value)
        all_poses_npz["num_views"] = len(all_poses_decoded)
        all_poses_npz_path = output_dir / "all_view_poses_decoded.npz"
        np.savez(all_poses_npz_path, **all_poses_npz)
        
        # 打印每个视角的解码后 pose
        logger.info("=" * 60)
        logger.info("All view poses (DECODED, each with its own pointmap_scale):")
        logger.info("=" * 60)
        for view_idx, pose in enumerate(all_poses_decoded):
            logger.info(f"  View {view_idx}:")
            if 'translation' in pose:
                t = pose['translation']
                logger.info(f"    translation: {t.flatten()[:3] if isinstance(t, np.ndarray) else t}")
            if 'rotation' in pose:
                r = pose['rotation']
                logger.info(f"    rotation (wxyz): {r.flatten()[:4] if isinstance(r, np.ndarray) else r}")
            if 'scale' in pose:
                s = pose['scale']
                logger.info(f"    scale: {s.flatten()[:3] if isinstance(s, np.ndarray) else s}")
            if 'pointmap_scale' in pose:
                ps = pose['pointmap_scale']
                logger.info(f"    pointmap_scale: {ps.flatten()[:3] if isinstance(ps, np.ndarray) else ps}")
        logger.info("=" * 60)
        
        # 创建多视角 pose 一致性可视化
        if da3_extrinsics is not None and glb_path.exists() and merge_da3_glb:
            try:
                multiview_glb_path = visualize_multiview_pose_consistency(
                    sam3d_glb_path=glb_path,
                    all_view_poses_decoded=all_poses_decoded,
                    da3_extrinsics=da3_extrinsics,
                    da3_scene_glb_path=da3_dir / "scene.glb" if da3_dir else None,
                    output_path=output_dir / "multiview_pose_consistency.glb",
                )
                if multiview_glb_path:
                    saved_files.append("multiview_pose_consistency.glb")
                    print(f"✓ Multi-view pose consistency visualization saved to: {multiview_glb_path}")
            except Exception as e:
                logger.warning(f"Failed to create multiview visualization: {e}")
                import traceback
                traceback.print_exc()
        
        # Print the pose values for reference
        logger.info("Pose parameters (canonical -> camera/metric space):")
        if 'translation' in params:
            logger.info(f"  Translation: {params['translation']}")
        if 'rotation' in params:
            logger.info(f"  Rotation (quaternion): {params['rotation']}")
        if 'scale' in params:
            logger.info(f"  Scale (in normalized space): {params['scale']}")
        if 'pointmap_scale' in params:
            logger.info(f"  Pointmap scale (normalization factor): {params['pointmap_scale']}")
        if 'pointmap_shift' in params:
            logger.info(f"  Pointmap shift: {params['pointmap_shift']}")
        if 'downsample_factor' in params:
            logger.info(f"  Downsample factor: {params['downsample_factor']}")
        if 'coords' in params:
            logger.info(f"  Coords shape: {params['coords'].shape}")
    
    print(f"\n{'='*60}")
    print(f"All output files saved to: {output_dir}")
    print(f"Saved files: {', '.join(saved_files)}")
    print(f"{'='*60}")
    
    if attention_logger is not None:
        attention_logger.close()
    
    # Save weighting analysis if enabled
    if weight_manager is not None and visualize_weights:
        logger.info("Saving weight visualizations...")
        
        # Save weights and visualizations
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_data = weight_manager.get_analysis_data()
        weights_downsampled = analysis_data.get("weights", {})  # 降采样维度的权重
        weights_expanded = analysis_data.get("expanded_weights", {})  # 扩展后的权重
        entropy_per_view = analysis_data.get("entropy_per_view", {})
        original_coords = analysis_data.get("original_coords")  # 原始 coords
        downsampled_coords = analysis_data.get("downsampled_coords")  # 降采样 coords
        downsample_idx = analysis_data.get("downsample_idx")  # idx 映射
        
        # Log dimension info
        if weights_downsampled:
            sample_w = list(weights_downsampled.values())[0]
            logger.info(f"Downsampled weights dimension: {sample_w.shape[0]}")
        if weights_expanded:
            sample_w = list(weights_expanded.values())[0]
            logger.info(f"Expanded weights dimension: {sample_w.shape[0]}")
        if original_coords is not None:
            logger.info(f"Original coords shape: {original_coords.shape}")
        if downsampled_coords is not None:
            logger.info(f"Downsampled coords shape: {downsampled_coords.shape}")
        
        # Save weights as .pt file
        torch.save({
            "weights_downsampled": {k: v.cpu() for k, v in weights_downsampled.items()} if weights_downsampled else {},
            "weights_expanded": {k: v.cpu() for k, v in weights_expanded.items()} if weights_expanded else {},
            "entropy": {k: v.cpu() for k, v in entropy_per_view.items()} if entropy_per_view else {},
            "config": {
                "entropy_alpha": weighting_config.entropy_alpha,
                "attention_layer": weighting_config.attention_layer,
                "attention_step": weighting_config.attention_step,
            },
            "original_coords": original_coords.cpu() if original_coords is not None else None,
            "downsampled_coords": downsampled_coords.cpu() if downsampled_coords is not None else None,
            "downsample_idx": downsample_idx.cpu() if downsample_idx is not None else None,
        }, weights_dir / "fusion_weights.pt")
        
        logger.info(f"Saved fusion weights to {weights_dir / 'fusion_weights.pt'}")
        
        # ============ Weight Analysis ============
        analysis_log = weights_dir / "weight_analysis.log"
        with open(analysis_log, "w") as f:
            def log_analysis(msg):
                f.write(msg + "\n")
                logger.info(msg)
            
            log_analysis("=" * 60)
            log_analysis("Weight Analysis Report")
            log_analysis("=" * 60)
            log_analysis(f"Number of views: {len(weights_downsampled)}")
            log_analysis(f"Entropy alpha: {weighting_config.entropy_alpha}")
            log_analysis(f"Attention layer: {weighting_config.attention_layer}")
            log_analysis(f"Attention step: {weighting_config.attention_step}")
            
            # Entropy analysis
            if entropy_per_view:
                log_analysis("\n--- Entropy Analysis ---")
                entropy_values = []
                for view_idx, e in sorted(entropy_per_view.items()):
                    log_analysis(
                        f"  View {view_idx}: min={e.min():.4f}, max={e.max():.4f}, "
                        f"mean={e.mean():.4f}, std={e.std():.4f}"
                    )
                    entropy_values.append(e)
                
                # Cross-view entropy difference
                if len(entropy_values) > 1:
                    entropy_stack = torch.stack(entropy_values, dim=0)
                    view_std = entropy_stack.std(dim=0)
                    log_analysis(f"\n  Cross-view entropy std (per latent):")
                    log_analysis(f"    min={view_std.min():.4f}, max={view_std.max():.4f}, mean={view_std.mean():.4f}")
            
            # Weight analysis
            log_analysis("\n--- Weight Analysis (Downsampled) ---")
            for view_idx, w in sorted(weights_downsampled.items()):
                log_analysis(
                    f"  View {view_idx}: min={w.min():.6f}, max={w.max():.6f}, "
                    f"mean={w.mean():.6f}, std={w.std():.6f}"
                )
            
            # Check weight sum
            views = sorted(weights_downsampled.keys())
            weight_sum = sum(weights_downsampled[v] for v in views)
            log_analysis(f"\n  Weight sum: min={weight_sum.min():.4f}, max={weight_sum.max():.4f}")
            
            # Cross-view weight difference
            weight_stack = torch.stack([weights_downsampled[v] for v in views], dim=0)
            view_std = weight_stack.std(dim=0)
            log_analysis(f"\n  Cross-view weight std (per latent):")
            log_analysis(f"    min={view_std.min():.6f}, max={view_std.max():.6f}, mean={view_std.mean():.6f}")
            
            # Find latents with most weight variation
            top_k = 5
            top_indices = torch.argsort(view_std, descending=True)[:top_k]
            log_analysis(f"\n  Top {top_k} latents with most weight variation:")
            for idx in top_indices:
                log_analysis(f"    Latent {idx.item()}: std={view_std[idx]:.4f}")
                for v in views:
                    log_analysis(f"      View {v}: {weights_downsampled[v][idx]:.4f}")
            
            log_analysis("\n" + "=" * 60)
        
        logger.info(f"Saved weight analysis to {analysis_log}")
        
        # Generate visualizations
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # numpy is already imported at module level as np
            
            # Weight distribution histogram (downsampled)
            if weights_downsampled:
                fig, axes = plt.subplots(1, len(weights_downsampled), figsize=(4 * len(weights_downsampled), 4))
                if len(weights_downsampled) == 1:
                    axes = [axes]
                
                for ax, (view_idx, w) in zip(axes, sorted(weights_downsampled.items())):
                    w_np = w.cpu().numpy()
                    ax.hist(w_np, bins=50, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Weight')
                    ax.set_ylabel('Count')
                    ax.set_title(f'View {view_idx} (downsampled)\nmean={w_np.mean():.4f}, std={w_np.std():.4f}')
                
                plt.tight_layout()
                plt.savefig(weights_dir / 'weight_distribution_downsampled.png', dpi=150)
                plt.close()
                logger.info("Saved downsampled weight distribution plot")
            
            # Weight distribution histogram (expanded)
            if weights_expanded:
                fig, axes = plt.subplots(1, len(weights_expanded), figsize=(4 * len(weights_expanded), 4))
                if len(weights_expanded) == 1:
                    axes = [axes]
                
                for ax, (view_idx, w) in zip(axes, sorted(weights_expanded.items())):
                    w_np = w.cpu().numpy()
                    ax.hist(w_np, bins=50, alpha=0.7, edgecolor='black', color='green')
                    ax.set_xlabel('Weight')
                    ax.set_ylabel('Count')
                    ax.set_title(f'View {view_idx} (expanded)\nmean={w_np.mean():.4f}, std={w_np.std():.4f}')
                
                plt.tight_layout()
                plt.savefig(weights_dir / 'weight_distribution_expanded.png', dpi=150)
                plt.close()
                logger.info("Saved expanded weight distribution plot")
            
            # Entropy distribution histogram
            if entropy_per_view:
                fig, axes = plt.subplots(1, len(entropy_per_view), figsize=(4 * len(entropy_per_view), 4))
                if len(entropy_per_view) == 1:
                    axes = [axes]
                
                for ax, (view_idx, e) in zip(axes, sorted(entropy_per_view.items())):
                    e_np = e.cpu().numpy()
                    ax.hist(e_np, bins=50, alpha=0.7, edgecolor='black', color='orange')
                    ax.set_xlabel('Entropy')
                    ax.set_ylabel('Count')
                    ax.set_title(f'View {view_idx}\nmean={e_np.mean():.4f}, std={e_np.std():.4f}')
                
                plt.tight_layout()
                plt.savefig(weights_dir / 'entropy_distribution.png', dpi=150)
                plt.close()
                logger.info("Saved entropy distribution plot")
            
            # 3D visualization with DOWNSAMPLED coords (where attention is computed)
            if downsampled_coords is not None and weights_downsampled:
                coords_np = downsampled_coords.cpu().numpy()
                x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
                
                # Normalize coordinates
                x = (x - x.min()) / (x.max() - x.min() + 1e-6)
                y = (y - y.min()) / (y.max() - y.min() + 1e-6)
                z = (z - z.min()) / (z.max() - z.min() + 1e-6)
                
                for view_idx, w in sorted(weights_downsampled.items()):
                    w_np = w.cpu().numpy()
                    
                    # Robust normalization
                    vmin, vmax = np.percentile(w_np, [2, 98])
                    w_norm = np.clip((w_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    
                    order = np.argsort(z)
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(
                        x[order], y[order], z[order],
                        c=w_norm[order],
                        cmap='viridis',
                        s=2,
                        alpha=0.6,
                    )
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'View {view_idx} Weight (Downsampled, {len(w_np)} points)')
                    
                    plt.colorbar(scatter, ax=ax, shrink=0.6, label='Weight')
                    plt.savefig(weights_dir / f'weight_3d_downsampled_view{view_idx:02d}.png', dpi=150)
                    plt.close()
                
                logger.info("Saved 3D weight visualizations (downsampled)")
            
            # 3D visualization with ORIGINAL coords (expanded weights)
            if original_coords is not None and weights_expanded:
                coords_np = original_coords.cpu().numpy()
                x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
                
                # Normalize coordinates
                x = (x - x.min()) / (x.max() - x.min() + 1e-6)
                y = (y - y.min()) / (y.max() - y.min() + 1e-6)
                z = (z - z.min()) / (z.max() - z.min() + 1e-6)
                
                for view_idx, w in sorted(weights_expanded.items()):
                    w_np = w.cpu().numpy()
                    
                    # Robust normalization
                    vmin, vmax = np.percentile(w_np, [2, 98])
                    w_norm = np.clip((w_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    
                    order = np.argsort(z)
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(
                        x[order], y[order], z[order],
                        c=w_norm[order],
                        cmap='viridis',
                        s=0.5,  # 更小的点，因为点更多
                        alpha=0.4,
                    )
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'View {view_idx} Weight (Expanded, {len(w_np)} points)')
                    
                    plt.colorbar(scatter, ax=ax, shrink=0.6, label='Weight')
                    plt.savefig(weights_dir / f'weight_3d_expanded_view{view_idx:02d}.png', dpi=150)
                    plt.close()
                
                logger.info("Saved 3D weight visualizations (expanded)")
                
        except ImportError as e:
            logger.warning(f"Could not generate visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects Weighted Inference - Per-latent weighted multi-view fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic weighted inference
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3
  
  # Disable weighting (simple average)
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 --no_weighting
  
  # With visualization
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 --visualize_weights
  
  # Custom parameters
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
      --entropy_alpha 3.0 --attention_layer 6
        """
    )
    
    # Input/Output
    parser.add_argument("--input_path", type=str, required=True, help="Input path")
    parser.add_argument("--mask_prompt", type=str, default=None, help="Mask folder name")
    parser.add_argument("--image_names", type=str, default=None, help="Image names (comma-separated)")
    parser.add_argument("--model_tag", type=str, default="hf", help="Model tag")
    
    # Inference parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stage1_steps", type=int, default=50, help="Stage 1 steps")
    parser.add_argument("--stage2_steps", type=int, default=25, help="Stage 2 steps")
    parser.add_argument("--decode_formats", type=str, default="gaussian,mesh", help="Decode formats")
    
    # Weighting parameters
    parser.add_argument("--no_weighting", action="store_true", 
                        help="Disable entropy-based weighting (use simple average)")
    parser.add_argument("--entropy_alpha", type=float, default=30.0,
                        help="Gibbs temperature for entropy weighting (higher = more contrast)")
    parser.add_argument("--attention_layer", type=int, default=6,
                        help="Which attention layer to use for weight computation")
    parser.add_argument("--attention_step", type=int, default=0,
                        help="Which diffusion step to use for weight computation")
    parser.add_argument("--min_weight", type=float, default=0.001,
                        help="Minimum weight to prevent complete zeroing")
    
    # Visualization
    parser.add_argument("--visualize_weights", action="store_true",
                        help="Save weight and entropy visualizations")
    parser.add_argument("--save_attention", action="store_true",
                        help="Save all attention weights (for analysis)")
    parser.add_argument("--attention_layers", type=str, default=None,
                        help="Which layers to save attention for (comma-separated)")
    
    # Stage 2 init saving (for iteration stability analysis)
    parser.add_argument("--save_stage2_init", action="store_true",
                        help="Save Stage 2 initial latent for iteration stability analysis")
    
    # External pointmap (from DA3)
    parser.add_argument("--da3_output", type=str, default=None,
                        help="Path to DA3 output npz file (from run_da3.py). "
                             "If provided, uses external pointmaps instead of internal depth model")
    
    # GLB merge visualization
    parser.add_argument("--merge_da3_glb", action="store_true",
                        help="Merge SAM3D output GLB with DA3 scene.glb for comparison (requires --da3_output)")
    
    # Overlay visualization - 将 SAM3D 结果叠加到输入 pointmap 上
    parser.add_argument("--overlay_pointmap", action="store_true",
                        help="Overlay SAM3D result on input pointmap for pose visualization. "
                             "Works with both MoGe (default) and DA3 (if --da3_output is provided)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    image_names = parse_image_names(args.image_names)
    decode_formats = [fmt.strip() for fmt in args.decode_formats.split(",") if fmt.strip()]
    
    try:
        run_weighted_inference(
            input_path=input_path,
            mask_prompt=args.mask_prompt,
            image_names=image_names,
            seed=args.seed,
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            decode_formats=decode_formats,
            model_tag=args.model_tag,
            use_weighting=not args.no_weighting,
            entropy_alpha=args.entropy_alpha,
            attention_layer=args.attention_layer,
            attention_step=args.attention_step,
            min_weight=args.min_weight,
            visualize_weights=args.visualize_weights,
            save_attention=args.save_attention,
            attention_layers_to_save=parse_attention_layers(args.attention_layers),
            save_stage2_init=args.save_stage2_init,
            da3_output_path=args.da3_output,
            merge_da3_glb=args.merge_da3_glb,
            overlay_pointmap=args.overlay_pointmap,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

