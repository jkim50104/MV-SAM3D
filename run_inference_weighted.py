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

# Import inference code
sys.path.append("notebook")
from inference import Inference
from load_images_and_masks import load_images_and_masks_from_path

from sam3d_objects.utils.cross_attention_logger import CrossAttentionLogger
from sam3d_objects.utils.latent_weighting import WeightingConfig, LatentWeightManager
from sam3d_objects.utils.coordinate_transforms import (
    Z_UP_TO_Y_UP,
    apply_sam3d_pose_to_mesh_vertices,
    apply_sam3d_pose_to_latent_coords,
    canonical_to_pytorch3d,
)
from pytorch3d.transforms import Transform3d, quaternion_to_matrix


def merge_glb_with_da3_aligned(
    sam3d_glb_path: Path, 
    da3_output_dir: Path,
    sam3d_pose: dict,
    output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Merge SAM3D reconstructed object with DA3's full scene GLB (aligned).
    
    DA3's scene.glb contains alignment matrix `hf_alignment` in metadata:
    A = T_center @ M @ w2c0, which includes:
    - w2c0: First frame's world-to-camera transform
    - M: CV -> glTF coordinate system transform
    - T_center: Centering translation
    
    SAM3D object transform chain:
    1. canonical (Z-up) -> Y-up rotation
    2. Apply SAM3D pose -> PyTorch3D camera space
    3. PyTorch3D -> CV camera space: (-x, -y, z) -> (x, y, z)
    4. Apply DA3's alignment matrix A (from scene.glb metadata)
    
    Args:
        sam3d_glb_path: SAM3D output GLB file path (canonical space)
        da3_output_dir: DA3 output directory containing scene.glb
        sam3d_pose: SAM3D pose parameters {'scale', 'rotation', 'translation'}
        output_path: Output path
    
    Returns:
        Aligned GLB file path
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed, cannot merge GLB files")
        return None
    
    if not sam3d_glb_path.exists():
        logger.warning(f"SAM3D GLB not found: {sam3d_glb_path}")
        return None
    
    # Find DA3's scene.glb
    da3_scene_glb = da3_output_dir / "scene.glb"
    da3_npz = da3_output_dir / "da3_output.npz"
    
    if not da3_scene_glb.exists():
        logger.warning(f"DA3 scene.glb not found: {da3_scene_glb}")
        logger.warning("Please run DA3 with visualization enabled")
        return None
    
    if output_path is None:
        output_path = sam3d_glb_path.parent / f"{sam3d_glb_path.stem}_merged_scene.glb"
    
    try:
        # Load SAM3D GLB
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # Load DA3 scene.glb
        da3_scene = trimesh.load(str(da3_scene_glb))
        
        # Try to read alignment matrix from DA3 scene metadata
        alignment_matrix = None
        if hasattr(da3_scene, 'metadata') and da3_scene.metadata is not None:
            alignment_matrix = da3_scene.metadata.get('hf_alignment', None)
        
        if alignment_matrix is None:
            logger.warning("DA3 scene.glb does not contain alignment matrix (hf_alignment)")
            logger.warning("Falling back to computing alignment from extrinsics")
            
            # Fallback: compute alignment from extrinsics
            if not da3_npz.exists():
                logger.warning(f"DA3 da3_output.npz not found: {da3_npz}")
                return None
            
            da3_data = np.load(da3_npz)
            da3_extrinsics = da3_data["extrinsics"]
            
            # Get first frame w2c
            w2c0 = da3_extrinsics[0]
            if w2c0.shape == (3, 4):
                w2c0_44 = np.eye(4, dtype=np.float64)
                w2c0_44[:3, :4] = w2c0
                w2c0 = w2c0_44
            
            # CV -> glTF coordinate transform
            M_cv_to_gltf = np.eye(4, dtype=np.float64)
            M_cv_to_gltf[1, 1] = -1.0
            M_cv_to_gltf[2, 2] = -1.0
            
            # Compute alignment matrix (without centering)
            A_no_center = M_cv_to_gltf @ w2c0
            
            # Get point cloud center from DA3 scene
            da3_points = []
            if isinstance(da3_scene, trimesh.Scene):
                for geom in da3_scene.geometry.values():
                    if hasattr(geom, 'vertices'):
                        da3_points.append(geom.vertices)
            elif hasattr(da3_scene, 'vertices'):
                da3_points.append(da3_scene.vertices)
            
            if da3_points:
                all_pts = np.vstack(da3_points)
                # DA3 scene is already centered
                # Since it is centered, center should be near 0
                # We need to compute original centering offset
                # This is complex, assume centering offset is 0 for now
                alignment_matrix = A_no_center
                logger.warning("Using alignment without centering (may be slightly off)")
        
        logger.info(f"[Merge Scene] Alignment matrix:\n{alignment_matrix}")
        
        # Extract SAM3D pose parameters
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
        # SAM3D object transform
        # ========================================
        
        # Z-up to Y-up rotation matrix
        z_up_to_y_up_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float32)
        
        # Build pose transform in PyTorch3D space
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
        
        # PyTorch3D to CV camera space transform
        p3d_to_cv = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        
        def transform_sam3d_to_da3_space(vertices):
            """
            Transform SAM3D canonical space vertices to DA3 scene space (glTF)
            """
            # Step 1: Z-up to Y-up
            v_rotated = vertices @ z_up_to_y_up_matrix.T
            
            # Step 2: Apply SAM3D pose -> PyTorch3D space
            pts_local = torch.from_numpy(v_rotated).float().unsqueeze(0)
            pts_p3d = pose_transform.transform_points(pts_local).squeeze(0).numpy()
            
            # Step 3: PyTorch3D -> CV camera space
            pts_cv = pts_p3d @ p3d_to_cv.T
            
            # Step 4: Apply DA3 alignment matrix
            pts_final = trimesh.transform_points(pts_cv, alignment_matrix)
            
            return pts_final
        
        # ========================================
        # Create merged scene
        # ========================================
        
        merged_scene = trimesh.Scene()
        
        # Add DA3 scene (keep as-is, already in correct coordinate system)
        if isinstance(da3_scene, trimesh.Scene):
            for name, geom in da3_scene.geometry.items():
                merged_scene.add_geometry(geom.copy(), node_name=f"da3_{name}")
        else:
            merged_scene.add_geometry(da3_scene.copy(), node_name="da3_scene")
        
        # Transform and add SAM3D object
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
        
        # Print alignment info
        if sam3d_vertices_final is not None:
            logger.info(f"[Merge Scene] SAM3D object in DA3 space:")
            logger.info(f"  X: [{sam3d_vertices_final[:, 0].min():.4f}, {sam3d_vertices_final[:, 0].max():.4f}]")
            logger.info(f"  Y: [{sam3d_vertices_final[:, 1].min():.4f}, {sam3d_vertices_final[:, 1].max():.4f}]")
            logger.info(f"  Z: [{sam3d_vertices_final[:, 2].min():.4f}, {sam3d_vertices_final[:, 2].max():.4f}]")
        
        # Print DA3 scene bounds
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
        
        # Export
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
    Visualize multi-view pose consistency: place each view's predicted object in world coordinates.
    
    If all views predict consistently, these objects should overlap.
    If inconsistent, can visually see which views deviate.
    
    Args:
        sam3d_glb_path: SAM3D output GLB file path (canonical space)
        all_view_poses_decoded: List of decoded poses for all views
        da3_extrinsics: DA3 camera extrinsics (N, 3, 4) or (N, 4, 4), world-to-camera
        da3_scene_glb_path: DA3 scene.glb path (optional, for adding scene background)
        output_path: Output path
    
    Returns:
        Visualization GLB file path
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
        # Load SAM3D GLB (canonical space)
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # Extract canonical vertices
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
        
        # Z-up to Y-up rotation matrix (same as merge_glb_with_da3_aligned)
        z_up_to_y_up_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=np.float32)
        
        # PyTorch3D to CV camera space transform
        p3d_to_cv = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        
        # CV to glTF coordinate transform
        M_cv_to_gltf = np.eye(4, dtype=np.float64)
        M_cv_to_gltf[1, 1] = -1.0
        M_cv_to_gltf[2, 2] = -1.0
        
        # Create scene
        merged_scene = trimesh.Scene()
        
        # If DA3 scene exists, add as background
        alignment_matrix = None
        if da3_scene_glb_path is not None and da3_scene_glb_path.exists():
            da3_scene = trimesh.load(str(da3_scene_glb_path))
            
            # Get alignment matrix
            if hasattr(da3_scene, 'metadata') and da3_scene.metadata is not None:
                alignment_matrix = da3_scene.metadata.get('hf_alignment', None)
            
            # Add DA3 scene (semi-transparent gray)
            if isinstance(da3_scene, trimesh.Scene):
                for name, geom in da3_scene.geometry.items():
                    geom_copy = geom.copy()
                    if hasattr(geom_copy, 'visual'):
                        geom_copy.visual.face_colors = [128, 128, 128, 100]
                    merged_scene.add_geometry(geom_copy, node_name=f"da3_{name}")
        
        # Create transformed object for each view
        colors_per_view = [
            [255, 0, 0, 200],     # View 0: Red
            [0, 255, 0, 200],     # View 1: Green
            [0, 0, 255, 200],     # View 2: Blue
            [255, 255, 0, 200],   # View 3: Yellow
            [255, 0, 255, 200],   # View 4: Magenta
            [0, 255, 255, 200],   # View 5: Cyan
            [255, 128, 0, 200],   # View 6: Orange
            [128, 0, 255, 200],   # View 7: Purple
        ]
        
        for view_idx, pose in enumerate(all_view_poses_decoded):
            # Extract pose parameters
            translation = np.array(pose.get('translation', [[0, 0, 0]])).flatten()[:3]
            rotation_quat = np.array(pose.get('rotation', [[1, 0, 0, 0]])).flatten()[:4]
            scale = np.array(pose.get('scale', [[1, 1, 1]])).flatten()[:3]
            
            # Build transform (same as merge_glb_with_da3_aligned)
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
            
            # Transform vertices
            # Step 1: Z-up to Y-up
            v_rotated = canonical_vertices @ z_up_to_y_up_matrix.T
            
            # Step 2: Apply SAM3D pose -> PyTorch3D space
            pts_local = torch.from_numpy(v_rotated).float().unsqueeze(0)
            pts_p3d = pose_transform.transform_points(pts_local).squeeze(0).numpy()
            
            # Step 3: PyTorch3D -> CV camera space
            pts_cv = pts_p3d @ p3d_to_cv.T
            
            # Step 4: View i camera space -> world coordinates
            w2c_i = da3_extrinsics[view_idx]
            if w2c_i.shape == (3, 4):
                w2c_i_44 = np.eye(4, dtype=np.float64)
                w2c_i_44[:3, :4] = w2c_i
                w2c_i = w2c_i_44
            c2w_i = np.linalg.inv(w2c_i)
            pts_world = trimesh.transform_points(pts_cv, c2w_i)
            
            # Step 5: World coordinates -> glTF coordinates
            pts_gltf = trimesh.transform_points(pts_world, M_cv_to_gltf)
            
            # Step 6: Apply centering offset if alignment matrix exists
            if alignment_matrix is not None and view_idx == 0:
                # Use View 0 to compute centering offset
                # Apply alignment_matrix to View 0 CV space points
                pts_aligned_v0 = trimesh.transform_points(pts_cv, alignment_matrix)
                center_offset = pts_aligned_v0.mean(axis=0) - pts_gltf.mean(axis=0)
            
            if alignment_matrix is not None:
                pts_final = pts_gltf + center_offset
            else:
                pts_final = pts_gltf
            
            # Filter invalid points
            valid = np.isfinite(pts_final).all(axis=1)
            pts_final = pts_final[valid]
            
            # Create mesh
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
        
        # Export
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


def convert_da3_extrinsics_to_camera_poses(
    da3_extrinsics: np.ndarray,
) -> List[dict]:
    """
    Convert DA3 extrinsics (world-to-camera) to camera_poses format.
    
    DA3 extrinsics are (N, 3, 4) or (N, 4, 4) w2c matrices.
    
    Args:
        da3_extrinsics: DA3 camera extrinsics, shape (N, 3, 4) or (N, 4, 4)
    
    Returns:
        List of camera pose dicts, each containing:
            - 'view_idx': int
            - 'c2w': (4, 4) camera-to-world matrix
            - 'w2c': (4, 4) world-to-camera matrix
            - 'R_c2w': (3, 3) rotation matrix
            - 't_c2w': (3,) translation vector
            - 'camera_position': (3,) camera position in world coordinates
    """
    num_views = da3_extrinsics.shape[0]
    camera_poses = []
    
    for view_idx in range(num_views):
        w2c_raw = da3_extrinsics[view_idx]  # (3, 4) or (4, 4)
        
        # Convert to (4, 4) format
        if w2c_raw.shape == (3, 4):
            w2c = np.eye(4)
            w2c[:3, :] = w2c_raw
        else:
            w2c = w2c_raw
        
        # Compute c2w = inv(w2c)
        c2w = np.linalg.inv(w2c)
        
        # Extract rotation and translation
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        camera_position = t_c2w  # Camera position is the translation part of c2w
        
        camera_poses.append({
            'view_idx': view_idx,
            'c2w': c2w,
            'w2c': w2c,
            'R_c2w': R_c2w,
            't_c2w': t_c2w,
            'camera_position': camera_position,
        })
    
    logger.info(f"[DA3 Extrinsics] Converted {num_views} extrinsics to camera poses")
    return camera_poses


def compute_camera_poses_from_object_poses(
    all_view_poses: List[dict],
) -> List[dict]:
    """
    Compute camera poses from object pose in each view's camera coordinate system.
    
    Assumptions:
    1. Object is stationary in world coordinates
    2. View 0 camera coordinate system is the world coordinate system
    
    Mathematical derivation (using 4x4 homogeneous transform matrices):
    
    Definitions:
    - M_obj_to_c0 = [R_0, T_0; 0, 1]: Object transform from canonical space to View 0 camera coordinates
    - M_obj_to_ci = [R_i, T_i; 0, 1]: Object transform from canonical space to View i camera coordinates
    
    Goal:
    - M_ci_to_c0: Transform from View i camera coordinates to View 0 camera coordinates (world)
      This is the camera-to-world (c2w) matrix
    
    Derivation:
    Using object coordinate system as bridge: Ci -> Object -> C0
    
    M_ci_to_c0 = M_obj_to_c0 @ inv(M_obj_to_ci)
    
    Expanding:
    - inv(M_obj_to_ci) = [R_i^T, -R_i^T @ T_i; 0, 1]
    - M_ci_to_c0 = [R_0, T_0; 0, 1] @ [R_i^T, -R_i^T @ T_i; 0, 1]
                = [R_0 @ R_i^T, R_0 @ (-R_i^T @ T_i) + T_0; 0, 1]
                = [R_0 @ R_i^T, T_0 - R_0 @ R_i^T @ T_i; 0, 1]
    
    Conclusion (camera-to-world):
    - R_c2w = R_0 @ R_i^T
    - T_c2w = T_0 - R_0 @ R_i^T @ T_i
    
    Args:
        all_view_poses: List of decoded poses for each view
            Each pose contains: translation (3,), rotation (4,) [wxyz quaternion], scale (3,)
    
    Returns:
        List of camera poses, each containing:
            - c2w: (4, 4) camera-to-world matrix
            - w2c: (4, 4) world-to-camera matrix
    """
    from scipy.spatial.transform import Rotation
    
    num_views = len(all_view_poses)
    
    # ========================================
    # Coordinate system notes
    # ========================================
    # SAM3D pose parameters (rotation, translation) are defined in PyTorch3D camera space (Y-up).
    # - Translation is in Y-up space
    # - Rotation quaternion is also defined in Y-up space
    # 
    # No additional coordinate conversion needed, use original quaternion.
    # ========================================
    
    # Extract View 0 pose as reference (defines world coordinate system)
    pose_0 = all_view_poses[0]
    T_0 = np.array(pose_0['translation']).flatten()[:3]
    quat_0 = np.array(pose_0['rotation']).flatten()[:4]  # wxyz
    # Convert quaternion from wxyz to xyzw (scipy format)
    quat_0_scipy = np.array([quat_0[1], quat_0[2], quat_0[3], quat_0[0]])
    R_0 = Rotation.from_quat(quat_0_scipy).as_matrix()
    
    logger.info(f"[Camera Pose] Reference (View 0 - Fixed):")
    logger.info(f"  T_0: {T_0}")
    logger.info(f"  R_0 euler (deg): {Rotation.from_matrix(R_0).as_euler('xyz', degrees=True)}")
    
    camera_poses = []
    
    # Process View 0 first (as reference/world coordinate system, camera pose should be identity)
    c2w_0 = np.eye(4)
    c2w_0[:3, :3] = np.eye(3)  # View 0 camera = world coordinate system
    c2w_0[:3, 3] = np.zeros(3)  # Camera position at origin
    
    camera_poses.append({
        'view_idx': 0,
        'c2w': c2w_0,
        'w2c': np.eye(4),  # w2c is also identity
        'R_c2w': np.eye(3),
        't_c2w': np.zeros(3),
        'camera_position': np.zeros(3),
    })
    
    logger.info(f"[Camera Pose] View 0:")
    logger.info(f"  Object pose: T={T_0}, R_euler={Rotation.from_matrix(R_0).as_euler('xyz', degrees=True)}")
    logger.info(f"  Camera position (world): [0, 0, 0] (reference)")
    
    # Process other views
    for view_idx in range(1, num_views):
        pose = all_view_poses[view_idx]
        
        T_i = np.array(pose['translation']).flatten()[:3]
        quat_i = np.array(pose['rotation']).flatten()[:4]  # wxyz
        
        # Convert quaternion from wxyz to xyzw (scipy format)
        quat_i_scipy = np.array([quat_i[1], quat_i[2], quat_i[3], quat_i[0]])
        R_i = Rotation.from_quat(quat_i_scipy).as_matrix()
        
        # Compute camera-to-world (c2w) transform
        # Formula: R_c2w = R_0 @ R_i^T, T_c2w = T_0 - R_0 @ R_i^T @ T_i
        R_c2w = R_0 @ R_i.T
        T_c2w = T_0 - R_0 @ R_i.T @ T_i
        
        # Build camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = T_c2w
        
        # Build world-to-camera matrix (inverse of c2w)
        w2c = np.linalg.inv(c2w)
        
        # Camera position in world coordinates is the translation part of c2w
        camera_position = T_c2w
        
        # Compute camera rotation angle (relative to View 0)
        rot_angle_deg = np.rad2deg(np.arccos(np.clip((np.trace(R_c2w) - 1) / 2, -1, 1)))
        
        camera_poses.append({
            'view_idx': view_idx,
            'c2w': c2w,
            'w2c': w2c,
            'R_c2w': R_c2w,
            't_c2w': T_c2w,
            'camera_position': camera_position,
        })
        
        logger.info(f"[Camera Pose] View {view_idx}:")
        logger.info(f"  Object pose: T={T_i}, R_euler={Rotation.from_quat(quat_i_scipy).as_euler('xyz', degrees=True)}")
        logger.info(f"  Camera c2w rotation angle from View 0: {rot_angle_deg:.1f} deg")
        logger.info(f"  Camera position (world): {camera_position}")
    
    return camera_poses


def create_camera_frustum(
    c2w: np.ndarray,
    scale: float = 0.1,
    color: List[int] = [255, 0, 0, 255],
):
    """
    Create camera frustum mesh for visualization.
    
    Camera coordinate system convention (PyTorch3D):
    - X: left
    - Y: up  
    - Z: forward (camera looking at +Z)
    
    Args:
        c2w: (4, 4) camera-to-world matrix
        scale: Frustum size
        color: RGBA color
    
    Returns:
        Camera frustum as trimesh.Trimesh
    """
    import trimesh
    
    # Camera frustum vertices (in camera coordinate system)
    # Camera at origin, looking at +Z direction
    h = scale  # Frustum height (along Z axis)
    w = scale * 0.6  # Frustum width
    
    # Frustum facing +Z
    vertices_cam = np.array([
        [0, 0, 0],           # 0: Camera center
        [-w, -w, h],         # 1: Bottom-left (far plane)
        [w, -w, h],          # 2: Bottom-right
        [w, w, h],           # 3: Top-right
        [-w, w, h],          # 4: Top-left
    ])
    
    # Transform to world coordinates
    vertices_world = (c2w[:3, :3] @ vertices_cam.T).T + c2w[:3, 3]
    
    # Define faces (triangles)
    faces = np.array([
        [0, 2, 1],  # Bottom triangle 1 (reversed winding for correct normals)
        [0, 3, 2],  # Bottom triangle 2
        [0, 4, 3],  # Bottom triangle 3
        [0, 1, 4],  # Bottom triangle 4
        [1, 2, 3],  # Far plane triangle 1
        [1, 3, 4],  # Far plane triangle 2
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces, process=False)
    mesh.visual.face_colors = color
    
    return mesh


def visualize_aligned_cameras_with_gt(
    sam3d_glb_path: Path,
    object_pose: dict,
    estimated_camera_poses: List[dict],
    gt_camera_poses: List[dict],
    output_path: Optional[Path] = None,
    camera_scale: float = 0.1,
) -> Optional[Path]:
    """
    Visualize aligned estimated and GT camera poses.
    
    Same view's prediction and GT use the same color:
    - GT camera: larger (1.5x), opaque
    - Predicted camera: smaller, semi-transparent
    - Error line: white
    
    Color scheme (by view):
    - View 0: Red
    - View 1: Green
    - View 2: Blue
    - View 3: Yellow
    - View 4: Magenta
    - View 5: Cyan
    - View 6: Orange
    - View 7: Purple
    
    Args:
        sam3d_glb_path: SAM3D output GLB file path
        object_pose: Object pose in world coordinates (View 0 camera coordinates)
        estimated_camera_poses: Aligned estimated camera poses
        gt_camera_poses: GT camera poses
        output_path: Output path
        camera_scale: Camera frustum size
    
    Returns:
        Output GLB file path
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
        output_path = sam3d_glb_path.parent / "result_cameras_aligned_with_gt.glb"
    
    try:
        # Load SAM3D GLB
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # Extract vertices
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
        
        # Use unified coordinate transform function
        # Transform chain: canonical (Z-up) -> Y-up -> scale -> rotate -> translate
        v_final = apply_sam3d_pose_to_mesh_vertices(canonical_vertices, object_pose, debug=True)
        
        logger.info(f"[Aligned Viz] Mesh world coords range: "
                    f"X=[{v_final[:, 0].min():.4f}, {v_final[:, 0].max():.4f}], "
                    f"Y=[{v_final[:, 1].min():.4f}, {v_final[:, 1].max():.4f}], "
                    f"Z=[{v_final[:, 2].min():.4f}, {v_final[:, 2].max():.4f}]")
        
        # Create scene
        merged_scene = trimesh.Scene()
        
        # Add object
        if canonical_faces is not None:
            obj_mesh = trimesh.Trimesh(vertices=v_final, faces=canonical_faces, process=False)
            obj_mesh.visual.face_colors = [200, 200, 200, 255]  # Gray
        else:
            obj_mesh = trimesh.PointCloud(v_final, colors=[200, 200, 200, 255])
        merged_scene.add_geometry(obj_mesh, node_name="object")
        
        # Add coordinate axes
        axis_length = camera_scale * 2
        axis_vertices = np.array([
            [0, 0, 0], [axis_length, 0, 0],  # X
            [0, 0, 0], [0, axis_length, 0],  # Y
            [0, 0, 0], [0, 0, axis_length],  # Z
        ])
        axis_colors = np.array([
            [255, 0, 0, 255], [255, 0, 0, 255],  # X - Red
            [0, 255, 0, 255], [0, 255, 0, 255],  # Y - Green
            [0, 0, 255, 255], [0, 0, 255, 255],  # Z - Blue
        ])
        axis_pc = trimesh.PointCloud(axis_vertices, colors=axis_colors)
        merged_scene.add_geometry(axis_pc, node_name="world_axes")
        
        # Color for each view (same as visualize_object_with_cameras)
        colors_per_view = [
            [255, 0, 0],       # View 0: Red
            [0, 255, 0],       # View 1: Green
            [0, 0, 255],       # View 2: Blue
            [255, 255, 0],     # View 3: Yellow
            [255, 0, 255],     # View 4: Magenta
            [0, 255, 255],     # View 5: Cyan
            [255, 128, 0],     # View 6: Orange
            [128, 0, 255],     # View 7: Purple
        ]
        
        # GT camera size multiplier and estimated camera size multiplier
        gt_scale_mult = 1.5      # GT larger
        est_scale_mult = 0.7     # Estimated smaller
        
        # Add GT cameras (opaque, larger, solid)
        for cam_pose in gt_camera_poses:
            view_idx = cam_pose['view_idx']
            c2w = np.array(cam_pose['c2w'])
            base_color = colors_per_view[view_idx % len(colors_per_view)]
            color = base_color + [255]  # Opaque
            
            frustum = create_camera_frustum(c2w, scale=camera_scale * gt_scale_mult, color=color)
            merged_scene.add_geometry(frustum, node_name=f"gt_camera_{view_idx}")
            
            # Add marker (large sphere)
            cam_pos = c2w[:3, 3]
            marker = trimesh.creation.icosphere(subdivisions=1, radius=camera_scale * 0.15)
            marker.apply_translation(cam_pos)
            marker.visual.face_colors = color
            merged_scene.add_geometry(marker, node_name=f"gt_marker_{view_idx}")
        
        # Add estimated cameras (semi-transparent, smaller)
        for cam_pose in estimated_camera_poses:
            view_idx = cam_pose['view_idx']
            c2w = np.array(cam_pose['c2w'])
            base_color = colors_per_view[view_idx % len(colors_per_view)]
            color = base_color + [150]  # Semi-transparent
            
            frustum = create_camera_frustum(c2w, scale=camera_scale * est_scale_mult, color=color)
            merged_scene.add_geometry(frustum, node_name=f"est_camera_{view_idx}")
            
            # Add marker (small sphere)
            cam_pos = c2w[:3, 3]
            marker = trimesh.creation.icosphere(subdivisions=1, radius=camera_scale * 0.08)
            marker.apply_translation(cam_pos)
            marker.visual.face_colors = base_color + [200]
            merged_scene.add_geometry(marker, node_name=f"est_marker_{view_idx}")
        
        # Add error lines (connecting estimated and GT camera positions, white)
        for est_pose, gt_pose in zip(estimated_camera_poses, gt_camera_poses):
            view_idx = est_pose['view_idx']
            est_pos = np.array(est_pose['camera_position'])
            gt_pos = np.array(gt_pose['camera_position'])
            
            # Create connecting line (white cylinder)
            direction = gt_pos - est_pos
            distance = np.linalg.norm(direction)
            if distance > 1e-6:
                direction = direction / distance
                midpoint = (est_pos + gt_pos) / 2
                
                line = trimesh.creation.cylinder(radius=camera_scale * 0.02, height=distance)
                # Rotate cylinder to point in direction
                z_axis = np.array([0, 0, 1])
                dot_product = np.dot(direction, z_axis)
                if np.abs(dot_product - 1) > 1e-6 and np.abs(dot_product + 1) > 1e-6:
                    rotation_axis = np.cross(z_axis, direction)
                    if np.linalg.norm(rotation_axis) > 1e-6:
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        rotation_angle = np.arccos(np.clip(dot_product, -1, 1))
                        rotation_matrix = trimesh.transformations.rotation_matrix(
                            rotation_angle, rotation_axis
                        )
                        line.apply_transform(rotation_matrix)
                elif dot_product < 0:
                    # Opposite direction, rotate 180 degrees around any perpendicular axis
                    rotation_matrix = trimesh.transformations.rotation_matrix(
                        np.pi, [1, 0, 0]
                    )
                    line.apply_transform(rotation_matrix)
                line.apply_translation(midpoint)
                line.visual.face_colors = [255, 255, 255, 200]  # White, slightly transparent
                merged_scene.add_geometry(line, node_name=f"error_line_{view_idx}")
        
        # Export
        merged_scene.export(str(output_path))
        logger.info(f"[Aligned Viz] Saved: {output_path}")
        logger.info(f"  Colors: View0=Red, View1=Green, View2=Blue, View3=Yellow, "
                   f"View4=Magenta, View5=Cyan, View6=Orange, View7=Purple")
        logger.info(f"  GT cameras: large (scale={gt_scale_mult}x), opaque")
        logger.info(f"  Estimated cameras: small (scale={est_scale_mult}x), semi-transparent")
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Failed to create aligned visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_object_with_cameras(
    sam3d_glb_path: Path,
    object_pose: dict,
    camera_poses: List[dict],
    output_path: Optional[Path] = None,
    camera_scale: float = 0.1,
) -> Optional[Path]:
    """
    Visualize object and all camera positions.
    
    Coordinate system notes:
    - SAM3D pose is in PyTorch3D camera coordinates (X-left, Y-up, Z-forward)
    - We use View 0 camera coordinates as world coordinates
    - Object is transformed to this coordinate system
    - Camera poses are also relative to this coordinate system
    
    Args:
        sam3d_glb_path: SAM3D output GLB file path
        object_pose: Object pose in world coordinates (View 0 camera coordinates)
        camera_poses: Camera poses for each view
        output_path: Output path
        camera_scale: Camera frustum size
    
    Returns:
        Output GLB file path
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
        output_path = sam3d_glb_path.parent / "result_with_cameras.glb"
    
    try:
        # Load SAM3D GLB
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # Extract vertices
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
        
        # Extract pose parameters
        scale = np.array(object_pose.get('scale', [1, 1, 1])).flatten()[:3]
        translation = np.array(object_pose.get('translation', [0, 0, 0])).flatten()[:3]
        rotation_quat = np.array(object_pose.get('rotation', [1, 0, 0, 0])).flatten()[:4]  # wxyz
        
        logger.info(f"[Viz] Object pose: scale={scale}, translation={translation}")
        logger.info(f"[Viz] Object rotation (wxyz): {rotation_quat}")
        
        # Use unified coordinate transform function
        # Transform chain: canonical (Z-up) -> Y-up -> scale -> rotate -> translate
        v_final = apply_sam3d_pose_to_mesh_vertices(canonical_vertices, object_pose)
        
        logger.info(f"[Viz] Object center: {v_final.mean(axis=0)}")
        logger.info(f"[Viz] Object bounds: [{v_final.min(axis=0)}, {v_final.max(axis=0)}]")
        
        # Create scene
        merged_scene = trimesh.Scene()
        
        # Add object
        if canonical_faces is not None:
            obj_mesh = trimesh.Trimesh(vertices=v_final, faces=canonical_faces, process=False)
            obj_mesh.visual.face_colors = [200, 200, 200, 255]  # Gray
        else:
            obj_mesh = trimesh.PointCloud(v_final, colors=[200, 200, 200, 255])
        merged_scene.add_geometry(obj_mesh, node_name="object")
        
        # Add coordinate axes (help understand orientation)
        # X axis - Red
        # Y axis - Green
        # Z axis - Blue
        axis_length = camera_scale * 2
        axis_vertices = np.array([
            [0, 0, 0], [axis_length, 0, 0],  # X
            [0, 0, 0], [0, axis_length, 0],  # Y
            [0, 0, 0], [0, 0, axis_length],  # Z
        ])
        axis_colors = np.array([
            [255, 0, 0, 255], [255, 0, 0, 255],  # X - Red
            [0, 255, 0, 255], [0, 255, 0, 255],  # Y - Green
            [0, 0, 255, 255], [0, 0, 255, 255],  # Z - Blue
        ])
        axis_pc = trimesh.PointCloud(axis_vertices, colors=axis_colors)
        merged_scene.add_geometry(axis_pc, node_name="world_axes")
        
        # Add cameras
        colors_per_view = [
            [255, 0, 0, 255],     # View 0: Red
            [0, 255, 0, 255],     # View 1: Green
            [0, 0, 255, 255],     # View 2: Blue
            [255, 255, 0, 255],   # View 3: Yellow
            [255, 0, 255, 255],   # View 4: Magenta
            [0, 255, 255, 255],   # View 5: Cyan
            [255, 128, 0, 255],   # View 6: Orange
            [128, 0, 255, 255],   # View 7: Purple
        ]
        
        for cam_pose in camera_poses:
            view_idx = cam_pose['view_idx']
            c2w = np.array(cam_pose['c2w'])
            color = colors_per_view[view_idx % len(colors_per_view)]
            
            # Camera position
            cam_pos = c2w[:3, 3]
            # Camera direction (Z axis direction)
            cam_dir = c2w[:3, 2]  # Third column is Z axis direction
            
            logger.info(f"[Viz] Camera {view_idx}: pos={cam_pos}, dir={cam_dir}")
            
            frustum = create_camera_frustum(c2w, scale=camera_scale, color=color)
            merged_scene.add_geometry(frustum, node_name=f"camera_{view_idx}")
        
        # Export
        merged_scene.export(str(output_path))
        logger.info(f"[Viz] Saved: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    Overlay SAM3D reconstructed object onto input pointmap.
    
    SAM3D pose parameters (scale, rotation, translation) are in real-world scale,
    and in PyTorch3D camera space.
    Input pointmap should also be in PyTorch3D camera space.
    
    Transform pipeline:
    SAM3D canonical (±0.5)
        ↓ scale * rotation + translation (SAM3D pose, real-world scale, PyTorch3D space)
    PyTorch3D camera space (real-world scale)
    
    Args:
        sam3d_glb_path: SAM3D output GLB file path (canonical space)
        input_pointmap: Input pointmap, shape (3, H, W), in PyTorch3D camera space
        sam3d_pose: SAM3D pose parameters {'scale', 'rotation', 'translation'}
        input_image: Original image for point cloud coloring
        output_path: Output path
    
    Returns:
        Overlaid GLB file path
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
        # Load SAM3D GLB
        sam3d_scene = trimesh.load(str(sam3d_glb_path))
        
        # Extract SAM3D pose parameters (already in PyTorch3D camera space, real-world scale)
        scale = sam3d_pose.get('scale', np.array([1.0, 1.0, 1.0]))
        rotation_quat = sam3d_pose.get('rotation', np.array([1.0, 0.0, 0.0, 0.0]))  # wxyz
        translation = sam3d_pose.get('translation', np.array([0.0, 0.0, 0.0]))
        
        if len(scale.shape) > 1:
            scale = scale.flatten()
        if len(rotation_quat.shape) > 1:
            rotation_quat = rotation_quat.flatten()
        if len(translation.shape) > 1:
            translation = translation.flatten()
        
        logger.info(f"[Overlay] SAM3D pose (PyTorch3D camera space):")
        logger.info(f"  scale: {scale} (object size, unit: meters)")
        logger.info(f"  rotation (wxyz): {rotation_quat}")
        logger.info(f"  translation: {translation} (object position, unit: meters)")
        
        # SAM3D internally applies z-up -> y-up rotation to GLB vertices
        # Must be consistent with layout_post_optimization_utils.get_mesh
        # Transform matrix: X = X, Y = -Z, Z = Y
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
            Transform SAM3D canonical space vertices to PyTorch3D camera space.
            
            Steps:
            1. Rotate canonical vertices from Z-up to Y-up (handled internally by SAM3D)
            2. Apply SAM3D pose (scale, rotation, translation)
            """
            # 1. Z-up to Y-up rotation
            v_rotated = vertices @ z_up_to_y_up_matrix.T
            pts_local = torch.from_numpy(v_rotated).float().unsqueeze(0)
            pts_world = pose_transform.transform_points(pts_local).squeeze(0).numpy()
            return pts_world
        
        # Create merged scene
        merged_scene = trimesh.Scene()
        
        # Transform and add SAM3D object
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
        
        # Create point cloud from input pointmap (already in PyTorch3D camera space)
        # input_pointmap shape: (3, H, W) or (1, 3, H, W)
        pm_np = input_pointmap
        if torch.is_tensor(pm_np):
            pm_tensor = pm_np.detach().cpu()
        else:
            pm_tensor = torch.from_numpy(pm_np).float()
            
        # Remove batch dimension
        while pm_tensor.ndim > 3:
            pm_tensor = pm_tensor[0]
        
        # Convert to (3, H, W)
        if pm_tensor.ndim == 3 and pm_tensor.shape[0] != 3:
            pm_tensor = pm_tensor.permute(2, 0, 1)
        
        # De-normalize (if needed)
        if pointmap_scale is not None and pointmap_shift is not None:
            normalizer = SSIPointmapNormalizer()
            scale_t = torch.as_tensor(pointmap_scale).float().view(-1)
            shift_t = torch.as_tensor(pointmap_shift).float().view(-1)
            pm_tensor = normalizer.denormalize(pm_tensor, scale_t, shift_t)
        
        pm_np = pm_tensor.permute(1, 2, 0).numpy()
        H, W = pm_np.shape[:2]
        
        # Get colors (from original image)
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
        
        # Filter invalid points (NaN, Inf)
        valid_mask = np.all(np.isfinite(pm_np), axis=-1)
        pm_points = pm_np[valid_mask].reshape(-1, 3)
        
        if colors is not None:
            colors = colors.reshape(H, W, 3)[valid_mask].reshape(-1, 3)
        else:
            # Default gray
            colors = np.full((len(pm_points), 3), 128, dtype=np.uint8)
        
        # Downsample
        if len(pm_points) > 100000:
            step = len(pm_points) // 100000
            pm_points = pm_points[::step]
            colors = colors[::step]
        
        # Create point cloud
        point_cloud = trimesh.points.PointCloud(vertices=pm_points, colors=colors)
        merged_scene.add_geometry(point_cloud, node_name="input_pointcloud")
        
        logger.info(f"[Overlay] Points in pointcloud: {len(pm_points)}")
        
        # Export
        merged_scene.export(str(output_path))
        logger.info(f"✓ Overlay GLB saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.warning(f"Failed to create overlay GLB: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Self-Occlusion Detection using Voxel Ray Casting
# ============================================================================

def ray_box_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> tuple:
    """
    Compute ray-AABB box intersection.
    
    Returns:
        (t_enter, t_exit) or (None, None) if no intersection
    """
    t_min = -np.inf
    t_max = np.inf
    
    for i in range(3):
        if abs(ray_dir[i]) < 1e-10:
            # Ray parallel to this axis
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return None, None
        else:
            t1 = (box_min[i] - ray_origin[i]) / ray_dir[i]
            t2 = (box_max[i] - ray_origin[i]) / ray_dir[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None, None
    
    return t_min, t_max


def trace_ray_3d_dda(
    start: np.ndarray,  # (3,) Ray start (can be outside voxel grid)
    end: np.ndarray,    # (3,) Ray end (voxel index)
    grid_size: int = 64,
) -> List[tuple]:
    """
    3D DDA (Digital Differential Analyzer) algorithm.
    Trace ray from start to end, return all traversed voxels.
    
    Optimization: if start is outside grid, compute intersection with grid boundary first.
    
    Args:
        start: Ray start (float coordinates)
        end: Ray end (float coordinates)
        grid_size: Voxel grid size
    
    Returns:
        List of (dim0, dim1, dim2) voxel indices, in order from start to end
    """
    # Ray direction
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return []
    direction = direction / length
    
    # Check if start is outside grid, if so, find entry point
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([grid_size, grid_size, grid_size])
    
    actual_start = start.copy()
    
    # If start is outside grid, compute entry point
    if not np.all((start >= 0) & (start < grid_size)):
        t_enter, t_exit = ray_box_intersection(start, direction, box_min, box_max)
        if t_enter is None or t_enter > length:
            # Ray does not pass through grid, or entry point is after end
            return []
        if t_enter > 0:
            # Start from entry point
            actual_start = start + direction * (t_enter + 0.001)
    
    # Current position
    current = actual_start.copy()
    
    # Current voxel
    voxel = np.floor(current).astype(int)
    # Ensure points on boundary are handled correctly
    voxel = np.clip(voxel, 0, grid_size - 1)
    
    end_voxel = np.floor(end).astype(int)
    end_voxel = np.clip(end_voxel, 0, grid_size - 1)
    
    # Step direction
    step = np.sign(direction).astype(int)
    step[step == 0] = 1  # Avoid division by zero
    
    # Compute distance to next voxel boundary
    tmax = np.zeros(3)
    tdelta = np.zeros(3)
    
    for i in range(3):
        if abs(direction[i]) < 1e-10:
            tmax[i] = float('inf')
            tdelta[i] = float('inf')
        else:
            if direction[i] > 0:
                tmax[i] = (voxel[i] + 1 - current[i]) / direction[i]
            else:
                tmax[i] = (voxel[i] - current[i]) / direction[i]
            tdelta[i] = abs(1.0 / direction[i])
    
    # Collect traversed voxels
    voxels = []
    max_steps = grid_size * 3  # Max steps in grid
    
    for _ in range(max_steps):
        # Check if inside grid
        if np.all((voxel >= 0) & (voxel < grid_size)):
            voxels.append(tuple(voxel))
        else:
            # Already outside grid, stop
            break
        
        # Check if reached end
        if np.all(voxel == end_voxel):
            break
        
        # Find minimum tmax, decide next step direction
        min_axis = np.argmin(tmax)
        
        # Step to next voxel
        voxel[min_axis] += step[min_axis]
        tmax[min_axis] += tdelta[min_axis]
    
    return voxels


def compute_self_occlusion(
    latent_coords: np.ndarray,  # (N, 4) or (N, 3) - voxel coordinates
    camera_position_voxel: np.ndarray,  # (3,) Camera position in voxel space
    grid_size: int = 64,
    neighbor_tolerance: float = 4.0,  # Ignore occluding voxels within this distance (4.0 handles grazing angles)
) -> np.ndarray:
    """
    Detect self-occlusion using 3D DDA ray tracing with neighbor tolerance.
    
    Core idea:
    - Build 64x64x64 occupancy grid
    - For each voxel, cast ray from camera to that voxel
    - If ray passes through other occupied voxels (far enough from target), the voxel is occluded
    
    Improvements:
    - neighbor_tolerance: ignore occluding voxels within this distance of target
    - This avoids false positives from adjacent voxels
    
    Args:
        latent_coords: (N, 4) or (N, 3), voxel coordinates
        camera_position_voxel: Camera position in voxel space
        grid_size: Voxel grid size (default 64)
        neighbor_tolerance: Ignore occluding voxels within this distance (default 1.5, ~sqrt(3), diagonal neighbors)
    
    Returns:
        self_visible: (N,) bool array, True = not self-occluded
    """
    # Handle coordinate format
    if latent_coords.shape[1] == 4:
        voxel_coords = latent_coords[:, 1:4].astype(int)
    else:
        voxel_coords = latent_coords.astype(int)
    
    N = len(voxel_coords)
    
    # Debug info
    logger.info(f"[Self-Occlusion DDA] Voxel coords range: "
               f"dim0=[{voxel_coords[:, 0].min()}, {voxel_coords[:, 0].max()}], "
               f"dim1=[{voxel_coords[:, 1].min()}, {voxel_coords[:, 1].max()}], "
               f"dim2=[{voxel_coords[:, 2].min()}, {voxel_coords[:, 2].max()}]")
    logger.info(f"[Self-Occlusion DDA] Camera position in voxel space: {camera_position_voxel}")
    logger.info(f"[Self-Occlusion DDA] Neighbor tolerance: {neighbor_tolerance}")
    
    # Step 1: Build occupancy grid
    occupancy = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    for coord in voxel_coords:
        d0, d1, d2 = coord[0], coord[1], coord[2]
        if 0 <= d0 < grid_size and 0 <= d1 < grid_size and 0 <= d2 < grid_size:
            occupancy[d0, d1, d2] = True
    
    logger.info(f"[Self-Occlusion DDA] Built occupancy grid: {occupancy.sum()} occupied voxels")
    
    # Step 2: DDA ray tracing for each voxel
    self_visible = np.ones(N, dtype=bool)
    occluded_count = 0
    
    # Pre-compute target voxel coordinates (float, for distance calculation)
    target_coords_float = voxel_coords.astype(float)
    
    tolerance_sq = neighbor_tolerance ** 2
    
    for i in range(N):
        target = target_coords_float[i] + 0.5  # Voxel center
        target_int = voxel_coords[i]
        
        # DDA ray tracing
        ray_voxels = trace_ray_3d_dda(
            camera_position_voxel,
            target,
            grid_size
        )
        
        # Check if there are occupied voxels along the ray (excluding target and neighbors)
        for voxel in ray_voxels:
            d0, d1, d2 = voxel
            
            # Skip voxels outside grid
            if not (0 <= d0 < grid_size and 0 <= d1 < grid_size and 0 <= d2 < grid_size):
                continue
            
            # Skip unoccupied voxels
            if not occupancy[d0, d1, d2]:
                continue
            
            # Compute distance to target (voxel units)
            dist_sq = (d0 - target_int[0])**2 + (d1 - target_int[1])**2 + (d2 - target_int[2])**2
            
            # If too close to target (including target itself), skip
            if dist_sq <= tolerance_sq:
                continue
            
            # Found real occluding voxel
            self_visible[i] = False
            occluded_count += 1
            break
    
    visible_count = N - occluded_count
    logger.info(f"[Self-Occlusion DDA] Results: {visible_count} visible, {occluded_count} occluded "
               f"({100 * visible_count / N:.1f}% visible)")
    
    return self_visible


def canonical_to_voxel(pos_canonical: np.ndarray, scale: float) -> np.ndarray:
    """
    Convert canonical space coordinates to voxel space coordinates.
    
    Transform chain (extracted from compute_latent_visibility):
    voxel [0, 64) → normalized [-0.5, 0.5] → Z_UP_TO_Y_UP → scale → canonical
    
    Inverse transform:
    canonical → /scale → Y_UP_TO_Z_UP → (x+0.5)*64 → voxel
    
    Args:
        pos_canonical: (..., 3) canonical space coordinates [x, y, z] (Y-up)
        scale: Object scale factor
    
    Returns:
        pos_voxel: (..., 3) voxel space coordinates, keeping [x, y, z] order for ray tracing
    """
    # 1. Remove scale
    pos_normalized = pos_canonical / scale
    
    # 2. Y-up -> Z-up inverse transform
    # Z_UP_TO_Y_UP: (x, y, z)_zup → (x, -z, y)_yup
    # Inverse transform: (x, y, z)_yup -> (x, z, -y)_zup
    # 
    # Note: voxel coords from argwhere are in [z, y, x] order
    # After Z_UP_TO_Y_UP transform, canonical = [dim0, dim2, -dim1]
    # x = a, y = c, z = -b
    # → a = x, b = -z, c = y
    # → normalized = [x, -z, y]
    
    x, y, z = pos_normalized[..., 0], pos_normalized[..., 1], pos_normalized[..., 2]
    
    # normalized in voxel order [a, b, c] where canonical = [a, c, -b]
    # so a = x, b = -z, c = y
    voxel_normalized = np.stack([x, -z, y], axis=-1)
    
    # 3. normalized [-0.5, 0.5] → voxel [0, 64)
    pos_voxel = (voxel_normalized + 0.5) * 64
    
    return pos_voxel


def compute_self_occlusion_for_all_views(
    latent_coords: np.ndarray,  # (N, 4) or (N, 3) - voxel coordinates
    camera_positions_canonical: List[np.ndarray],  # List of camera positions in canonical space
    scale: float,
    grid_size: int = 64,
    neighbor_tolerance: float = 4.0,  # Ignore occluding voxels within this distance
) -> np.ndarray:
    """
    Compute self-occlusion for all views.
    
    Args:
        latent_coords: Voxel coordinates
        camera_positions_canonical: Camera position for each view (canonical space)
        scale: Object scale factor
        grid_size: Voxel grid size
        neighbor_tolerance: Ignore occluding voxels within this distance (handles grazing angles)
    
    Returns:
        self_occlusion_matrix: (N, num_views) matrix, 1.0 = visible, 0.0 = self-occluded
    """
    num_views = len(camera_positions_canonical)
    N = len(latent_coords)
    
    self_occlusion_matrix = np.zeros((N, num_views), dtype=np.float32)
    
    for view_idx, camera_pos_canonical in enumerate(camera_positions_canonical):
        # Convert camera position to voxel space
        camera_pos_voxel = canonical_to_voxel(camera_pos_canonical, scale)
        
        logger.info(f"[Self-Occlusion] View {view_idx}: "
                   f"camera canonical={camera_pos_canonical}, voxel={camera_pos_voxel}")
        
        # Compute self-occlusion for this view
        self_visible = compute_self_occlusion(
            latent_coords, 
            camera_pos_voxel,
            grid_size,
            neighbor_tolerance
        )
        
        self_occlusion_matrix[:, view_idx] = self_visible.astype(np.float32)
    
    return self_occlusion_matrix


def visualize_self_occlusion_per_view(
    self_occlusion_matrix: np.ndarray,  # (N, num_views) - self-occlusion result
    visibility_result: dict,  # Result from compute_latent_visibility (contains canonical coords and camera poses)
    output_dir: Path,
) -> List[Path]:
    """
    Visualize self-occlusion results for each view.
    
    Directly use canonical coordinates and camera poses from visibility_result (verified),
    ensuring alignment with latent_visibility_per_view.
    
    Green: visible (not self-occluded)
    Red: self-occluded
    
    Returns:
        List of output file paths
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed, cannot visualize")
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Directly use canonical coords from visibility_result (verified correct)
    canonical_coords = visibility_result['canonical_coords']
    canonical_camera_poses = visibility_result['canonical_camera_poses']
    scale = visibility_result['scale']
    
    num_views = len(canonical_camera_poses)
    output_paths = []
    
    # Define view colors (same as latent_visibility_per_view)
    view_colors = [
        [255, 100, 100, 255],  # Red
        [100, 255, 100, 255],  # Green
        [100, 100, 255, 255],  # Blue
        [255, 255, 100, 255],  # Yellow
        [255, 100, 255, 255],  # Purple
        [100, 255, 255, 255],  # Cyan
        [255, 180, 100, 255],  # Orange
        [180, 100, 255, 255],  # Purple-blue
    ]
    
    for view_idx in range(num_views):
        scene = trimesh.Scene()
        
        # Self-occlusion status
        self_visible = self_occlusion_matrix[:, view_idx] > 0.5
        
        # Colors: green=visible, red=occluded
        colors_visible = np.zeros((len(canonical_coords), 4), dtype=np.uint8)
        colors_visible[self_visible] = [0, 255, 0, 255]   # Green: visible
        colors_visible[~self_visible] = [255, 0, 0, 255]  # Red: occluded
        
        # Use spheres to display latent points (larger and clearer)
        # Sample display (if too many points)
        max_spheres = 10000
        if len(canonical_coords) > max_spheres:
            indices = np.random.choice(len(canonical_coords), max_spheres, replace=False)
        else:
            indices = np.arange(len(canonical_coords))
        
        sphere_radius = scale * 0.008  # Sphere radius
        for idx in indices:
            sphere = trimesh.creation.icosphere(radius=sphere_radius, subdivisions=1)
            sphere.apply_translation(canonical_coords[idx])
            color = colors_visible[idx]
            sphere.visual.vertex_colors = color
            scene.add_geometry(sphere, node_name=f"latent_{idx}")
        
        # Add all cameras, current view larger, others smaller
        for cam_idx, cam_pose in enumerate(canonical_camera_poses):
            camera_pos = cam_pose['camera_position']
            
            # Current view camera larger, others smaller
            if cam_idx == view_idx:
                radius = scale * 0.08  # Large
                subdivisions = 2
            else:
                radius = scale * 0.03  # Small
                subdivisions = 1
            
            camera_sphere = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
            camera_sphere.apply_translation(camera_pos)
            
            # Color
            color = view_colors[cam_idx % len(view_colors)]
            camera_sphere.visual.vertex_colors = color
            
            scene.add_geometry(camera_sphere, node_name=f"camera_{cam_idx}")
        
        # Save
        output_path = output_dir / f"self_occlusion_view_{view_idx:02d}.glb"
        scene.export(str(output_path))
        output_paths.append(output_path)
        
        visible_count = self_visible.sum()
        logger.info(f"[Self-Occlusion Viz] View {view_idx}: "
                   f"{visible_count}/{len(canonical_coords)} visible ({100*visible_count/len(canonical_coords):.1f}%), "
                   f"saved to {output_path.name}")
    
    return output_paths


def compute_latent_visibility(
    latent_coords: np.ndarray,  # (N, 4) or (N, 3) - Stage 2 latent coordinates (voxel space)
    object_pose: dict,  # Object pose from Stage 1 {'scale', 'rotation', 'translation'} (in view0 coordinates)
    camera_poses: List[dict],  # List of camera poses, each containing {'c2w', 'w2c', 'camera_position', 'R_c2w'}
    self_occlusion_tolerance: float = 4.0,  # Self-occlusion detection tolerance (voxel units)
) -> dict:
    """
    Compute visibility of each latent point from each view in CANONICAL space.
    
    **Core idea**:
    - Object in canonical space, only apply scale (not rotation/translation)
    - Transform camera poses to canonical space
    - Use self-occlusion (DDA ray tracing) for visibility
    
    Args:
        latent_coords: Stage 2 latent coordinates (N, 4) or (N, 3)
        object_pose: Object pose {'scale', 'rotation' (wxyz), 'translation'}
        camera_poses: List of camera poses
        self_occlusion_tolerance: Self-occlusion detection tolerance (voxel units)
    
    Returns:
        dict: 
            - visibility_matrix: (N_latents, N_views) visibility matrix (0=occluded, 1=visible)
            - canonical_coords: (N, 3) latent points in canonical space
            - canonical_camera_poses: Camera poses in canonical space
            - scale: Object scale
    """
    from scipy.spatial.transform import Rotation as R_scipy
    
    num_views = len(camera_poses)
    
    # === Helper function to convert tensor to numpy ===
    def to_numpy(x):
        if x is None:
            return None
        if hasattr(x, 'cpu'):
            return x.cpu().numpy()
        return np.array(x)
    
    # === Step 1: Object pose parameters ===
    obj_scale = np.atleast_1d(to_numpy(object_pose.get('scale', [1, 1, 1]))).flatten()
    if len(obj_scale) == 1:
        obj_scale = np.array([obj_scale[0], obj_scale[0], obj_scale[0]])
    obj_rotation_quat = np.atleast_1d(to_numpy(object_pose.get('rotation', [1, 0, 0, 0]))).flatten()
    obj_translation = np.atleast_1d(to_numpy(object_pose.get('translation', [0, 0, 0]))).flatten()
    
    # Object rotation matrix (wxyz -> scipy xyzw)
    obj_R = R_scipy.from_quat([obj_rotation_quat[1], obj_rotation_quat[2], 
                               obj_rotation_quat[3], obj_rotation_quat[0]]).as_matrix()
    
    # Z-up to Y-up transform (consistent with GLB standard)
    Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    
    logger.info(f"[Visibility Canonical] Computing visibility in CANONICAL space")
    logger.info(f"[Visibility Canonical] Object scale: {obj_scale}")
    logger.info(f"[Visibility Canonical] Object rotation (wxyz): {obj_rotation_quat}")
    logger.info(f"[Visibility Canonical] Object translation: {obj_translation}")
    
    # === Step 2: Transform latent points to canonical space (apply scale, Y-up only) ===
    # Convert to numpy if tensor
    if hasattr(latent_coords, 'cpu'):
        latent_coords = latent_coords.cpu().numpy()
    
    # Handle (N, 4) format
    if latent_coords.shape[1] == 4:
        coords = latent_coords[:, 1:4].copy()
    else:
        coords = latent_coords.copy()
    
    # Convert voxel indices to canonical [-0.5, 0.5]
    if coords.max() > 1.0:
        coords = (coords / 64.0) - 0.5
    coords = np.clip(coords, -0.5, 0.5)
    
    # Apply Z-up to Y-up and scale
    canonical_coords = (coords @ Z_UP_TO_Y_UP) * obj_scale[0]
    
    num_latents = canonical_coords.shape[0]
    
    logger.info(f"[Visibility Canonical] Latent points in canonical space:")
    logger.info(f"  Count: {num_latents}")
    logger.info(f"  X: [{canonical_coords[:, 0].min():.4f}, {canonical_coords[:, 0].max():.4f}]")
    logger.info(f"  Y: [{canonical_coords[:, 1].min():.4f}, {canonical_coords[:, 1].max():.4f}]")
    logger.info(f"  Z: [{canonical_coords[:, 2].min():.4f}, {canonical_coords[:, 2].max():.4f}]")
    
    # === Step 3: Transform camera poses to canonical space ===
    # Use the same method as visualize_in_canonical_space
    canonical_camera_poses = []
    camera_positions_for_occlusion = []  # For self-occlusion calculation
    for view_idx, cam_pose in enumerate(camera_poses):
        # Get camera pose in world coordinates
        camera_pos_world = np.array(cam_pose.get('camera_position', [0, 0, 0])).flatten()
        cam_R_c2w = np.array(cam_pose.get('R_c2w', np.eye(3)))
        
        # Transform camera position to canonical space:
        # 1. Subtract object translation
        # 2. Apply inverse of object rotation
        # 3. Apply Z-up to Y-up transform
        camera_pos_obj = obj_R.T @ (camera_pos_world - obj_translation)
        camera_pos_canonical = camera_pos_obj @ Z_UP_TO_Y_UP
        
        # Transform camera's three axis vectors separately (not rotation matrix directly)
        # This ensures consistency with visualize_in_canonical_space
        camera_forward_world = cam_R_c2w @ np.array([0, 0, 1])  # Z axis (forward)
        camera_up_world = cam_R_c2w @ np.array([0, 1, 0])        # Y axis (up)
        camera_right_world = cam_R_c2w @ np.array([1, 0, 0])     # X axis (right)
        
        # Transform each axis to canonical space
        camera_forward_obj = obj_R.T @ camera_forward_world
        camera_forward_canonical = camera_forward_obj @ Z_UP_TO_Y_UP
        camera_forward_canonical = camera_forward_canonical / (np.linalg.norm(camera_forward_canonical) + 1e-8)
        
        camera_up_obj = obj_R.T @ camera_up_world
        camera_up_canonical = camera_up_obj @ Z_UP_TO_Y_UP
        camera_up_canonical = camera_up_canonical / (np.linalg.norm(camera_up_canonical) + 1e-8)
        
        camera_right_obj = obj_R.T @ camera_right_world
        camera_right_canonical = camera_right_obj @ Z_UP_TO_Y_UP
        camera_right_canonical = camera_right_canonical / (np.linalg.norm(camera_right_canonical) + 1e-8)
        
        # Rebuild c2w and w2c matrices
        cam_R_canonical = np.column_stack([camera_right_canonical, camera_up_canonical, camera_forward_canonical])
        
        # Compute w2c matrix in canonical space
        w2c_canonical = np.eye(4)
        w2c_canonical[:3, :3] = cam_R_canonical.T  # R_w2c = R_c2w.T
        w2c_canonical[:3, 3] = -cam_R_canonical.T @ camera_pos_canonical
        
        canonical_camera_poses.append({
            'camera_position': camera_pos_canonical,
            'R_c2w': cam_R_canonical,
            'w2c': w2c_canonical,
            'camera_forward': camera_forward_canonical,  # Also save forward direction for visualization
            'view_idx': view_idx,
        })
        
        # Save camera position for self-occlusion calculation
        camera_positions_for_occlusion.append(camera_pos_canonical)
        
        if view_idx == 0:
            logger.info(f"[Visibility Canonical] Camera 0 in canonical space:")
            logger.info(f"  Position: {camera_pos_canonical}")
            logger.info(f"  Forward: {camera_forward_canonical}")
    
    # === Step 4: Use self-occlusion (DDA) for visibility ===
    logger.info(f"[Visibility] Computing self-occlusion with tolerance={self_occlusion_tolerance}")
    
    # Call self-occlusion calculation function
    visibility_matrix = compute_self_occlusion_for_all_views(
        latent_coords=latent_coords,  # Original voxel coordinates
        camera_positions_canonical=camera_positions_for_occlusion,
        scale=obj_scale[0],
        grid_size=64,
        neighbor_tolerance=self_occlusion_tolerance,
    )
    
    # Statistics
    logger.info(f"[Visibility Canonical] Visibility matrix computed: shape={visibility_matrix.shape}")
    logger.info(f"[Visibility Canonical] Stats: mean={visibility_matrix.mean():.3f}, "
               f"min={visibility_matrix.min():.3f}, max={visibility_matrix.max():.3f}")
    
    for view_idx in range(num_views):
        view_vis = visibility_matrix[:, view_idx]
        visible_count = (view_vis > 0.5).sum()
        logger.info(f"  View {view_idx}: visible={visible_count}/{num_latents} ({visible_count/num_latents:.1%})")
    
    # Return visibility matrix and canonical space data (for visualization)
    return {
        'visibility_matrix': visibility_matrix,
        'canonical_coords': canonical_coords,
        'canonical_camera_poses': canonical_camera_poses,
        'scale': obj_scale[0],
    }


def visualize_in_canonical_space(
    latent_coords: np.ndarray,
    visibility_matrix: np.ndarray,
    scale: float,
    reference_glb = None,
    camera_poses: Optional[List[dict]] = None,
    object_pose: Optional[dict] = None,  # Need object pose to transform cameras to canonical space
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Display Latent and Mesh in object Canonical space (Y-up, consistent with GLB standard).
    
    **Key finding**:
    - GLB export applies Z-up -> Y-up transform to mesh: (x,y,z) -> (x,-z,y)
    - Latent coords don't have this transform
    - So we need to apply the same transform to latent for alignment
    
    **Camera pose handling**:
    - Input camera_poses are in world coordinates
    - Need to use inverse of object_pose to transform cameras to object canonical space
    
    Args:
        latent_coords: Latent coordinates (N, 4) or (N, 3)
        visibility_matrix: Visibility matrix (N, N_views)
        scale: Object scale factor
        reference_glb: Reference mesh (trimesh.Scene), already in Y-up space
        camera_poses: List of camera poses (optional, in world coordinates)
        object_pose: Object pose dict (scale, rotation, translation), for transforming cameras to canonical space
        output_path: Output path
    
    Returns:
        Output file path
    """
    try:
        import trimesh
    except ImportError:
        logger.error("trimesh not installed")
        return None
    
    # Convert scale to numpy float if it's a tensor
    if hasattr(scale, 'cpu'):  # It's a tensor
        scale = float(scale.cpu().numpy().flatten()[0])
    elif hasattr(scale, '__len__'):  # It's an array
        scale = float(np.atleast_1d(scale).flatten()[0])
    else:
        scale = float(scale)
    
    if output_path is None:
        output_path = Path("visualization") / "canonical_view.glb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[Canonical Viz] Creating visualization in Y-up canonical space (GLB standard)")
    logger.info(f"[Canonical Viz] Scale = {scale:.4f}")
    
    # === Process Latent coords ===
    # Convert to numpy if tensor
    if hasattr(latent_coords, 'cpu'):
        latent_coords = latent_coords.cpu().numpy()
    
    # Handle (N, 4) format
    if latent_coords.shape[1] == 4:
        coords = latent_coords[:, 1:4].copy()
    else:
        coords = latent_coords.copy()
    
    # Convert voxel indices to canonical [-0.5, 0.5]
    if coords.max() > 1.0:
        coords = (coords / 64.0) - 0.5
    coords = np.clip(coords, -0.5, 0.5)
    
    logger.info(f"[Canonical Viz] Latent in Z-up canonical space:")
    logger.info(f"  dim0: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}], range={coords[:, 0].max()-coords[:, 0].min():.4f}")
    logger.info(f"  dim1: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}], range={coords[:, 1].max()-coords[:, 1].min():.4f}")
    logger.info(f"  dim2: [{coords[:, 2].min():.4f}, {coords[:, 2].max():.4f}], range={coords[:, 2].max()-coords[:, 2].min():.4f}")
    
    # Apply Z-up to Y-up transformation (same as GLB export does for mesh)
    # This transforms (x, y, z) -> (x, -z, y)
    Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    latent_yup = coords @ Z_UP_TO_Y_UP
    
    # Apply scale
    latent_world = latent_yup * scale
    
    logger.info(f"[Canonical Viz] Latent after Z-up->Y-up transform and scale:")
    logger.info(f"  X: [{latent_world[:, 0].min():.4f}, {latent_world[:, 0].max():.4f}], range={latent_world[:, 0].max()-latent_world[:, 0].min():.4f}")
    logger.info(f"  Y: [{latent_world[:, 1].min():.4f}, {latent_world[:, 1].max():.4f}], range={latent_world[:, 1].max()-latent_world[:, 1].min():.4f}")
    logger.info(f"  Z: [{latent_world[:, 2].min():.4f}, {latent_world[:, 2].max():.4f}], range={latent_world[:, 2].max()-latent_world[:, 2].min():.4f}")
    
    # === Process Mesh ===
    # GLB mesh is already in Y-up space, just apply scale
    mesh_for_scene = None
    if reference_glb is not None:
        try:
            mesh_vertices = None
            mesh_faces = None
            if isinstance(reference_glb, trimesh.Scene):
                for name, geom in reference_glb.geometry.items():
                    if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                        mesh_vertices = geom.vertices.copy()
                        mesh_faces = geom.faces.copy()
                        break
            elif hasattr(reference_glb, 'vertices'):
                mesh_vertices = reference_glb.vertices.copy()
                mesh_faces = reference_glb.faces.copy() if hasattr(reference_glb, 'faces') else None
            
            if mesh_vertices is not None:
                # GLB mesh is already in Y-up, already scaled (vertices are in [-0.5, 0.5])
                # Just apply scale to match latent
                mesh_world = mesh_vertices * scale
                
                logger.info(f"[Canonical Viz] Mesh (Y-up, from GLB) after scale:")
                logger.info(f"  X: [{mesh_world[:, 0].min():.4f}, {mesh_world[:, 0].max():.4f}], range={mesh_world[:, 0].max()-mesh_world[:, 0].min():.4f}")
                logger.info(f"  Y: [{mesh_world[:, 1].min():.4f}, {mesh_world[:, 1].max():.4f}], range={mesh_world[:, 1].max()-mesh_world[:, 1].min():.4f}")
                logger.info(f"  Z: [{mesh_world[:, 2].min():.4f}, {mesh_world[:, 2].max():.4f}], range={mesh_world[:, 2].max()-mesh_world[:, 2].min():.4f}")
                
                if mesh_faces is not None:
                    mesh_for_scene = trimesh.Trimesh(vertices=mesh_world, faces=mesh_faces)
                    mesh_for_scene.visual.face_colors = [0, 255, 255, 100]  # Cyan, semi-transparent
        except Exception as e:
            logger.warning(f"[Canonical Viz] Could not process mesh: {e}")
    
    # === Create scene ===
    scene = trimesh.Scene()
    
    # Visibility coloring
    visibility_scores = visibility_matrix.mean(axis=1)
    colors = np.zeros((len(latent_world), 4), dtype=np.uint8)
    colors[:, 0] = (255 * (1.0 - visibility_scores)).astype(np.uint8)  # Red for invisible
    colors[:, 1] = (255 * visibility_scores).astype(np.uint8)  # Green for visible
    colors[:, 3] = 200  # Alpha
    
    # Add latent point cloud
    point_cloud = trimesh.PointCloud(vertices=latent_world, colors=colors)
    scene.add_geometry(point_cloud, node_name="latent_points")
    
    # Add mesh
    if mesh_for_scene is not None:
        scene.add_geometry(mesh_for_scene, node_name="mesh_reference")
        logger.info(f"[Canonical Viz] Added mesh to scene")
    
    # === Add camera pose visualization (if provided) ===
    if camera_poses is not None and len(camera_poses) > 0 and object_pose is not None:
        logger.info(f"[Canonical Viz] Adding {len(camera_poses)} camera poses (transformed to canonical space)")
        
        # Get object pose for camera transform
        from scipy.spatial.transform import Rotation as R_scipy
        
        obj_scale = np.atleast_1d(object_pose.get('scale', [1, 1, 1])).flatten()
        if len(obj_scale) == 1:
            obj_scale = np.array([obj_scale[0], obj_scale[0], obj_scale[0]])
        obj_rotation_quat = np.atleast_1d(object_pose.get('rotation', [1, 0, 0, 0])).flatten()
        obj_translation = np.atleast_1d(object_pose.get('translation', [0, 0, 0])).flatten()
        
        # Object rotation matrix (wxyz -> scipy needs xyzw)
        obj_R = R_scipy.from_quat([obj_rotation_quat[1], obj_rotation_quat[2], 
                                    obj_rotation_quat[3], obj_rotation_quat[0]]).as_matrix()
        
        # Z-up to Y-up transform (consistent with GLB)
        Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        
        # Camera colors (consistent with other visualizations)
        camera_colors = [
            [255, 0, 0],    # View 0: Red
            [0, 255, 0],    # View 1: Green
            [0, 0, 255],    # View 2: Blue
            [255, 255, 0],  # View 3: Yellow
            [255, 0, 255],  # View 4: Magenta
            [0, 255, 255],  # View 5: Cyan
            [255, 128, 0],  # View 6: Orange
            [128, 0, 255],  # View 7: Purple
        ]
        
        for i, cam_pose in enumerate(camera_poses):
            try:
                # Get camera pose in world coordinates
                # camera_poses format: R_c2w, t_c2w, camera_position, c2w, w2c
                cam_R_c2w = np.array(cam_pose.get('R_c2w', np.eye(3)))
                camera_pos_world = np.array(cam_pose.get('camera_position', [0, 0, 0])).flatten()
                
                # Transform camera position to object canonical space (SCALED):
                # Note: Object in canonical_view is canonical * scale
                # So camera should also be in the same scaled space
                # 1. Subtract object translation (world coordinates)
                # 2. Apply inverse of object rotation (get position in object frame)
                # 3. Do not divide by scale (displayed object is also scaled)
                # 4. Apply Z-up to Y-up transform
                camera_pos_obj = obj_R.T @ (camera_pos_world - obj_translation)
                # Do not divide by scale! Displayed object is in scaled canonical space
                camera_pos_canonical = camera_pos_obj @ Z_UP_TO_Y_UP
                
                # Camera direction also needs transform (Z axis direction in c2w frame)
                camera_forward_world = cam_R_c2w @ np.array([0, 0, 1])  # Z axis direction of c2w
                camera_forward_obj = obj_R.T @ camera_forward_world
                camera_forward_canonical = camera_forward_obj @ Z_UP_TO_Y_UP
                # Normalize
                camera_forward_canonical = camera_forward_canonical / (np.linalg.norm(camera_forward_canonical) + 1e-8)
                
                color = camera_colors[i % len(camera_colors)]
                
                # Create camera sphere
                cam_marker = trimesh.creation.icosphere(radius=0.03, subdivisions=1)
                cam_marker.apply_translation(camera_pos_canonical)
                cam_marker.visual.face_colors = color + [255]
                scene.add_geometry(cam_marker, node_name=f"camera_{i}")
                
                # Add direction indicator line
                line_end = camera_pos_canonical + camera_forward_canonical * 0.15
                line_verts = np.array([camera_pos_canonical, line_end])
                line_colors = np.array([color + [255], color + [255]])
                line_pc = trimesh.PointCloud(vertices=line_verts, colors=line_colors)
                scene.add_geometry(line_pc, node_name=f"camera_{i}_dir")
                
                if i == 0:
                    logger.info(f"[Canonical Viz] Camera 0 world pos: {camera_pos_world}")
                    logger.info(f"[Canonical Viz] Camera 0 canonical pos: {camera_pos_canonical}")
                
            except Exception as e:
                logger.warning(f"[Canonical Viz] Could not add camera {i}: {e}")
    
    # Add coordinate axes (at origin)
    axis_length = scale * 0.3
    axis_verts = np.array([
        [0, 0, 0], [axis_length, 0, 0],
        [0, 0, 0], [0, axis_length, 0],
        [0, 0, 0], [0, 0, axis_length],
    ])
    axis_colors = np.array([
        [255, 0, 0, 255], [255, 0, 0, 255],  # X - Red
        [0, 255, 0, 255], [0, 255, 0, 255],  # Y - Green (up in Y-up space)
        [0, 0, 255, 255], [0, 0, 255, 255],  # Z - Blue
    ])
    axis_pc = trimesh.PointCloud(vertices=axis_verts, colors=axis_colors)
    scene.add_geometry(axis_pc, node_name="axes")
    
    # Save
    scene.export(str(output_path))
    logger.info(f"[Canonical Viz] Saved to: {output_path}")
    
    return output_path


def visualize_latent_visibility(
    visibility_result: dict,  # Result from compute_latent_visibility
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Visualize latent point visibility in CANONICAL space.
    
    Generate a GLB file containing:
    1. Latent points (colored by visibility: green=visible, red=invisible)
    2. Camera positions and orientations
    3. Coordinate axes
    
    Note: Does not show mesh, only latent points
    
    Args:
        visibility_result: Dictionary returned by compute_latent_visibility, containing:
            - visibility_matrix: Visibility matrix (N, N_views)
            - canonical_coords: Latent points in canonical space
            - canonical_camera_poses: Camera poses in canonical space
            - scale: Object scale
        output_path: Output path
    
    Returns:
        Output GLB file path
    """
    try:
        import trimesh
    except ImportError:
        logger.error("trimesh not installed, cannot create visualization")
        return None
    
    if output_path is None:
        output_path = Path("visualization") / "latent_visibility.glb"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    visibility_matrix = visibility_result['visibility_matrix']
    canonical_coords = visibility_result['canonical_coords']
    canonical_camera_poses = visibility_result['canonical_camera_poses']
    scale = visibility_result.get('scale', 1.0)
    
    logger.info(f"[Visibility Viz Canonical] Creating visualization in CANONICAL space...")
    logger.info(f"[Visibility Viz Canonical] {len(canonical_coords)} latent points, {len(canonical_camera_poses)} cameras")
    logger.info(f"[Visibility Viz Canonical] Latent coords range: "
                f"X=[{canonical_coords[:, 0].min():.4f}, {canonical_coords[:, 0].max():.4f}], "
                f"Y=[{canonical_coords[:, 1].min():.4f}, {canonical_coords[:, 1].max():.4f}], "
                f"Z=[{canonical_coords[:, 2].min():.4f}, {canonical_coords[:, 2].max():.4f}]")
    
    # Compute visibility score for each point (average across all views)
    visibility_scores = visibility_matrix.mean(axis=1)  # (N,)
    
    # Create scene
    scene = trimesh.Scene()
    
    # Color latent points by visibility score
    # 0.0 = red (invisible), 1.0 = green (fully visible)
    colors = np.zeros((len(canonical_coords), 3), dtype=np.uint8)
    colors[:, 0] = (255 * (1.0 - visibility_scores)).astype(np.uint8)  # Red component
    colors[:, 1] = (255 * visibility_scores).astype(np.uint8)  # Green component
    colors[:, 2] = 0  # Blue component
    
    # Create point cloud
    point_cloud = trimesh.PointCloud(vertices=canonical_coords, colors=colors)
    scene.add_geometry(point_cloud, node_name="latent_points")
    
    # Different color for each camera
    view_colors = [
        [255, 0, 0, 200],    # View 0: Red
        [0, 255, 0, 200],    # View 1: Green
        [0, 0, 255, 200],    # View 2: Blue
        [255, 255, 0, 200],  # View 3: Yellow
        [255, 0, 255, 200],  # View 4: Magenta
        [0, 255, 255, 200],  # View 5: Cyan
        [255, 128, 0, 200],  # View 6: Orange
        [128, 0, 255, 200],  # View 7: Purple
    ]
    
    # Add cameras (in canonical space) - simple sphere+direction line visualization
    for cam_idx, cam_pose in enumerate(canonical_camera_poses):
        camera_pos = cam_pose['camera_position']
        camera_forward = cam_pose.get('camera_forward', np.array([0, 0, 1]))
        
        color = view_colors[cam_idx % len(view_colors)]
        
        # Camera position marker (small sphere)
        camera_sphere = trimesh.creation.icosphere(subdivisions=1, radius=scale * 0.02)
        camera_sphere.apply_translation(camera_pos)
        camera_sphere.visual.face_colors = color
        scene.add_geometry(camera_sphere, node_name=f"camera_{cam_idx}")
        
        # Camera direction line (from camera position towards object)
        line_length = scale * 0.15
        line_end = camera_pos + camera_forward * line_length
        line_verts = np.array([camera_pos, line_end])
        line_colors = np.array([color, color])
        line_pc = trimesh.PointCloud(vertices=line_verts, colors=line_colors)
        scene.add_geometry(line_pc, node_name=f"camera_{cam_idx}_dir")
    
    # Add coordinate axes (length scaled accordingly)
    axis_length = scale * 0.3
    axis_vertices = np.array([
        [0, 0, 0], [axis_length, 0, 0],  # X
        [0, 0, 0], [0, axis_length, 0],  # Y
        [0, 0, 0], [0, 0, axis_length],  # Z
    ])
    axis_colors = np.array([
        [255, 0, 0, 255], [255, 0, 0, 255],  # X - Red
        [0, 255, 0, 255], [0, 255, 0, 255],  # Y - Green (up)
        [0, 0, 255, 255], [0, 0, 255, 255],  # Z - Blue
    ])
    axis_pc = trimesh.PointCloud(axis_vertices, colors=axis_colors)
    scene.add_geometry(axis_pc, node_name="canonical_axes")
    
    # Save
    scene.export(str(output_path))
    logger.info(f"[Visibility Viz Canonical] Saved to: {output_path}")
    
    return output_path


def visualize_visibility_per_view(
    latent_coords: np.ndarray,
    visibility_matrix: np.ndarray,
    object_pose: dict,
    camera_poses: List[dict],
    output_dir: Path,
    num_views_per_image: int = 6,
) -> None:
    """
    Generate one image per view showing visible latent points from that view.
    Each image contains multiple view renderings (grid layout) to show visible parts.
    
    Args:
        latent_coords: Latent coordinates (N, 4) or (N, 3)
        visibility_matrix: Visibility matrix (N, N_views)
        object_pose: Object pose
        camera_poses: List of camera poses
        output_dir: Output directory
        num_views_per_image: Number of views per image (grid layout)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.error("matplotlib not installed, cannot create visibility images")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_views = visibility_matrix.shape[1]
    num_latents = visibility_matrix.shape[0]
    
    logger.info(f"[Visibility Images] Creating {num_views} visibility images...")
    
    # Use unified coordinate transform function (same as visualize_object_with_cameras)
    # Transform chain: canonical (Z-up) -> Y-up -> scale -> rotate -> translate
    world_coords = apply_sam3d_pose_to_latent_coords(latent_coords, object_pose)
    
    # Compute grid layout (2 columns x 3 rows = 6 views)
    n_cols = 2
    n_rows = (num_views_per_image + n_cols - 1) // n_cols
    
    # Generate one image for each view
    for view_idx in range(num_views):
        # Visibility for this view
        view_visibility = visibility_matrix[:, view_idx]  # (N,)
        
        # Create figure with multiple subplots (multiple view renderings)
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        fig.suptitle(f'Visibility from View {view_idx}\n(Red=Invisible, Green=Visible)', 
                     fontsize=14, fontweight='bold')
        
        # Show num_views_per_image view renderings
        views_to_show = min(num_views_per_image, num_views)
        
        for subplot_idx in range(views_to_show):
            ax = fig.add_subplot(n_rows, n_cols, subplot_idx + 1, projection='3d')
            
            # Use different view visibility to filter points
            if subplot_idx < num_views:
                subplot_view_visibility = visibility_matrix[:, subplot_idx]
            else:
                subplot_view_visibility = view_visibility
            
            # Only show visible points (visibility > 0)
            visible_mask = subplot_view_visibility > 0.5
            visible_coords = world_coords[visible_mask]
            visible_scores = subplot_view_visibility[visible_mask]
            
            if len(visible_coords) > 0:
                # Color by visibility score
                colors_visible = np.zeros((len(visible_coords), 3))
                colors_visible[:, 0] = (1.0 - visible_scores)  # Red (invisible)
                colors_visible[:, 1] = visible_scores  # Green (visible)
                colors_visible[:, 2] = 0  # Blue
                
                # Draw point cloud
                ax.scatter(visible_coords[:, 0], visible_coords[:, 1], visible_coords[:, 2],
                          c=colors_visible, s=1, alpha=0.6)
            
            # Draw camera position and direction (if available)
            if subplot_idx < len(camera_poses):
                cam_pose = camera_poses[subplot_idx]
                camera_pos = cam_pose.get('camera_position')
                c2w = cam_pose.get('c2w')
                
                if camera_pos is not None:
                    # Camera position
                    ax.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]], 
                              c='magenta', s=50, marker='o', label='Camera')
                    
                    # Camera direction (Z axis)
                    if c2w is not None:
                        camera_z = c2w[:3, 2]
                        camera_end = camera_pos + camera_z * 0.1
                        ax.plot([camera_pos[0], camera_end[0]], 
                               [camera_pos[1], camera_end[1]], 
                               [camera_pos[2], camera_end[2]], 
                               'c-', linewidth=2, label='View Dir')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'View {subplot_idx}' + 
                        (f' (Visible: {visible_mask.sum()}/{num_latents})' if len(visible_coords) > 0 else ' (No visible points)'))
            ax.legend()
        
        plt.tight_layout()
        
        # Save image
        output_file = output_dir / f"visibility_view_{view_idx:02d}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[Visibility Images] Saved view {view_idx} visibility image: {output_file}")
    
    logger.info(f"[Visibility Images] Created {num_views} visibility images in {output_dir}")


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
    # Per-view pose optimization
    optimize_per_view_pose: bool = False,
    # Camera pose estimation
    estimate_camera_pose: bool = False,
    pose_refine_steps: int = 50,
    camera_pose_mode: str = "fixed_shape",
    # Latent visibility computation (uses self-occlusion / DDA ray tracing)
    enable_latent_visibility: bool = False,
    self_occlusion_tolerance: float = 4.0,  # neighbor tolerance for visibility DDA
    # Weight source selection
    weight_source: str = "entropy",  # "entropy", "visibility", "mixed"
    visibility_alpha: float = 30.0,  # Alpha for visibility weighting
    weight_combine_mode: str = "average",  # "average" or "multiply"
    visibility_weight_ratio: float = 0.5,  # Ratio for averaging in mixed mode
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
    da3_intrinsics = None  # Camera intrinsics (if available)
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
        
        # Load intrinsics if available
        if "intrinsics" in da3_data:
            da3_intrinsics = da3_data["intrinsics"]
            logger.info(f"  DA3 intrinsics shape: {da3_intrinsics.shape}")
        
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
    
    # Check parameter conflicts
    # 1. --merge_da3_glb requires --da3_output
    if merge_da3_glb and da3_output_path is None:
        raise ValueError(
            "Parameter conflict: --merge_da3_glb requires --da3_output.\n"
            "  --merge_da3_glb needs DA3's scene.glb to merge with SAM3D output.\n"
            "  Please provide: --da3_output <path_to_da3_output.npz>\n"
            "  Or remove --merge_da3_glb."
        )
    
    # 2. --optimize_per_view_pose requires --da3_output with extrinsics
    if optimize_per_view_pose:
        if da3_extrinsics is None:
            raise ValueError(
                "Parameter conflict: --optimize_per_view_pose requires --da3_output with valid extrinsics.\n"
                "  --optimize_per_view_pose needs camera extrinsics to visualize multi-view pose consistency.\n"
                "  Please provide: --da3_output <path_to_da3_output.npz>\n"
                "  Or remove --optimize_per_view_pose to use default mode."
            )
        logger.info("Per-view pose optimization enabled: each view will iterate its own pose")
    
    output_dir = get_output_dir(input_path, mask_prompt, image_names, is_single_view, use_weighting, entropy_alpha)
    
    # Setup logging
    log_file = output_dir / "inference.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
    )
    
    decode_formats = decode_formats or ["gaussian", "mesh"]
    
    # Create visibility callback if needed (uses self-occlusion / DDA ray tracing)
    visibility_callback = None
    if weight_source in ["visibility", "mixed"]:
        if da3_extrinsics is None:
            raise ValueError(
                f"weight_source='{weight_source}' requires DA3 output with camera extrinsics.\n"
                f"Please provide --da3_output with valid extrinsics."
            )
        
        # Create callback for visibility based on self-occlusion (DDA ray tracing)
        def create_visibility_callback(camera_poses_data, tolerance):
            """Create a visibility callback using self-occlusion (DDA ray tracing)."""
            def visibility_callback_impl(downsampled_coords, num_views, object_pose):
                """
                Compute visibility matrix for downsampled coords using self-occlusion.
                
                Args:
                    downsampled_coords: (N, 4) coords in voxel space [batch, z, y, x]
                    num_views: Number of views
                    object_pose: Dict with 'scale', 'rotation', 'translation'
                
                Returns:
                    visibility_matrix: (num_views, N) matrix where 1=visible, 0=self-occluded
                """
                from scipy.spatial.transform import Rotation
                
                # Helper to convert tensors to numpy
                def _to_np(x):
                    if hasattr(x, 'cpu'):
                        return x.cpu().numpy()
                    return np.array(x)
                
                # Convert object_pose tensors to numpy
                obj_scale = _to_np(object_pose.get('scale', [1, 1, 1])).flatten()
                obj_rotation = _to_np(object_pose.get('rotation', [1, 0, 0, 0])).flatten()
                obj_translation = _to_np(object_pose.get('translation', [0, 0, 0])).flatten()
                
                # Extract camera positions in canonical space
                camera_positions_canonical = []
                for cam_pose in camera_poses_data:
                    # Transform camera from world to canonical space
                    cam_pos_world = np.array(cam_pose.get('camera_position', [0, 0, 0])).flatten()
                    
                    # World to canonical: inverse of object pose
                    # canonical_pos = R_obj^T @ (world_pos - t_obj)
                    R_obj = Rotation.from_quat([obj_rotation[1], obj_rotation[2], 
                                                obj_rotation[3], obj_rotation[0]]).as_matrix()
                    cam_pos_canonical = R_obj.T @ (cam_pos_world - obj_translation)
                    camera_positions_canonical.append(cam_pos_canonical)
                
                # Compute self-occlusion for all views
                # visibility_matrix: (N, num_views) where 1=visible, 0=occluded
                visibility_matrix = compute_self_occlusion_for_all_views(
                    latent_coords=downsampled_coords,
                    camera_positions_canonical=camera_positions_canonical,
                    scale=float(obj_scale[0]),
                    grid_size=64,
                    neighbor_tolerance=tolerance,
                )
                
                # Transpose to (num_views, N) to match expected format
                vis_matrix = visibility_matrix.T
                return vis_matrix
            
            return visibility_callback_impl
        
        # Convert da3_extrinsics to camera_poses format
        camera_poses_for_visibility = convert_da3_extrinsics_to_camera_poses(da3_extrinsics)
        logger.info(f"Converted {len(camera_poses_for_visibility)} camera poses for visibility callback")
        
        visibility_callback = create_visibility_callback(
            camera_poses_data=camera_poses_for_visibility,
            tolerance=self_occlusion_tolerance,
        )
        logger.info(f"Visibility callback created (self-occlusion based), tolerance={self_occlusion_tolerance}")
    
    # Setup weighting config
    weighting_config = WeightingConfig(
        use_entropy=use_weighting,
        entropy_alpha=entropy_alpha,
        attention_layer=attention_layer,
        attention_step=attention_step,
        min_weight=min_weight,
        # Visibility-related parameters
        weight_source=weight_source,
        visibility_alpha=visibility_alpha,
        weight_combine_mode=weight_combine_mode,
        visibility_weight_ratio=visibility_weight_ratio,
        visibility_callback=visibility_callback,
    )
    
    logger.info(f"Weighting config: use_weighting={use_weighting}, weight_source={weight_source}, "
                f"entropy_alpha={entropy_alpha}, visibility_alpha={visibility_alpha}, "
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
            # Per-view pose optimization
            optimize_per_view_pose=optimize_per_view_pose,
        )
        weight_manager = result.get("weight_manager")
        
        # Log if stage2_init was saved
        if save_stage2_init and (output_dir / "stage2_init.pt").exists():
            logger.info(f"Stage 2 initial latent saved to: {output_dir / 'stage2_init.pt'}")
        
        # Camera pose estimation (Stage 2: refine pose per view)
        if estimate_camera_pose:
            logger.info("=" * 60)
            logger.info(f"Camera Pose Estimation (mode: {camera_pose_mode})")
            logger.info("=" * 60)
            
            # Get view_ss_input_dicts (need to get from result)
            view_ss_input_dicts = result.get('view_ss_input_dicts', None)
            if view_ss_input_dicts is None:
                logger.warning("view_ss_input_dicts not found in result, cannot estimate poses")
            else:
                # Use unified entry function, support all modes
                fixed_shape_latent = None
                if camera_pose_mode != "independent":
                    if 'shape' not in result:
                        logger.warning(f"shape not found in result, cannot use {camera_pose_mode} mode")
                    else:
                        fixed_shape_latent = result['shape']
                
                # If shape check passes (or independent mode), execute estimation
                if camera_pose_mode == "independent" or fixed_shape_latent is not None:
                    all_view_poses_raw = inference._pipeline.estimate_camera_poses_with_mode(
                        view_ss_input_dicts=view_ss_input_dicts,
                        mode=camera_pose_mode,
                        fixed_shape_latent=fixed_shape_latent,
                        inference_steps=pose_refine_steps,
                    )
                else:
                    logger.error(f"[Camera Pose] Cannot use {camera_pose_mode} mode: missing shape")
                    all_view_poses_raw = None
                
                if all_view_poses_raw is not None and len(all_view_poses_raw) > 0:
                    # Decode pose for each view
                    all_view_poses_decoded = inference._pipeline._decode_all_view_poses(
                        # Convert list to dict format
                        {
                            key: torch.stack([pose[key] for pose in all_view_poses_raw])
                            for key in all_view_poses_raw[0].keys()
                        },
                        view_ss_input_dicts,
                    )
                    
                    # Compute average scale
                    scales = [np.array(pose['scale']).flatten()[:3] for pose in all_view_poses_decoded]
                    avg_scale = np.mean(scales, axis=0)
                    scale_std = np.std(scales, axis=0)
                    logger.info(f"[Camera Pose] Average scale across views: {avg_scale}")
                    logger.info(f"[Camera Pose] Scale std: {scale_std}")
                    logger.info(f"[Camera Pose] Scale consistency: {'Good' if scale_std.max() < 0.1 else 'Poor'}")
                    
                    # Compute camera poses
                    camera_poses = compute_camera_poses_from_object_poses(all_view_poses_decoded)
                    
                    # Save results
                    result['refined_poses'] = all_view_poses_decoded
                    result['camera_poses'] = camera_poses
                    result['avg_scale'] = avg_scale
                    result['camera_pose_mode'] = camera_pose_mode
                    logger.info(f"[Camera Pose] Results saved to result dict (mode: {camera_pose_mode})")
                
                else:
                    logger.error(f"[Camera Pose] Failed to obtain view poses (mode: {camera_pose_mode}). all_view_poses_raw is None or empty.")
        
        # Latent visibility computation
        # Note: visibility analysis only uses DA3 GT camera poses, not estimated poses
        if enable_latent_visibility:
            logger.info("=" * 60)
            logger.info("Computing Latent Visibility (using DA3 GT camera poses)")
            logger.info("=" * 60)
            
            # Check required data
            if 'coords' not in result:
                logger.warning("[Visibility] No 'coords' found in result, cannot compute visibility")
            elif da3_extrinsics is None:
                logger.warning("[Visibility] Cannot compute visibility: requires --da3_output with extrinsics. "
                             "Visibility analysis only uses GT camera poses, not estimated poses.")
            else:
                latent_coords = result['coords']  # (N, 4) or (N, 3)
                
                # Get object pose (use Stage 1 output, this is the true object pose)
                # Stage 2 refined_poses is for camera pose estimation, not the true object pose
                object_pose = {}
                if 'scale' in result:
                    object_pose['scale'] = result['scale'].cpu().numpy() if torch.is_tensor(result['scale']) else result['scale']
                if 'rotation' in result:
                    object_pose['rotation'] = result['rotation'].cpu().numpy() if torch.is_tensor(result['rotation']) else result['rotation']
                if 'translation' in result:
                    object_pose['translation'] = result['translation'].cpu().numpy() if torch.is_tensor(result['translation']) else result['translation']
                logger.info("[Visibility] Using Stage 1 pose for object (true object pose)")
                
                if not object_pose:
                    logger.warning("[Visibility] No object pose found in result, using default")
                    object_pose = {
                        'scale': np.array([1.0, 1.0, 1.0]),
                        'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
                        'translation': np.array([0.0, 0.0, 0.0]),
                    }
                
                # Only use DA3 GT camera poses (not estimated poses)
                # Note: need to transform DA3 GT camera poses to View 0 coordinate system (consistent with object pose)
                logger.info("[Visibility] Using DA3 GT camera poses (extrinsics)")
                camera_poses_da3 = convert_da3_extrinsics_to_camera_poses(da3_extrinsics)
                
                # Transform DA3 camera poses to View 0 coordinate system
                # DA3 View 0 w2c represents: DA3 world coordinates -> View 0 camera coordinates
                # So w2c_view0 @ P_world = P_view0
                # For camera i, its c2w in DA3 world coordinates is c2w_i_world
                # To transform to View 0 coordinate system: c2w_i_view0 = w2c_view0 @ c2w_i_world
                w2c_view0_da3 = da3_extrinsics[0]  # DA3 View 0 w2c (3, 4) or (4, 4)
                if w2c_view0_da3.shape == (3, 4):
                    w2c_view0_da3_44 = np.eye(4)
                    w2c_view0_da3_44[:3, :] = w2c_view0_da3
                    w2c_view0_da3 = w2c_view0_da3_44
                
                # Transform all DA3 camera poses to View 0 coordinate system, from OpenCV to PyTorch3D space
                # 
                # Coordinate system notes：
                # - DA3 extrinsics in OpenCV space: X-right, Y-down, Z-forward
                # - SAM3D pose in PyTorch3D space: X-left, Y-up, Z-forward
                # - Need OpenCV -> PyTorch3D transform: (-x, -y, z)
                #
                # For c2w matrix, need to transform both rotation and translation:
                # M_p3d = M_cv_to_p3d @ M_cv @ M_p3d_to_cv
                # where M_cv_to_p3d = diag(-1, -1, 1)
                opencv_to_pytorch3d = np.diag([-1.0, -1.0, 1.0])
                
                camera_poses = []
                for da3_cam_pose in camera_poses_da3:
                    # DA3 camera c2w in DA3 world coordinates
                    c2w_da3_world = da3_cam_pose['c2w']
                    
                    # Transform to View 0 coordinate system (still in OpenCV space)
                    # P_view0 = w2c_view0 @ P_world = w2c_view0 @ c2w_i_world @ P_cam_i
                    # So c2w_i_view0 = w2c_view0 @ c2w_i_world
                    c2w_view0_cv = w2c_view0_da3 @ c2w_da3_world
                    
                    # Transform from OpenCV space to PyTorch3D space
                    # Rotation: R_p3d = M @ R_cv @ M^T
                    # Translation: t_p3d = M @ t_cv
                    R_cv = c2w_view0_cv[:3, :3]
                    t_cv = c2w_view0_cv[:3, 3]
                    
                    R_p3d = opencv_to_pytorch3d @ R_cv @ opencv_to_pytorch3d.T
                    t_p3d = opencv_to_pytorch3d @ t_cv
                    
                    c2w_view0 = np.eye(4)
                    c2w_view0[:3, :3] = R_p3d
                    c2w_view0[:3, 3] = t_p3d
                    
                    camera_poses.append({
                        'view_idx': da3_cam_pose['view_idx'],
                        'c2w': c2w_view0,
                        'w2c': np.linalg.inv(c2w_view0),
                        'R_c2w': c2w_view0[:3, :3],
                        't_c2w': c2w_view0[:3, 3],
                        'camera_position': c2w_view0[:3, 3],
                    })
                
                logger.info("[Visibility] Converted DA3 GT camera poses to View 0 coordinate system (PyTorch3D space)")
                
                # Compute visibility (using self-occlusion / DDA ray tracing)
                visibility_result = compute_latent_visibility(
                    latent_coords=latent_coords.cpu().numpy() if torch.is_tensor(latent_coords) else latent_coords,
                    object_pose=object_pose,
                    camera_poses=camera_poses,
                    self_occlusion_tolerance=self_occlusion_tolerance,
                )
                
                if visibility_result is not None:
                    result['latent_visibility'] = visibility_result['visibility_matrix']
                    result['visibility_canonical_data'] = visibility_result  # Save full data for subsequent analysis
                    
                    # GLB visualization (in canonical space) - latent_visibility.glb
                    viz_path = visualize_latent_visibility(
                        visibility_result=visibility_result,
                        output_path=output_dir / "latent_visibility.glb",
                    )
                    
                    if viz_path:
                        logger.info(f"✓ Latent visibility GLB (canonical) saved to: {viz_path}")
                    
                    # Per-view visibility GLB visualization (one GLB file per view)
                    viz_dir = output_dir / "latent_visibility_per_view"
                    visualize_self_occlusion_per_view(
                        self_occlusion_matrix=visibility_result['visibility_matrix'],
                        visibility_result=visibility_result,
                        output_dir=viz_dir,
                    )
                    
                    if viz_dir.exists():
                        logger.info(f"✓ Latent visibility per-view GLB files saved to: {viz_dir}")
                    
                    # Statistics
                    visibility_matrix = visibility_result['visibility_matrix']
                    for view_idx in range(visibility_matrix.shape[1]):
                        visible_ratio = visibility_matrix[:, view_idx].mean()
                        logger.info(f"  View {view_idx}: {visible_ratio*100:.1f}% visible")
                    
                else:
                    logger.warning("[Visibility] Visibility computation returned None")
    
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
            # Note: SAM3D pose parameters are already in real-world scale
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
        # Only overlay on actually used pointmaps
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
                    # Try to read scale/shift from per-view input
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
    
    # Save camera pose estimation results (only in estimate_camera_pose mode)
    if 'camera_poses' in result and 'refined_poses' in result:
        import json
        
        # Prepare JSON data
        estimated_data = {
            "num_views": len(result['refined_poses']),
            "mode": result.get('camera_pose_mode', 'unknown'),
            "avg_scale": result['avg_scale'].tolist() if isinstance(result['avg_scale'], np.ndarray) else result['avg_scale'],
            "object_poses": [],
            "camera_poses": [],
        }
        
        for pose in result['refined_poses']:
            pose_data = {}
            for key, value in pose.items():
                if isinstance(value, np.ndarray):
                    pose_data[key] = value.tolist()
                else:
                    pose_data[key] = value
            estimated_data["object_poses"].append(pose_data)
        
        for cam_pose in result['camera_poses']:
            cam_data = {
                "view_idx": cam_pose['view_idx'],
                "camera_position": cam_pose['camera_position'].tolist(),
                "R_c2w": cam_pose['R_c2w'].tolist(),
                "t_c2w": cam_pose['t_c2w'].tolist(),
                "c2w": cam_pose['c2w'].tolist(),
                "w2c": cam_pose['w2c'].tolist(),
            }
            estimated_data["camera_poses"].append(cam_data)
        
        # Save JSON
        estimated_path = output_dir / "estimated_poses.json"
        with open(estimated_path, 'w') as f:
            json.dump(estimated_data, f, indent=2)
        saved_files.append("estimated_poses.json")
        print(f"✓ Estimated poses saved to: {estimated_path}")
        
        # Create object+camera visualization
        if glb_path is not None and glb_path.exists():
            # Use Stage 1 pose (multi-view fusion result) as object pose in world coordinates
            # Note: refined_poses is Camera Pose Estimation step output, for camera pose estimation
            # But object should use Stage 1 pose, consistent with result_cameras_aligned_with_gt.glb
            object_pose = {}
            if 'scale' in result:
                object_pose['scale'] = result['scale'].cpu().numpy() if torch.is_tensor(result['scale']) else result['scale']
            if 'rotation' in result:
                object_pose['rotation'] = result['rotation'].cpu().numpy() if torch.is_tensor(result['rotation']) else result['rotation']
            if 'translation' in result:
                object_pose['translation'] = result['translation'].cpu().numpy() if torch.is_tensor(result['translation']) else result['translation']
            
            # Auto-adjust camera frustum size based on object size
            avg_scale = result['avg_scale']
            if isinstance(avg_scale, np.ndarray):
                obj_size = avg_scale.mean()
            else:
                obj_size = np.mean(avg_scale)
            camera_scale = max(0.05, obj_size * 0.3)  # Camera frustum size ~30% of object
            logger.info(f"[Viz] Object size: {obj_size:.4f}, camera scale: {camera_scale:.4f}")
            
            viz_path = visualize_object_with_cameras(
                sam3d_glb_path=glb_path,
                object_pose=object_pose,
                camera_poses=result['camera_poses'],
                output_path=output_dir / "result_with_cameras.glb",
                camera_scale=camera_scale,
            )
            if viz_path:
                saved_files.append("result_with_cameras.glb")
                print(f"✓ Object with cameras visualization saved to: {viz_path}")
        
        # Evaluate and visualize GT camera pose comparison (if DA3 GT available)
        if 'camera_poses' in result and da3_extrinsics is not None:
            logger.info("=" * 60)
            logger.info("Evaluating Camera Poses against DA3 GT")
            logger.info("=" * 60)
            
            # Transform DA3 GT camera poses
            gt_camera_poses = convert_da3_extrinsics_to_camera_poses(da3_extrinsics)
            
            # Evaluate camera poses
            from scipy.spatial.transform import Rotation
            
            # Prepare evaluation data
            est_positions = np.array([pose['camera_position'] for pose in result['camera_poses']])
            gt_positions = np.array([pose['camera_position'] for pose in gt_camera_poses])
            
            # Use Umeyama alignment for positions (without changing rotation)
            from scipy.linalg import orthogonal_procrustes
            
            # Center
            est_centered = est_positions - est_positions.mean(axis=0)
            gt_centered = gt_positions - gt_positions.mean(axis=0)
            
            # Compute similarity transform (scale, rotation, translation)
            # Use simplified version: only align positions
            H = est_centered.T @ gt_centered
            U, S, Vt = np.linalg.svd(H)
            R_align = Vt.T @ U.T
            if np.linalg.det(R_align) < 0:
                Vt[-1, :] *= -1
                R_align = Vt.T @ U.T
            
            # Compute scale
            # Correct method: scale = sqrt(sum(|gt_centered|^2) / sum(|R_align @ est_centered.T|^2))
            # Or use trace: trace(gt_centered.T @ (R_align @ est_centered.T)) / trace(est_centered.T @ est_centered)
            R_est_centered = R_align @ est_centered.T  # (3, 3) @ (3, N) = (3, N)
            scale_align = np.sqrt(np.sum(gt_centered ** 2) / np.sum(R_est_centered ** 2))
            
            # Compute translation
            t_align = gt_positions.mean(axis=0) - scale_align * R_align @ est_positions.mean(axis=0)
            
            # Aligned estimated positions
            est_positions_aligned = scale_align * (R_align @ est_positions.T).T + t_align
            
            # Compute error (after alignment)
            position_errors = np.linalg.norm(est_positions_aligned - gt_positions, axis=1)
            position_rmse = np.sqrt(np.mean(position_errors ** 2))
            
            # Compute rotation error (no alignment needed)
            rotation_errors = []
            for est_pose, gt_pose in zip(result['camera_poses'], gt_camera_poses):
                R_est = est_pose['R_c2w']
                R_gt = gt_pose['R_c2w']
                R_rel = R_est @ R_gt.T
                rot = Rotation.from_matrix(R_rel)
                angle = np.abs(rot.as_rotvec())
                rotation_errors.append(np.linalg.norm(angle) * 180 / np.pi)
            rotation_rmse = np.sqrt(np.mean(np.array(rotation_errors) ** 2))
            
            logger.info(f"[Camera Pose Eval] Position RMSE (aligned): {position_rmse:.4f} m")
            logger.info(f"[Camera Pose Eval] Rotation RMSE: {rotation_rmse:.4f} deg")
            
            # Save evaluation results
            eval_data = {
                "position_rmse_m": float(position_rmse),
                "rotation_rmse_deg": float(rotation_rmse),
                "alignment": {
                    "scale": float(scale_align),
                    "R": R_align.tolist(),
                    "t": t_align.tolist(),
                },
                "per_view_errors": {
                    "position_errors_m": position_errors.tolist(),
                    "rotation_errors_deg": rotation_errors,
                }
            }
            
            import json
            eval_path = output_dir / "camera_pose_evaluation.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_data, f, indent=2)
            saved_files.append("camera_pose_evaluation.json")
            print(f"✓ Camera pose evaluation saved to: {eval_path}")
            
            # Visualize aligned GT and estimated camera poses
            # Note: for visualization, object and cameras should be in View 0 coordinate system
            if glb_path is not None and glb_path.exists():
                # Use Stage 1 object pose (this is the true object pose)
                # Stage 2 refined_poses is for camera pose estimation, not the true object pose
                object_pose = {}
                if 'scale' in result:
                    object_pose['scale'] = result['scale'].cpu().numpy() if torch.is_tensor(result['scale']) else result['scale']
                if 'rotation' in result:
                    object_pose['rotation'] = result['rotation'].cpu().numpy() if torch.is_tensor(result['rotation']) else result['rotation']
                if 'translation' in result:
                    object_pose['translation'] = result['translation'].cpu().numpy() if torch.is_tensor(result['translation']) else result['translation']
                
                # Transform GT camera poses to View 0 coordinate system (consistent with object pose)
                w2c_view0_da3 = da3_extrinsics[0]
                if w2c_view0_da3.shape == (3, 4):
                    w2c_view0_da3_44 = np.eye(4)
                    w2c_view0_da3_44[:3, :] = w2c_view0_da3
                    w2c_view0_da3 = w2c_view0_da3_44
                
                # Transform GT camera poses to View 0 coordinate system, from OpenCV to PyTorch3D space
                opencv_to_pytorch3d = np.diag([-1.0, -1.0, 1.0])
                
                gt_camera_poses_view0 = []
                for gt_pose in gt_camera_poses:
                    # Transform to View 0 coordinate system (still in OpenCV space)
                    c2w_view0_cv = w2c_view0_da3 @ gt_pose['c2w']
                    
                    # Transform from OpenCV space to PyTorch3D space
                    R_cv = c2w_view0_cv[:3, :3]
                    t_cv = c2w_view0_cv[:3, 3]
                    R_p3d = opencv_to_pytorch3d @ R_cv @ opencv_to_pytorch3d.T
                    t_p3d = opencv_to_pytorch3d @ t_cv
                    
                    c2w_view0 = np.eye(4)
                    c2w_view0[:3, :3] = R_p3d
                    c2w_view0[:3, 3] = t_p3d
                    
                    gt_camera_poses_view0.append({
                        'view_idx': gt_pose['view_idx'],
                        'c2w': c2w_view0,
                        'w2c': np.linalg.inv(c2w_view0),
                        'R_c2w': c2w_view0[:3, :3],
                        't_c2w': c2w_view0[:3, 3],
                        'camera_position': c2w_view0[:3, 3],
                    })
                
                # Align estimated camera poses to GT coordinate system (for error calculation, but use View 0 for visualization)
                aligned_est_camera_poses = []
                for est_pose in result['camera_poses']:
                    # Align positions (align in View 0 coordinate system)
                    aligned_pos = scale_align * (R_align @ est_pose['camera_position']) + t_align
                    # Use rotation directly (no change)
                    aligned_c2w = est_pose['c2w'].copy()
                    aligned_c2w[:3, 3] = aligned_pos
                    
                    aligned_est_camera_poses.append({
                        'view_idx': est_pose['view_idx'],
                        'c2w': aligned_c2w,
                        'w2c': np.linalg.inv(aligned_c2w),
                        'R_c2w': aligned_c2w[:3, :3],
                        't_c2w': aligned_pos,
                        'camera_position': aligned_pos,
                    })
                
                # Auto-adjust camera frustum size based on object size
                avg_scale = result['avg_scale']
                if isinstance(avg_scale, np.ndarray):
                    obj_size = avg_scale.mean()
                else:
                    obj_size = np.mean(avg_scale)
                camera_scale = max(0.05, obj_size * 0.3)
                
                # Visualize (object and cameras in View 0 coordinate system)
                aligned_viz_path = visualize_aligned_cameras_with_gt(
                    sam3d_glb_path=glb_path,
                    object_pose=object_pose,
                    estimated_camera_poses=aligned_est_camera_poses,
                    gt_camera_poses=gt_camera_poses_view0,  # Use transformed GT poses
                    output_path=output_dir / "result_cameras_aligned_with_gt.glb",
                    camera_scale=camera_scale,
                )
                
                if aligned_viz_path:
                    saved_files.append("result_cameras_aligned_with_gt.glb")
                    print(f"✓ Aligned cameras with GT visualization saved to: {aligned_viz_path}")
        
        # Multi-view overlay: when estimate_camera_pose=True and camera_pose_mode=independent
        # Generate overlay for each view, using unified shape and view-specific pose
        if result.get('camera_pose_mode') == 'independent' and glb_path is not None and glb_path.exists():
            logger.info("=" * 60)
            logger.info("Generating per-view overlays (independent mode)")
            logger.info("=" * 60)
            
            refined_poses = result['refined_poses']
            num_views = len(refined_poses)
            
            # Get pointmap for each view
            raw_view_pointmaps = result.get('raw_view_pointmaps', [])
            view_ss_input_dicts = result.get('view_ss_input_dicts', [])
            
            if len(raw_view_pointmaps) < num_views and len(view_ss_input_dicts) < num_views:
                logger.warning(f"Not enough pointmaps for all views: "
                              f"raw_view_pointmaps={len(raw_view_pointmaps)}, "
                              f"view_ss_input_dicts={len(view_ss_input_dicts)}, "
                              f"num_views={num_views}")
            else:
                for view_idx in range(num_views):
                    # Get pose for this view
                    view_pose = refined_poses[view_idx]
                    sam3d_pose = {}
                    for key in ['scale', 'rotation', 'translation']:
                        if key in view_pose:
                            value = view_pose[key]
                            if isinstance(value, np.ndarray):
                                sam3d_pose[key] = value.flatten()
                            else:
                                sam3d_pose[key] = np.array(value).flatten()
                    
                    # Get pointmap for this view
                    pointmap_data = None
                    pm_scale_np = None
                    pm_shift_np = None
                    
                    if view_idx < len(raw_view_pointmaps) and raw_view_pointmaps[view_idx] is not None:
                        # raw_view_pointmaps format: (H, W, 3), need to convert to (3, H, W)
                        pointmap_data = raw_view_pointmaps[view_idx]
                        if pointmap_data.ndim == 3 and pointmap_data.shape[-1] == 3:
                            pointmap_data = pointmap_data.transpose(2, 0, 1)  # HWC -> CHW
                        logger.info(f"[Overlay View {view_idx}] Using raw_view_pointmaps (metric)")
                    elif view_idx < len(view_ss_input_dicts):
                        internal_pm = view_ss_input_dicts[view_idx].get('pointmap')
                        if internal_pm is not None:
                            pointmap_data = internal_pm
                            logger.info(f"[Overlay View {view_idx}] Using normalized pointmap from view_ss_input_dicts")
                        pm_scale = view_ss_input_dicts[view_idx].get('pointmap_scale')
                        pm_shift = view_ss_input_dicts[view_idx].get('pointmap_shift')
                        if pm_scale is not None:
                            pm_scale_np = pm_scale.detach().cpu().numpy() if torch.is_tensor(pm_scale) else np.array(pm_scale)
                        if pm_shift is not None:
                            pm_shift_np = pm_shift.detach().cpu().numpy() if torch.is_tensor(pm_shift) else np.array(pm_shift)
                    
                    if pointmap_data is not None:
                        # Get image for this view (for point cloud coloring)
                        view_image = view_images[view_idx] if view_images and view_idx < len(view_images) else None
                        
                        overlay_path = overlay_sam3d_on_pointmap(
                            glb_path,
                            pointmap_data,
                            sam3d_pose,
                            input_image=view_image,
                            output_path=output_dir / f"result_overlay_view{view_idx}.glb",
                            pointmap_scale=pm_scale_np,
                            pointmap_shift=pm_shift_np,
                        )
                        if overlay_path:
                            saved_files.append(f"result_overlay_view{view_idx}.glb")
                            print(f"✓ Overlay (View {view_idx}) saved to: {overlay_path}")
                    else:
                        logger.warning(f"[Overlay View {view_idx}] No pointmap available, skipping")
    
    # Save decoded pose for all views (only in optimize_per_view_pose mode)
    if 'all_view_poses_decoded' in result:
        all_poses_decoded = result['all_view_poses_decoded']
        import json
        
        # Save as JSON format
        all_poses_json = {
            "num_views": len(all_poses_decoded),
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
        
        # Create multi-view pose consistency visualization (requires DA3 extrinsics)
        if da3_extrinsics is not None and glb_path.exists():
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
        weights_downsampled = analysis_data.get("weights", {})  # Weights in downsampled dimension
        weights_expanded = analysis_data.get("expanded_weights", {})  # Expanded weights
        entropy_per_view = analysis_data.get("entropy_per_view", {})
        original_coords = analysis_data.get("original_coords")  # Original coords
        downsampled_coords = analysis_data.get("downsampled_coords")  # Downsampled coords
        downsample_idx = analysis_data.get("downsample_idx")  # Index mapping
        
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
                        s=0.5,  # Smaller points because more points
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
    
    # Overlay visualization - overlay SAM3D result on input pointmap
    parser.add_argument("--overlay_pointmap", action="store_true",
                        help="Overlay SAM3D result on input pointmap for pose visualization. "
                             "Works with both MoGe (default) and DA3 (if --da3_output is provided)")
    
    # Per-view pose optimization - optimize pose independently for each view
    parser.add_argument("--optimize_per_view_pose", action="store_true",
                        help="Optimize pose independently for each view. "
                             "When enabled, each view maintains and iterates its own pose. "
                             "Requires --da3_output for camera extrinsics to visualize multi-view consistency. "
                             "Outputs: all_view_poses_decoded.json, multiview_pose_consistency.glb")
    
    # Camera pose estimation - estimate camera poses
    parser.add_argument("--estimate_camera_pose", action="store_true",
                        help="Estimate camera poses from object poses. "
                             "Stage 2: Fix shape, refine pose for each view independently. "
                             "Then compute relative camera poses assuming the object is static. "
                             "Outputs: estimated_poses.json, result_with_cameras.glb")
    parser.add_argument("--pose_refine_steps", type=int, default=50,
                        help="Number of steps for pose refinement in Stage 2 (default: 50)")
    parser.add_argument("--camera_pose_mode", type=str, default="manual_sync_time",
                        choices=["fixed_shape", "independent", "manual_sync_time", "mixed_update"],
                        help="Mode for camera pose estimation: "
                             "'fixed_shape' (default): Fix multi-view fused shape, only refine pose. "
                             "'independent': Each view optimizes shape+pose independently from noise. "
                             "'manual_sync_time': Manually sync shape to same time step as pose (no iteration). "
                             "'mixed_update': Mixed update: network prediction + target guidance.")
    
    # Latent visibility computation
    parser.add_argument("--compute_latent_visibility", action="store_true",
                        help="Compute and visualize latent point visibility based on GT camera poses. "
                             "Requires --da3_output with extrinsics. Uses self-occlusion (DDA ray tracing) for visibility.")
    parser.add_argument("--self_occlusion_tolerance", type=float, default=4.0,
                        help="Tolerance for self-occlusion/visibility detection (in voxel units). "
                             "Ignore occluding voxels within this distance to avoid grazing angle issues. "
                             "Higher values = more lenient (fewer false occlusions). "
                             "Default: 4.0 (covers grazing angles well)")
    
    # Weight source selection
    parser.add_argument("--weight_source", type=str, default="entropy",
                        choices=["entropy", "visibility", "mixed"],
                        help="Source for multi-view fusion weights: "
                             "'entropy' (default): Use attention entropy only. "
                             "'visibility': Use self-occlusion based visibility (requires DA3 for camera poses). "
                             "'mixed': Combine entropy and visibility.")
    parser.add_argument("--visibility_alpha", type=float, default=30.0,
                        help="Alpha parameter for visibility weighting (higher = more contrast). Default: 30.0")
    parser.add_argument("--weight_combine_mode", type=str, default="average",
                        choices=["average", "multiply"],
                        help="How to combine entropy and visibility weights in 'mixed' mode: "
                             "'average': weighted average (1-r)*entropy + r*visibility. "
                             "'multiply': multiply then normalize. Default: 'average'")
    parser.add_argument("--visibility_weight_ratio", type=float, default=0.5,
                        help="Ratio for averaging in 'mixed' mode (0.0-1.0). "
                             "0.0 = entropy only, 1.0 = visibility only. Default: 0.5")
    
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
            optimize_per_view_pose=args.optimize_per_view_pose,
            estimate_camera_pose=args.estimate_camera_pose,
            pose_refine_steps=args.pose_refine_steps,
            camera_pose_mode=args.camera_pose_mode,
            enable_latent_visibility=args.compute_latent_visibility,
            self_occlusion_tolerance=args.self_occlusion_tolerance,
            weight_source=args.weight_source,
            visibility_alpha=args.visibility_alpha,
            weight_combine_mode=args.weight_combine_mode,
            visibility_weight_ratio=args.visibility_weight_ratio,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

