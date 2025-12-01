#!/usr/bin/env python3
"""
调试 overlay 对齐问题

直接加载 SAM3D 的输出，检查：
1. GLB 顶点的原始范围
2. pose 参数
3. 变换后的顶点范围
4. pointmap 的范围

看看它们是否应该对齐
"""

import sys
import numpy as np
import torch
from pathlib import Path

# 添加项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="SAM3D 输出目录")
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    print("=" * 70)
    print("Debug Overlay Alignment")
    print("=" * 70)
    print(f"Result dir: {result_dir}")
    
    # 1. 加载 GLB
    glb_path = result_dir / "result.glb"
    if not glb_path.exists():
        print(f"ERROR: GLB not found: {glb_path}")
        return
    
    import trimesh
    scene = trimesh.load(str(glb_path))
    
    print("\n1. GLB 原始顶点 (canonical space):")
    vertices = None
    if isinstance(scene, trimesh.Scene):
        print(f"  Scene contains {len(scene.geometry)} geometries:")
        for name, geom in scene.geometry.items():
            if hasattr(geom, 'vertices'):
                v = geom.vertices
                print(f"  {name}: {v.shape}")
                print(f"    X: [{v[:, 0].min():.4f}, {v[:, 0].max():.4f}]")
                print(f"    Y: [{v[:, 1].min():.4f}, {v[:, 1].max():.4f}]")
                print(f"    Z: [{v[:, 2].min():.4f}, {v[:, 2].max():.4f}]")
                if vertices is None:
                    vertices = v  # 使用第一个几何体
    else:
        vertices = scene.vertices
        print(f"  Shape: {vertices.shape}")
        print(f"  X: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}]")
        print(f"  Y: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}]")
        print(f"  Z: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]")
    
    # 2. 加载 pose 参数
    params_path = result_dir / "params.npz"
    if not params_path.exists():
        print(f"ERROR: params.npz not found: {params_path}")
        return
    
    params = np.load(params_path)
    print("\n2. Pose 参数:")
    print(f"  Keys: {list(params.keys())}")
    
    scale = params.get('scale', np.array([1.0, 1.0, 1.0]))
    rotation = params.get('rotation', np.array([1.0, 0.0, 0.0, 0.0]))
    translation = params.get('translation', np.array([0.0, 0.0, 0.0]))
    
    if len(scale.shape) > 1:
        scale = scale.flatten()
    if len(rotation.shape) > 1:
        rotation = rotation.flatten()
    if len(translation.shape) > 1:
        translation = translation.flatten()
    
    print(f"  scale: {scale}")
    print(f"  rotation (wxyz): {rotation}")
    print(f"  translation: {translation}")
    
    if 'pointmap_scale' in params:
        print(f"  pointmap_scale: {params['pointmap_scale']}")
    if 'pointmap_shift' in params:
        print(f"  pointmap_shift: {params['pointmap_shift']}")
    
    # 3. 计算变换后的顶点
    from scipy.spatial.transform import Rotation as R
    
    # 四元数转旋转矩阵 (wxyz -> xyzw)
    quat_xyzw = np.array([rotation[1], rotation[2], rotation[3], rotation[0]])
    R_mat = R.from_quat(quat_xyzw).as_matrix()
    
    print("\n3. 旋转矩阵:")
    print(R_mat)
    
    # z-up to y-up 旋转 (与官方 get_mesh 一致，使用 .T)
    R_zup_to_yup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    
    print("\n4. 变换步骤:")
    
    # Step 1: z-up to y-up (注意：官方用的是 v @ R.T)
    v1 = vertices @ R_zup_to_yup.T
    print(f"  After z-up to y-up:")
    print(f"    X: [{v1[:, 0].min():.4f}, {v1[:, 0].max():.4f}]")
    print(f"    Y: [{v1[:, 1].min():.4f}, {v1[:, 1].max():.4f}]")
    print(f"    Z: [{v1[:, 2].min():.4f}, {v1[:, 2].max():.4f}]")
    
    # Step 2: scale
    s = scale[0]
    v2 = v1 * s
    print(f"  After scale ({s:.4f}):")
    print(f"    X: [{v2[:, 0].min():.4f}, {v2[:, 0].max():.4f}]")
    print(f"    Y: [{v2[:, 1].min():.4f}, {v2[:, 1].max():.4f}]")
    print(f"    Z: [{v2[:, 2].min():.4f}, {v2[:, 2].max():.4f}]")
    
    # Step 3: rotate (v @ R)
    v3 = v2 @ R_mat
    print(f"  After rotation (v @ R):")
    print(f"    X: [{v3[:, 0].min():.4f}, {v3[:, 0].max():.4f}]")
    print(f"    Y: [{v3[:, 1].min():.4f}, {v3[:, 1].max():.4f}]")
    print(f"    Z: [{v3[:, 2].min():.4f}, {v3[:, 2].max():.4f}]")
    
    # Step 4: translate
    v4 = v3 + translation
    print(f"  After translation ({translation}):")
    print(f"    X: [{v4[:, 0].min():.4f}, {v4[:, 0].max():.4f}]")
    print(f"    Y: [{v4[:, 1].min():.4f}, {v4[:, 1].max():.4f}]")
    print(f"    Z: [{v4[:, 2].min():.4f}, {v4[:, 2].max():.4f}]")
    
    # 5. 检查 overlay GLB
    overlay_path = result_dir / "result_overlay.glb"
    if overlay_path.exists():
        print("\n5. Overlay GLB 内容:")
        overlay_scene = trimesh.load(str(overlay_path))
        if isinstance(overlay_scene, trimesh.Scene):
            for name, geom in overlay_scene.geometry.items():
                if hasattr(geom, 'vertices'):
                    v = geom.vertices
                    print(f"  {name}: {v.shape}")
                    print(f"    X: [{v[:, 0].min():.4f}, {v[:, 0].max():.4f}]")
                    print(f"    Y: [{v[:, 1].min():.4f}, {v[:, 1].max():.4f}]")
                    print(f"    Z: [{v[:, 2].min():.4f}, {v[:, 2].max():.4f}]")
                elif isinstance(geom, trimesh.PointCloud):
                    pts = geom.vertices
                    print(f"  {name} (PointCloud): {pts.shape}")
                    print(f"    X: [{pts[:, 0].min():.4f}, {pts[:, 0].max():.4f}]")
                    print(f"    Y: [{pts[:, 1].min():.4f}, {pts[:, 1].max():.4f}]")
                    print(f"    Z: [{pts[:, 2].min():.4f}, {pts[:, 2].max():.4f}]")
    
    # 6. 分析 pointmap 的坐标系
    print("\n6. Pointmap 坐标系分析:")
    if overlay_path.exists():
        overlay_scene = trimesh.load(str(overlay_path))
        if isinstance(overlay_scene, trimesh.Scene):
            for name, geom in overlay_scene.geometry.items():
                if isinstance(geom, trimesh.PointCloud) or (hasattr(geom, 'vertices') and 'pointcloud' in name.lower()):
                    pts = geom.vertices
                    print(f"  Pointcloud: {name}")
                    print(f"    Total points: {len(pts)}")
                    
                    # 分析 Z 方向的分布
                    z_values = pts[:, 2]
                    z_positive = (z_values > 0).sum()
                    z_negative = (z_values < 0).sum()
                    print(f"    Z positive: {z_positive} ({z_positive/len(pts)*100:.1f}%)")
                    print(f"    Z negative: {z_negative} ({z_negative/len(pts)*100:.1f}%)")
                    
                    # 分析 Y 方向的分布
                    y_values = pts[:, 1]
                    y_positive = (y_values > 0).sum()
                    y_negative = (y_values < 0).sum()
                    print(f"    Y positive: {y_positive} ({y_positive/len(pts)*100:.1f}%)")
                    print(f"    Y negative: {y_negative} ({y_negative/len(pts)*100:.1f}%)")
                    
                    # 计算点云的主方向（PCA）
                    try:
                        centered = pts - pts.mean(axis=0)
                        cov = np.cov(centered.T)
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        # 最小特征值对应的特征向量是法向量方向
                        normal = eigenvectors[:, 0]
                        print(f"    Pointcloud normal (smallest eigenvalue): {normal}")
                        print(f"    (如果是平面，这个向量应该接近 [0, 0, 1] 或 [0, 1, 0])")
                    except:
                        pass
    
    # 7. 分析对齐问题
    print("\n" + "=" * 70)
    print("分析")
    print("=" * 70)
    
    print(f"""
变换后的 SAM3D 物体应该在相机空间中。
- 如果 translation.z > 0，物体应该在相机前方
- pointmap 的 Z 也应该是正的（远离相机）

检查：
- SAM3D 变换后 Z 范围: [{v4[:, 2].min():.4f}, {v4[:, 2].max():.4f}]
- translation.z: {translation[2]:.4f}

如果 pointmap 和 SAM3D 物体的 Z 范围差很多，说明对齐有问题。

PyTorch3D 相机坐标系:
- X: 左(-)，右(+)  -- 但实际上是翻转的，所以是 左(+)，右(-)
- Y: 下(-)，上(+)  -- 但实际上是翻转的，所以是 下(+)，上(-)
- Z: 前(+)，后(-)

如果 pointmap 的 Y 大部分是正的，说明物体在相机下方。
如果 pointmap 的 Z 大部分是正的，说明物体在相机前方（正确）。
""")


if __name__ == "__main__":
    main()

