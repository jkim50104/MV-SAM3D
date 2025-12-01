#!/usr/bin/env python3
"""
对比 DA3 和 MoGe 的 pointmap，找出坐标系差异

用法：
    python scripts/compare_da3_moge_pointmap.py \
        --image data/example/images/0.png \
        --da3_output da3_outputs/example/da3_output.npz
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# 添加项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebook"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--da3_output", type=str, required=True, help="DA3 输出的 npz 文件")
    parser.add_argument("--view_idx", type=int, default=0, help="使用哪个视角的 DA3 pointmap")
    args = parser.parse_args()
    
    import torch
    from PIL import Image
    from pytorch3d.transforms import Transform3d
    
    # 加载图像
    image_path = Path(args.image)
    image = Image.open(image_path)
    image_np = np.array(image)
    print(f"Image shape: {image_np.shape}")
    
    # 加载 DA3 pointmap
    da3_data = np.load(args.da3_output)
    da3_pointmap = da3_data["pointmaps_sam3d"][args.view_idx]  # (3, H, W)
    print(f"\nDA3 pointmap shape: {da3_pointmap.shape}")
    
    # DA3 原始值（标准相机坐标系）
    print("\n" + "=" * 60)
    print("DA3 Pointmap (原始，标准相机坐标系)")
    print("=" * 60)
    print(f"  X: [{da3_pointmap[0].min():.4f}, {da3_pointmap[0].max():.4f}], mean={da3_pointmap[0].mean():.4f}")
    print(f"  Y: [{da3_pointmap[1].min():.4f}, {da3_pointmap[1].max():.4f}], mean={da3_pointmap[1].mean():.4f}")
    print(f"  Z: [{da3_pointmap[2].min():.4f}, {da3_pointmap[2].max():.4f}], mean={da3_pointmap[2].mean():.4f}")
    
    # 加载 MoGe 模型并计算 pointmap
    print("\n" + "=" * 60)
    print("加载 MoGe 模型...")
    print("=" * 60)
    
    from inference import Inference
    config_path = str(PROJECT_ROOT / "checkpoints/hf/pipeline.yaml")
    inference = Inference(config_path, compile=False)
    
    # 准备图像（需要 RGBA 格式）
    if image_np.shape[-1] == 3:
        # 添加 alpha 通道
        alpha = np.ones((image_np.shape[0], image_np.shape[1], 1), dtype=np.uint8) * 255
        image_rgba = np.concatenate([image_np, alpha], axis=-1)
    else:
        image_rgba = image_np
    
    # 计算 MoGe pointmap
    pipeline = inference._pipeline
    
    # 转换为 float 并加载
    loaded_image = pipeline.image_to_float(image_rgba)
    loaded_image = torch.from_numpy(loaded_image)
    loaded_image_3hw = loaded_image.permute(2, 0, 1).contiguous()[:3]
    
    print(f"Image tensor shape: {loaded_image_3hw.shape}")
    
    # 调用 MoGe
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=pipeline.dtype):
            output = pipeline.depth_model(loaded_image_3hw)
    
    moge_pointmap_raw = output["pointmaps"]  # (H, W, 3)
    print(f"\nMoGe pointmap shape (raw): {moge_pointmap_raw.shape}")
    
    # MoGe 原始值
    print("\n" + "=" * 60)
    print("MoGe Pointmap (原始，depth_model 输出)")
    print("=" * 60)
    print(f"  X: [{moge_pointmap_raw[..., 0].min():.4f}, {moge_pointmap_raw[..., 0].max():.4f}], mean={moge_pointmap_raw[..., 0].mean():.4f}")
    print(f"  Y: [{moge_pointmap_raw[..., 1].min():.4f}, {moge_pointmap_raw[..., 1].max():.4f}], mean={moge_pointmap_raw[..., 1].mean():.4f}")
    print(f"  Z: [{moge_pointmap_raw[..., 2].min():.4f}, {moge_pointmap_raw[..., 2].max():.4f}], mean={moge_pointmap_raw[..., 2].mean():.4f}")
    
    # 应用 camera_to_pytorch3d 变换
    from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
    
    camera_convention_transform = (
        Transform3d()
        .rotate(camera_to_pytorch3d_camera(device=pipeline.device).rotation)
        .to(pipeline.device)
    )
    
    moge_pointmap_p3d = camera_convention_transform.transform_points(moge_pointmap_raw)
    
    print("\n" + "=" * 60)
    print("MoGe Pointmap (变换后，PyTorch3D 坐标系)")
    print("=" * 60)
    print(f"  X: [{moge_pointmap_p3d[..., 0].min():.4f}, {moge_pointmap_p3d[..., 0].max():.4f}], mean={moge_pointmap_p3d[..., 0].mean():.4f}")
    print(f"  Y: [{moge_pointmap_p3d[..., 1].min():.4f}, {moge_pointmap_p3d[..., 1].max():.4f}], mean={moge_pointmap_p3d[..., 1].mean():.4f}")
    print(f"  Z: [{moge_pointmap_p3d[..., 2].min():.4f}, {moge_pointmap_p3d[..., 2].max():.4f}], mean={moge_pointmap_p3d[..., 2].mean():.4f}")
    
    # 对 DA3 应用修正后的变换
    # Step 1: 翻转 Y 和 Z，使 DA3 和 MoGe 原始输出一致
    da3_pointmap_hwc = da3_pointmap.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    da3_tensor = torch.from_numpy(da3_pointmap_hwc).float().to(pipeline.device)
    
    # 翻转 Y 和 Z
    da3_tensor_corrected = da3_tensor.clone()
    da3_tensor_corrected[..., 1] = -da3_tensor_corrected[..., 1]  # 翻转 Y
    da3_tensor_corrected[..., 2] = -da3_tensor_corrected[..., 2]  # 翻转 Z
    
    print("\n" + "=" * 60)
    print("DA3 Pointmap (翻转 Y/Z 后，现在和 MoGe 原始一致)")
    print("=" * 60)
    print(f"  X: [{da3_tensor_corrected[..., 0].min():.4f}, {da3_tensor_corrected[..., 0].max():.4f}], mean={da3_tensor_corrected[..., 0].mean():.4f}")
    print(f"  Y: [{da3_tensor_corrected[..., 1].min():.4f}, {da3_tensor_corrected[..., 1].max():.4f}], mean={da3_tensor_corrected[..., 1].mean():.4f}")
    print(f"  Z: [{da3_tensor_corrected[..., 2].min():.4f}, {da3_tensor_corrected[..., 2].max():.4f}], mean={da3_tensor_corrected[..., 2].mean():.4f}")
    
    # Step 2: 应用 camera_to_pytorch3d
    da3_pointmap_p3d = camera_convention_transform.transform_points(da3_tensor_corrected)
    
    print("\n" + "=" * 60)
    print("DA3 Pointmap (变换后，PyTorch3D 坐标系)")
    print("=" * 60)
    print(f"  X: [{da3_pointmap_p3d[..., 0].min():.4f}, {da3_pointmap_p3d[..., 0].max():.4f}], mean={da3_pointmap_p3d[..., 0].mean():.4f}")
    print(f"  Y: [{da3_pointmap_p3d[..., 1].min():.4f}, {da3_pointmap_p3d[..., 1].max():.4f}], mean={da3_pointmap_p3d[..., 1].mean():.4f}")
    print(f"  Z: [{da3_pointmap_p3d[..., 2].min():.4f}, {da3_pointmap_p3d[..., 2].max():.4f}], mean={da3_pointmap_p3d[..., 2].mean():.4f}")
    
    # 对比分析
    print("\n" + "=" * 60)
    print("对比分析")
    print("=" * 60)
    
    # 检查符号
    moge_x_sign = "+" if moge_pointmap_raw[..., 0].mean() >= 0 else "-"
    moge_y_sign = "+" if moge_pointmap_raw[..., 1].mean() >= 0 else "-"
    moge_z_sign = "+" if moge_pointmap_raw[..., 2].mean() >= 0 else "-"
    
    da3_x_sign = "+" if da3_pointmap[0].mean() >= 0 else "-"
    da3_y_sign = "+" if da3_pointmap[1].mean() >= 0 else "-"
    da3_z_sign = "+" if da3_pointmap[2].mean() >= 0 else "-"
    
    print(f"MoGe 原始符号: X={moge_x_sign}, Y={moge_y_sign}, Z={moge_z_sign}")
    print(f"DA3 原始符号:  X={da3_x_sign}, Y={da3_y_sign}, Z={da3_z_sign}")
    
    if moge_y_sign != da3_y_sign:
        print("⚠️  Y 轴符号不同！DA3 需要翻转 Y")
    if moge_z_sign != da3_z_sign:
        print("⚠️  Z 轴符号不同！DA3 需要翻转 Z")
    
    # 检查变换后
    moge_p3d_x_sign = "+" if moge_pointmap_p3d[..., 0].mean() >= 0 else "-"
    moge_p3d_y_sign = "+" if moge_pointmap_p3d[..., 1].mean() >= 0 else "-"
    moge_p3d_z_sign = "+" if moge_pointmap_p3d[..., 2].mean() >= 0 else "-"
    
    da3_p3d_x_sign = "+" if da3_pointmap_p3d[..., 0].mean() >= 0 else "-"
    da3_p3d_y_sign = "+" if da3_pointmap_p3d[..., 1].mean() >= 0 else "-"
    da3_p3d_z_sign = "+" if da3_pointmap_p3d[..., 2].mean() >= 0 else "-"
    
    print(f"\nMoGe PyTorch3D 符号: X={moge_p3d_x_sign}, Y={moge_p3d_y_sign}, Z={moge_p3d_z_sign}")
    print(f"DA3 PyTorch3D 符号:  X={da3_p3d_x_sign}, Y={da3_p3d_y_sign}, Z={da3_p3d_z_sign}")
    
    if moge_p3d_y_sign != da3_p3d_y_sign:
        print("⚠️  变换后 Y 轴符号不同！")
    if moge_p3d_z_sign != da3_p3d_z_sign:
        print("⚠️  变换后 Z 轴符号不同！")
    
    # 尺度对比
    moge_scale = moge_pointmap_p3d[..., 2].mean().item()
    da3_scale = da3_pointmap_p3d[..., 2].mean().item()
    print(f"\n平均深度 (Z mean):")
    print(f"  MoGe: {moge_scale:.4f}")
    print(f"  DA3:  {da3_scale:.4f}")
    print(f"  比值 (DA3/MoGe): {da3_scale/moge_scale:.4f}")


if __name__ == "__main__":
    main()

