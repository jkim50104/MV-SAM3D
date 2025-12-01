#!/usr/bin/env python3
"""
调试：比较 all_view_poses_raw 中的 View 0 和 ss_return_dict 中的 pose
"""

import sys
from pathlib import Path
import numpy as np
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_pose_raw.py <result_dir>")
        sys.exit(1)
    
    result_dir = Path(sys.argv[1])
    
    # 检查是否有 all_view_poses.npz（保存原始 pose 的文件）
    npz_path = result_dir / "all_view_poses.npz"
    if npz_path.exists():
        print("=" * 60)
        print("From all_view_poses.npz (raw poses):")
        print("=" * 60)
        data = dict(np.load(npz_path, allow_pickle=True))
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}")
                if len(value.shape) >= 1 and value.shape[0] >= 1:
                    print(f"    View 0: {value[0].flatten()[:6]}")
    else:
        print(f"all_view_poses.npz not found")
    
    # 检查 all_view_poses_decoded.json
    json_path = result_dir / "all_view_poses_decoded.json"
    if json_path.exists():
        print("\n" + "=" * 60)
        print("From all_view_poses_decoded.json:")
        print("=" * 60)
        with open(json_path) as f:
            data = json.load(f)
        
        if "views" in data and len(data["views"]) > 0:
            view0 = data["views"][0]
            print(f"  View 0 translation: {np.array(view0['translation']).flatten()[:3]}")
            print(f"  View 0 rotation:    {np.array(view0['rotation']).flatten()[:4]}")
            print(f"  View 0 scale:       {np.array(view0['scale']).flatten()[:3]}")
    
    # 检查 params.npz
    params_path = result_dir / "params.npz"
    if params_path.exists():
        print("\n" + "=" * 60)
        print("From params.npz (final output):")
        print("=" * 60)
        params = dict(np.load(params_path))
        print(f"  translation: {params.get('translation', 'N/A')}")
        print(f"  rotation:    {params.get('rotation', 'N/A')}")
        print(f"  scale:       {params.get('scale', 'N/A')}")


if __name__ == "__main__":
    main()

