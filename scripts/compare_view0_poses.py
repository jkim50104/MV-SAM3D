#!/usr/bin/env python3
"""
比较 View 0 的两个 pose 来源：
1. all_view_poses_decoded.json 中的 view 0
2. params.npz 中的最终 pose

如果这两个不同，说明解码过程有问题。
"""

import sys
from pathlib import Path
import numpy as np
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_view0_poses.py <result_dir>")
        sys.exit(1)
    
    result_dir = Path(sys.argv[1])
    
    # 加载 all_view_poses_decoded.json
    json_path = result_dir / "all_view_poses_decoded.json"
    view0_decoded = None
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        view0_decoded = data["views"][0]
        print("=" * 60)
        print("From all_view_poses_decoded.json (View 0):")
        print("=" * 60)
        t = np.array(view0_decoded['translation']).flatten()[:3]
        r = np.array(view0_decoded['rotation']).flatten()[:4]
        s = np.array(view0_decoded['scale']).flatten()[:3]
        ps = np.array(view0_decoded.get('pointmap_scale', [[0,0,0]])).flatten()[:3]
        print(f"  translation:    {t}")
        print(f"  rotation:       {r}")
        print(f"  scale:          {s}")
        print(f"  pointmap_scale: {ps}")
    else:
        print(f"all_view_poses_decoded.json not found")
    
    # 加载 params.npz
    npz_path = result_dir / "params.npz"
    params = None
    if npz_path.exists():
        params = dict(np.load(npz_path))
        print("\n" + "=" * 60)
        print("From params.npz (final output):")
        print("=" * 60)
        t2 = params.get('translation', np.zeros(3)).flatten()[:3]
        r2 = params.get('rotation', np.zeros(4)).flatten()[:4]
        s2 = params.get('scale', np.ones(3)).flatten()[:3]
        ps2 = params.get('pointmap_scale', np.zeros(3)).flatten()[:3]
        print(f"  translation:    {t2}")
        print(f"  rotation:       {r2}")
        print(f"  scale:          {s2}")
        print(f"  pointmap_scale: {ps2}")
    else:
        print(f"params.npz not found")
    
    # 比较差异
    if view0_decoded is not None and params is not None:
        print("\n" + "=" * 60)
        print("Comparison (View 0 decoded vs Final output):")
        print("=" * 60)
        
        t1 = np.array(view0_decoded['translation']).flatten()[:3]
        t2 = params.get('translation', np.zeros(3)).flatten()[:3]
        print(f"  translation diff: {np.abs(t1 - t2)}")
        print(f"  translation same: {np.allclose(t1, t2, atol=1e-5)}")
        
        r1 = np.array(view0_decoded['rotation']).flatten()[:4]
        r2 = params.get('rotation', np.zeros(4)).flatten()[:4]
        print(f"  rotation diff:    {np.abs(r1 - r2)}")
        print(f"  rotation same:    {np.allclose(r1, r2, atol=1e-5)}")
        
        s1 = np.array(view0_decoded['scale']).flatten()[:3]
        s2 = params.get('scale', np.ones(3)).flatten()[:3]
        print(f"  scale diff:       {np.abs(s1 - s2)}")
        print(f"  scale same:       {np.allclose(s1, s2, atol=1e-5)}")
        
        ps1 = np.array(view0_decoded.get('pointmap_scale', [[0,0,0]])).flatten()[:3]
        ps2 = params.get('pointmap_scale', np.zeros(3)).flatten()[:3]
        print(f"  pointmap_scale diff: {np.abs(ps1 - ps2)}")
        print(f"  pointmap_scale same: {np.allclose(ps1, ps2, atol=1e-5)}")
        
        print("\n" + "=" * 60)
        print("Conclusion:")
        print("=" * 60)
        if np.allclose(t1, t2, atol=1e-5) and np.allclose(r1, r2, atol=1e-5) and np.allclose(s1, s2, atol=1e-5):
            print("✓ View 0 decoded pose matches final output pose!")
            print("  The multiview visualization should use the SAME pose as merge_scene.")
        else:
            print("✗ View 0 decoded pose DIFFERS from final output pose!")
            print("  This is the root cause of the alignment issue.")
            print("\n  Possible reasons:")
            print("  1. Different pointmap_scale/shift used for decoding")
            print("  2. all_view_poses_raw doesn't match ss_return_dict")


if __name__ == "__main__":
    main()

