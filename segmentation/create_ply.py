#!/usr/bin/env python3
# triangles_to_ply.py
import argparse, os, sys
import numpy as np
import torch
import trimesh



def parse_vec3(s):
    try:
        x, y, z = [float(v) for v in s.split(",")]
        return torch.tensor([x, y, z], dtype=torch.float32)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid vec3 '{s}', expected 'x,y,z'") from e

def load_scene(scene_dir, device="cpu"):
    path = os.path.join(scene_dir, "point_cloud/iteration_30000", "point_cloud_state_dict.pt")
    if not os.path.isfile(path): raise FileNotFoundError(f"Could not find '{path}'")
    sd = torch.load(path, map_location=device)
    verts   = sd["triangles_points"].to(device).to(torch.float32)        # [V,3]
    faces   = sd["_triangle_indices"].to(device).to(torch.int64)         # [T,3]
    f_dc    = sd["features_dc"].to(device).to(torch.float32)             # [V,1,3]
    f_rest  = sd["features_rest"].to(device).to(torch.float32)           # [V,?,3]
    act_deg = int(sd.get("active_sh_degree", 3))

    triangle_hits = torch.load(os.path.join(scene_dir, 'segmentation/triangle_hits_mask.pt'),   map_location=device)
    triangle_hits_total = torch.load(os.path.join(scene_dir, 'segmentation/triangle_hits_total.pt'), map_location=device)

    min_hits, ratio_threshold = 1, 0.9
    triangle_ratio = torch.zeros_like(triangle_hits, dtype=torch.float32)
    valid_mask = triangle_hits_total > 0
    triangle_ratio[valid_mask] = triangle_hits[valid_mask].float() / triangle_hits_total[valid_mask].float()
    keep_mask = (triangle_ratio >= ratio_threshold) & (triangle_hits >= min_hits)

    faces = faces[keep_mask]                       # keep only selected triangles
    if faces.numel() == 0:                         # nothing left, return empties that are consistent
        empty_faces = faces.new_zeros((0, 3), dtype=torch.long)
        empty_verts = verts.new_zeros((0, 3))
        empty_dc    = f_dc.new_zeros((0,) + f_dc.shape[1:])
        empty_rest  = f_rest.new_zeros((0,) + f_rest.shape[1:]) if f_rest.numel() > 0 else f_rest
        return empty_verts, empty_faces, empty_dc, empty_rest, act_deg

    # ---- prune vertices and remap face indices ----
    used, _ = torch.sort(torch.unique(faces.reshape(-1)))                 # [K]
    remap = torch.full((verts.shape[0],), -1, dtype=torch.long, device=device)
    remap[used] = torch.arange(used.numel(), device=device)
    faces = remap[faces]                                                  # reindex faces
    verts = verts[used]                                                   # keep only used verts
    f_dc  = f_dc[used]
    f_rest = f_rest[used] if f_rest.numel() > 0 else f_rest

    return verts, faces, f_dc, f_rest, act_deg

def export_ply(out_path, verts_np, faces_np, colors_rgb, zup_to_yup=False):
    if zup_to_yup:
        # Create a copy to avoid modifying the original data
        transformed_verts = verts_np.copy()
        
        # Apply transformation: (x, y, z) -> (x, z, -y)
        # This should convert from Z-up to Y-up
        transformed_verts[:, 1] = verts_np[:, 2]   # y = z
        transformed_verts[:, 2] = -verts_np[:, 1]  # z = -y
        
        verts_np = transformed_verts

    # Create mesh with vertex colors
    mesh = trimesh.Trimesh(vertices=verts_np.astype(np.float32),
                           faces=faces_np.astype(np.int32),
                           vertex_colors=colors_rgb.astype(np.uint8),
                           process=False)
    
    # Export as PLY
    mesh.export(out_path, file_type='ply')

def main():
    p = argparse.ArgumentParser(description="Export triangle scene to PLY with per-vertex colors.")
    p.add_argument("scene_dir", type=str, help="Directory containing point_cloud_state_dict.pt")
    p.add_argument("--out", type=str, default="mesh.ply", help="Output path, default <scene_dir>/mesh.ply")
    p.add_argument("--camera", type=parse_vec3, default="0,0,0", help="Camera center x,y,z")
    p.add_argument("--degree", type=int, default=None, help="Override SH degree to evaluate")
    p.add_argument("--cpu", action="store_true", help="Force CPU for loading and SH eval")
    p.add_argument("--zup", action="store_true", help="Input is Z-up, rotate to Y-up for standard viewers")
    args = p.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    verts, faces, f_dc, f_rest, act_deg = load_scene(args.scene_dir, device=device)
    camera_center = args.camera.to(device)

    # Compute colors (same as original 3D Gaussian Splatting training)
    SH_C0 = 0.28209479177387814
    colors = SH_C0 * f_dc + 0.5
    colors = torch.clamp(colors, 0.0, 1.0)
    colors_u8 = (colors * 255.0).round().to(torch.uint8).cpu().numpy()
    colors_u8 = colors_u8.squeeze()  # Remove the middle dimension [V, 1, 3] -> [V, 3]

    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    out_path = args.out or os.path.join(args.scene_dir, "mesh.ply")
    export_ply(out_path, verts_np, faces_np, colors_u8, zup_to_yup=args.zup)
    print(f"Saved PLY to: {out_path}")
    print(f"Vertices: {verts_np.shape[0]}, Faces: {faces_np.shape[0]}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)