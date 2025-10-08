#
# Copyright (C) 2024, Inria, University of Liege, KAUST and University of Oxford
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  jan.held@uliege.be
#


import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from triangle_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from triangle_renderer import TriangleModel
import numpy as np
from utils.render_utils import generate_path, create_videos
import cv2
from PIL import Image
import torch.nn.functional as F


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_as", default="output_video", type=str)
    parser.add_argument("--path_mask", type=str, help="Path to the directory containing input masks.")
    parser.add_argument("--object_id", type=int, default=1, help="ID of the object to segment.")
    args = get_combined_args(parser)
    print("Creating video for " + args.model_path)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    triangles = TriangleModel(dataset.sh_degree)

    triangles.upscaling_factor = 4

    scene = Scene(args=dataset,
                  triangles=triangles,
                  init_opacity=None,
                  set_sigma=None,
                  load_iteration=args.iteration,
                  shuffle=False)

    # Initialize counters for total renders and mask hits per triangle
    n_triangles = triangles.get_triangle_indices.shape[0]
    triangle_total_renders = torch.zeros(n_triangles, device='cuda', dtype=torch.long)
    triangle_mask_hits = torch.zeros(n_triangles, device='cuda', dtype=torch.long)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()

    path_mask = args.path_mask
    
    with torch.no_grad():
        depth_tolerance = 4.5  # Meters

        cou = 0
        while viewpoint_stack:
            viewpoint_cam = viewpoint_stack.pop(0)
            render_pkg = render(viewpoint_cam, triangles, pipe, background)

            # Get rendering outputs
            ids_hr = render_pkg["rend_ids"]  # Shape: [1, H_high, W_high]
            depth = render_pkg["depth_full"]  # Shape: [1, H_high, W_high]

            # Load and upsample mask
            mask_filename = f"frame_{cou:04d}_obj{args.object_id}_mask.png"
            mask_path = os.path.join(path_mask, mask_filename)
            mask = np.array(Image.open(mask_path).convert('L'))
            mask_lr = torch.from_numpy(mask).cuda().bool()
            
            mask_hr = F.interpolate(
                mask_lr.unsqueeze(0).unsqueeze(0).float(),
                size=(ids_hr.shape[1], ids_hr.shape[2]),
                mode='nearest'
            ).squeeze(0).squeeze(0).bool()

            # Prepare depth and ID maps
            depth_squeezed = depth.squeeze(0)  # [H_high, W_high]
            ids_squeezed = ids_hr.squeeze(0)   # [H_high, W_high]

            # Calculate reference depth (median of masked region)
            masked_depth = depth_squeezed[mask_hr]
            if len(masked_depth) > 0:
                reference_depth = torch.median(masked_depth)
                
                # Create depth-based mask (pixels within tolerance of reference depth)
                depth_mask = torch.abs(depth_squeezed - reference_depth) < depth_tolerance
                
                # Create combined validity mask:
                combined_mask = mask_hr & depth_mask & (ids_squeezed != -1)

                # Count total renders and mask hits for all triangles
                valid_ids = ids_squeezed[combined_mask]
                all_ids = ids_squeezed[ids_squeezed != -1]  # All rendered triangles

                # Update counters
                if len(valid_ids) > 0:
                    unique_ids_mask, counts_mask = torch.unique(valid_ids, return_counts=True)
                    valid_mask = (unique_ids_mask >= 0) & (unique_ids_mask < n_triangles)
                    triangle_mask_hits.index_add_(0, unique_ids_mask[valid_mask].long(), counts_mask[valid_mask])
                
                if len(all_ids) > 0:
                    unique_ids_all, counts_all = torch.unique(all_ids, return_counts=True)
                    valid_all = (unique_ids_all >= 0) & (unique_ids_all < n_triangles)
                    triangle_total_renders.index_add_(0, unique_ids_all[valid_all].long(), counts_all[valid_all])
            else:
                print(f"Warning: Empty mask for image {cou:04d}")
            
            cou += 1
        
        path_save = os.path.join(args.model_path, 'segmentation')
        os.makedirs(path_save, exist_ok=True)

        torch.save(triangle_mask_hits, os.path.join(path_save, 'triangle_hits_mask.pt'))
        torch.save(triangle_total_renders, os.path.join(path_save, 'triangle_hits_total.pt')) 

    