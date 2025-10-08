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
from triangle_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from triangle_renderer import TriangleModel


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_as", default="output_video", type=str)
    parser.add_argument("--ratio_threshold", default=0.75, type=float) # higher for only triangles that are mostly part of the object
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
                  shuffle=False, 
                  segment=True,
                  ratio_threshold=args.ratio_threshold
                  )


    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()

    traj_dir = os.path.join(args.model_path, 'extracted_object')
    os.makedirs(traj_dir, exist_ok=True)
    
    with torch.no_grad():
        cou = 0
        while viewpoint_stack:
            viewpoint_cam = viewpoint_stack.pop(0)
            rendering = render(viewpoint_cam, triangles, pipe, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(traj_dir, f"{cou:04d}.jpg"))
            cou += 1

    