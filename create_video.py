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
from triangle_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from triangle_renderer import TriangleModel
from utils.render_utils import generate_path
import cv2

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_as", default="output_video", type=str)
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


    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    traj_dir = os.path.join(args.model_path, 'traj')
    os.makedirs(traj_dir, exist_ok=True)

    render_path = os.path.join(traj_dir, "renders")
    os.makedirs(render_path, exist_ok=True)
    
    n_frames = 240*5
    cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_frames)
    
    with torch.no_grad():
        for idx, view in enumerate(tqdm(cam_traj, desc="Rendering progress")):
            rendering = render(view, triangles, pipe, background)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(traj_dir, "renders", '{0:05d}'.format(idx) + ".png"))

    image_folder = os.path.join(traj_dir, "renders")
    output_video = args.save_as + '.mp4'

    # Get all image files sorted by name
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # Create video writer (FPS = 30)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

    # Write each image to the video
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        video.write(img)

    video.release()

    print(f'Video saved as {output_video}')
