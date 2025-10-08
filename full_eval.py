#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
from argparse import ArgumentParser

# mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
# mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
mipnerf360_outdoor_scenes = ["garden",]
mipnerf360_indoor_scenes = []


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1 "
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        print("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args)
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        common_args += " --indoor "
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        # os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)