#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget -P checkpoints"
elif command -v curl &> /dev/null; then
    CMD="curl -L -o"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Create checkpoints directory
mkdir -p checkpoints

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_t_url="${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
sam2p1_hiera_b_plus_url="${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

# SAM 2.1 checkpoints
echo "Downloading sam2.1_hiera_tiny.pt checkpoint..."
wget -P checkpoints $sam2p1_hiera_t_url || { echo "Failed to download $sam2p1_hiera_t_url"; exit 1; }

echo "Downloading sam2.1_hiera_small.pt checkpoint..."
wget -P checkpoints $sam2p1_hiera_s_url || { echo "Failed to download $sam2p1_hiera_s_url"; exit 1; }

echo "Downloading sam2.1_hiera_base_plus.pt checkpoint..."
wget -P checkpoints $sam2p1_hiera_b_plus_url || { echo "Failed to download $sam2p1_hiera_b_plus_url"; exit 1; }

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
wget -P checkpoints $sam2p1_hiera_l_url || { echo "Failed to download $sam2p1_hiera_l_url"; exit 1; }

echo "All checkpoints downloaded successfully to ./checkpoints/"
