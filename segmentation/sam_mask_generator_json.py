import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json

from sam2.build_sam import build_sam2_video_predictor


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the directory containing input images.")
    parser.add_argument("--save_path", type=str, help="Path to the directory to save output masks.")
    parser.add_argument("--tmp_dir", type=str, help="Path to the temporary directory for intermediate files.", default="tmp_sam/")
    parser.add_argument("--resolution", "-r", type=float, default=1., help="Resolution for resizing images.")
    parser.add_argument("--json_path", type=str, help="Path to the json file containing points and labels.")

    args = parser.parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    image_names = os.listdir(args.data_path)
    sorted_image_names = sorted(image_names, key=lambda x: os.path.basename(x).split(".")[0])

    # Create tmp dir with simlink
    newnames = []
    for i, fname in enumerate(sorted_image_names, 1):
        newname = f"{i:04d}.JPG"
        newnames.append(newname)
        src_path = os.path.abspath(os.path.join(args.data_path, fname))
        dst_path = os.path.join(args.tmp_dir, newname)
        # Remove existing symlink if present
        if os.path.islink(dst_path) or os.path.exists(dst_path):
            os.remove(dst_path)
        os.symlink(src_path, dst_path)

    checkpoint = "sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    inference_state = predictor.init_state(video_path=args.tmp_dir)

    ann_frame_idx = 0


    # Load points and labels from JSON
    with open(args.json_path, 'r') as f:
        prompts_data = json.load(f)

    prompts = {}
    all_points = []
    all_labels = []
    all_obj_ids = []

    for i, (obj_name, data) in enumerate(prompts_data.items()):
        obj_id = i + 1
        points = np.array(data['points'], dtype=np.float32) / args.resolution
        labels = np.array(data['labels'], dtype=np.int32)
        
        boxes = []
        if 'boxes' in data and data['boxes']:
            boxes = [np.array(b, dtype=np.float32) / args.resolution for b in data['boxes']]
        
        prompts[obj_id] = (points, labels, boxes)
        all_points.append(points)
        all_labels.append(labels)
        all_obj_ids.append(obj_id)

        # Add points and first box
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=boxes[0] if boxes else None
        )

        # Add subsequent boxes
        if len(boxes) > 1:
            for box in boxes[1:]:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    box=box
                )

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):

        for i, obj_id in enumerate(out_obj_ids):
            mask_np = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()  # shape (H, W)
            mask_u8 = (mask_np.astype(np.uint8) * 255)
            Image.fromarray(mask_u8).save(os.path.join(args.save_path, f"frame_{out_frame_idx:04d}_obj{obj_id}_mask.png"))

    shutil.rmtree(args.tmp_dir)