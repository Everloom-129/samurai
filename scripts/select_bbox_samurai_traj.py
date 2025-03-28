import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import tkinter as tk
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor
from select_bbox import get_bbox_coordinates

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def main(args):
    # Get bounding box from UI
    bbox_coords = get_bbox_coordinates(args.video_path)
    if not bbox_coords:
        print("No bounding box selected. Exiting...")
        return
        
    x1, y1, x2, y2 = bbox_coords
    bbox = (x1, y1, x2, y2)
    
    # Initialize model and tracking
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    
    video_basename = os.path.basename(args.video_path)
    video_basename = os.path.splitext(video_basename)[0]
    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"traj_samurai_{video_basename}.txt")
    video_output_path = os.path.join(args.result_dir, f"traj_samurai_{video_basename}.mp4")
    
    # Setup video writer if needed
    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) 
                           if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, frame_rate, (width, height))
    
    # Run tracking
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
        
        with open(result_path, 'w') as result_file:
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0
                    non_zero_indices = np.argwhere(mask)
                    
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    
                    # Save bbox in MOT format
                    result_file.write(f"{frame_idx+1},{obj_id+1},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},1,-1,-1,-1\n")
                    
                    if args.save_to_video:
                        img = loaded_frames[frame_idx].copy()
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask] = (255, 0, 0)
                        img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)
                        cv2.rectangle(img, (bbox[0], bbox[1]), 
                                    (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                                    (255, 0, 0), 2)
                        out.write(img)
    
    if args.save_to_video:
        out.release()
    
    # Cleanup
    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", 
                       help="Path to the model checkpoint.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    parser.add_argument("--result_dir", default="traj_results", help="Directory to save tracking results.")
    args = parser.parse_args()
    main(args)
