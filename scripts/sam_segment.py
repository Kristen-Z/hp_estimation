import sys
sys.path.append(".")
sys.path.append("./lang-segment-anything")
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from lang_sam import LangSAM
from src.utils.parser import add_common_args
from src.utils.seg_utils import sam_pred
from argparse import ArgumentParser
import ipdb

def main():
    parser = ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--text", required=True, type=str)
    parser.add_argument("--out_path", required=True, type=str)
    parser.add_argument("--chosen_frames", type=str)
    parser.add_argument("--overwrite", action="store_true")

    args =  parser.parse_args()
    
    
    base_path = os.path.join(args.root_dir, args.seq_path)
    if 'session' in args.seq_path:
        args.seq_path = args.seq_path.split('_session_')[-1]
        if 'session' in args.text:
            args.text = args.text.split('_session_')[-1]
    
    model = LangSAM()
    
    usable_frames_path = os.path.join(base_path, "pose_dumps", "usable_frames.json")
    if os.path.exists(usable_frames_path):
        with open(usable_frames_path, 'r') as f:
            usable_frames = json.load(f)
    else:
        usable_frames = []
            
    frames = {}
    all_views = os.listdir(os.path.join(base_path, "images", "image"))
    
    seg_dir = os.path.join(base_path, "images", "segmented_sam")
            
    for view in all_views:
        frames[view] = {}
        for frame in os.listdir(os.path.join(base_path, "images", "image", view)):
            
            frame_no = int(frame.split('.')[0])
            if len(usable_frames) != 0:
                if frame_no not in usable_frames:
                    continue
            
            if not args.overwrite:
                if os.path.exists(seg_dir):
                    seg_path = os.path.join(seg_dir, view, str(frame.split('.')[0]) + ".png") 
                    if os.path.exists(seg_path):
                        print("Segmentation exists at ", seg_path, " Skipping...")
                        continue
                
            image = cv2.imread(os.path.join(base_path, "images", "image", view, frame))
            frame_name = frame.split('.')[0]
            frames[view][frame_name] = image
    
    sam_pred(model, frames, args.text, os.path.join(args.out_path, args.seq_path))

if __name__ == "__main__":
    main()
