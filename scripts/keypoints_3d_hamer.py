import os
import sys
import cv2
import json
import torch
import shutil
import argparse
import numpy as np
# import xml.etree.cElementTree as ET

from tqdm import tqdm
from glob import glob
from natsort import natsorted
from pathlib import Path

sys.path.append(".")
from src.triangulate import triangulate_joints, simple_processor, ransac_processor
from src.utils.reader import Reader
import src.utils.params as param_utils
from src.utils.parser import add_common_args
# from src.utils.triangulate import traingulate

# sys.path.append("./AlphaPose_mp")
sys.path.append("./hamer")
from keypoints_2d_hamer import keypoints_2d
from vitpose_model import ViTPoseModel
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy


# -------------------- Arguments -------------------- #
parser = argparse.ArgumentParser(description='AlphaPose Keypoints Parser')
parser.add_argument("--separate_calib", action="store_true")
parser.add_argument("--allow_missing_finger", action="store_true")
parser.add_argument("--bin_size", type=int, default=5)
parser.add_argument("--use_optim_params", action="store_true")
parser.add_argument("--all_frames", default=False, action="store_true")
parser.add_argument("--easymocap", default=False, action="store_true", help='use Easymocap for triangulation')

add_common_args(parser)
args = parser.parse_args()

base_path = os.path.join(args.root_dir, args.seq_path)
image_base = os.path.join(base_path, "synced")

# Loads the camera parameters
if args.use_optim_params:
    params_txt = "optim_params.txt"
else:
    params_txt = "params.txt"

if args.separate_calib:
    params_path = os.path.join(base_path, "calib", params_txt)
else:
    params_path = os.path.join(args.root_dir, "calib", params_txt)

params = param_utils.read_params(params_path)
cam_names = params[:]["cam_name"]

# Gets the projection matrices and distortion parameters
projs = []
intrs = []
dist_intrs = []
dists = []
rot = []
trans = []
for i in range(len(params)):
    extr = param_utils.get_extr(params[i])
    intr, dist = param_utils.get_intr(params[i])
    r, t = param_utils.get_rot_trans(params[i])

    rot.append(r)
    trans.append(t)

    intrs.append(intr.copy())
    if args.undistort:
        dist_intrs.append(intr.copy())

    projs.append(intr @ extr)
    dists.append(dist)


# Get files to process
reader = Reader(args.input_type, image_base)

keypoints3d_dir = os.path.join(base_path, "keypoints_3d")
os.makedirs(keypoints3d_dir, exist_ok=True)

keypoints2d_dir = os.path.join(base_path, "keypoints_2d")
os.makedirs(keypoints2d_dir, exist_ok=True)

bbox_dir = os.path.join(base_path, "bboxes")
os.makedirs(bbox_dir, exist_ok=True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Load Hamer detector
from detectron2.config import LazyConfig
import hamer
cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)
# keypoint detector
cpm = ViTPoseModel(device)

print("Total frames",reader.frame_count)
keypoints3d = {}

if (args.all_frames):
    chosen_frames = range(0, reader.frame_count, 1)
else:
    chosen_frames = range(args.start, args.end, args.stride)

video_paths = natsorted(glob(f"{image_base}/*/*.avi"))
for idx, vid in enumerate(video_paths):
    # AlphaPose 2D keypoints
    outdir = os.path.join(keypoints2d_dir, os.path.basename(vid)[:-4])
    keypoints_2d(vid, outdir, cpm, detector)

    print("running keypoints 2d for", vid, '\nsave kpt at', outdir) 

print('finish running 2d detection.')


for frame in chosen_frames:
    frame = f"{frame:08d}"
    keypoints2d = []
    for cam in cam_names:
        ap_keypoints_path = os.path.join(keypoints2d_dir, cam, f"{frame}.json")
        with open(ap_keypoints_path, "r") as f:
            data = json.load(f)
            # cam_keypoints = np.array(data).reshape(-1, 3).tolist()
            keypoints2d.append(data)


    if not args.easymocap:
        keypoints3d, residuals = triangulate_joints(np.asarray(keypoints2d), np.asarray(projs), processor=ransac_processor, residual_threshold=10, min_samples=5)
        print(f"Error: {residuals.mean()}")
    else:
        sys.path.append("./EasyMocap")
        from myeasymocap.operations.triangulate import SimpleTriangulate
        # Easy Mocap for 3D keypoints
        cameras = { 'K': np.asarray(intrs),
            'R': np.asarray(rot), 
            'T': np.asarray(trans),
            'dist': np.asarray(dists),
            'P': np.asarray(projs) }
        triangulation = SimpleTriangulate("iterative")
        keypoints3d = triangulation(np.asarray(keypoints2d), cameras)['keypoints3d']
        
    keypt_file = os.path.join(keypoints3d_dir, f"{frame}.json")
    print(f"Writing 3D keypoints to {keypt_file}")
    with open(keypt_file, "w") as f:
        json.dump(keypoints3d.tolist(), f)