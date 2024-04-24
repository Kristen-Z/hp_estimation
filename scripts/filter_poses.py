"""
Filter out bad poses based upon some simple heuristics
"""
import sys
sys.path.append(".")

import numpy as np
import json
import os
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", "-r", required=True, type=str)
parser.add_argument("--seq_path", "-s", required=True, type=str)
parser.add_argument("--bin_size", type=int, default=5)
parser.add_argument("--ignore_missing_tip", action="store_true", help="Should a missing fingertip be allowed")

args = parser.parse_args()

base_path = os.path.join(args.root_dir, args.seq_path)
keypoints3d_dir = os.path.join(base_path, "keypoints_3d")
kyps_files = list(sorted(glob(f"{keypoints3d_dir}/*.json")))

chosen_frames = []

# Indices of fingers
# If none of the keypoints are present for any finger, skip the frame
finger_idx = [list(range(2, 5))] + [ list(range(i, i+4)) for i in range(5, 18, 4) ]

# Indices of finger tips
# If any of them are missing, skip the frame
tip_idx = [4, 8, 12, 16, 20]
print('bin_size',args.bin_size)

for i in range(0, len(kyps_files), args.bin_size):
    # Read back the files
    kyps_3d = []
    start_frame_id = int(os.path.basename(kyps_files[i]).split(".")[0])
    for j in range(args.bin_size):
        if i+j >= len(kyps_files):
            break
        with open(kyps_files[i+j], "r") as f:
            kyps_3d.append(np.asarray(json.load(f)))
            print('loading files',i+j,kyps_3d[-1].shape)
    kyps_3d = np.stack(kyps_3d)
   
    # Remove frames which have complete finger missing
    to_use = np.ones(kyps_3d.shape[0], dtype=bool)
    for idx in finger_idx:
        to_use = np.logical_and(to_use, np.any(kyps_3d[:,idx,3], axis=1))
    
    # Remove frames which have any of the finger tips missing
    if not args.ignore_missing_tip:
        to_use = np.logical_and(to_use, np.all(kyps_3d[:,tip_idx,3], axis=1))
        if not np.any(to_use):
            continue

    # Find frame with maximum number of detected keypoints
    unfound_count = kyps_3d.shape[1] * np.ones(kyps_3d.shape[0])
    unfound_count[to_use] = np.count_nonzero(np.isclose(kyps_3d[to_use,:,3], 0), axis=1)
    chosen_frame = kyps_files[i+np.argmin(unfound_count)]
    chosen_frame_id = int(os.path.basename(chosen_frame).split(".")[0])
    # print('chosen frames',chosen_frame_id)
    chosen_frames.append(chosen_frame_id)

chosen_path = os.path.join(base_path, "chosen_frames.json")
with open(chosen_path, "w") as f:
    json.dump(chosen_frames, f, indent=2)
