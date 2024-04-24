#!/usr/bin/env bash

ROOT=$1
WHICH=$2
TEXT_PROMPT=$3
SCENE=$4
INGP_PATH=$5

PARAMS_PATH="${ROOT}calib/params.txt" 

BASE_PATH=$ROOT/$WHICH 

export TOKENIZERS_PARALLELISM=true
python scripts/sam_segment.py --root_dir $BASE_PATH --out_path $BASE_PATH --seq_path $SCENE --text "${TEXT_PROMPT}"

python scripts/object_segment2.py \
    --base_path $BASE_PATH/$SCENE \
    --train_dir_name 'segmented_sam' \
    --save_dir_name 'camera_check' \
    --network "${INGP_PATH}/configs/nerf/base.json" \
    --params_path $PARAMS_PATH \
    --batch_size 32512 \
    --n_steps 15000 \
    --marching_cubes_res 256 \
    --camera_scale 1.0 \
    --align_bounding_box \
    --save_segmented_images \
    --overwrite_segmentation \
    --downscale_factor 0.45 \
    --optimize_extrinsics #--gui

exit

# python scripts/sam.py \
#     --base_dir $BASE_PATH/$SCENE \
#     --input_dir_name 'segmented_ngp' \
#     --save_dir_name 'refined_sam' \
#     --ckpt './data/ckpts/sam_vit_h_4b8939.pth'

# python scripts/object_segment2.py \
#     --base_path $BASE_PATH/$SCENE \
#     --train_dir_name 'segmented_ngp' \
#     --save_dir_name 'camera_check' \
#     --network "${INGP_PATH}/configs/nerf/base.json" \
#     --params_path $PARAMS_PATH \
#     --batch_size 32512 \
#     --n_steps 20000 \
#     --marching_cubes_res 256 \
#     --camera_scale 1.0 \
#     --align_bounding_box \
#     --save_segmented_images \
#     --downscale_factor 0.45 \
#     --optimize_extrinsics
