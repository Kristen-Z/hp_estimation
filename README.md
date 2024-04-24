# Hand Pose Estimation

This pipeline is designed to take input videos or images from BRICS system, performing 3D hand keypoint estimation and providing visualization results.

For 2D hand keypoints detection, we utilize `detectron2` to establish an approximate bounding box and subsequently employ `ViTpose` to detect keypoints within this region. 

For 3D hand keypoints estimation, we employ triangulation techniques to derive 3D points from multi-view 2D keypoints. Our scripts supports both `EasyMocap` triangulation and RANSAC-based methods.

The scripts for each stage of the preprocessing pipeline are present in the `scripts` directory. The pipeline works in the following way:
```
Calibration > Segmentation & Optimize camera params > 3D keypoints detection > Easymocap MANO Fitting
```

## Installation

To run the whole pipeline, we actually need 3 kinds of environments right now, please switch to different conda environments for different steps.

First of all, clone this repo recursively:

```shell
git clone https://github.com/Kristen-Z/hp_estimation.git --recursive
cd hp_estimation
```

+ For Calibration: Install [COLMAP](https://colmap.github.io/) on ccv.

+ For Segmentation & Optimize camera params: Install lang-SAM and instant-NGP

  ```shell
  # For lang-SAM
  pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  mkdir ckpts
  mv <checkpoint path> ckpts
  
  # For instant-NGP ccv version
  git clone --recursive https://github.com/nvlabs/instant-ngp
  cd instant-ngp
  module cmake/3.24.1 glew/2.1 gcc/7.2 
  module load openssl/3.0.0 
  module load libarchive/3.6.1 
  module load curl/7.86.0
  module load ffmpeg/4.0.1
  ```

  - Add following lines to the `CMakeLists.txt` file in the Instant-NGP root directory. After the line `set(NGP_VERSION "${CMAKE_PROJECT_VERSION}")`.

  ```
  set(EIGEN_DIR "dependencies/eigen/")
  set(EIGEN3_INCLUDE_DIR "dependencies/eigen/Eigen")
  list(APPEND CMAKE_PREFIX_PATH "/gpfs/runtime/opt/glew/2.1.0/")
  find_package(GLEW REQUIRED)
  set(GLEW_INCLUDE_DIRS "/gpfs/runtime/opt/glew/2.1.0/include/")
  set(GLEW_LIBRARIES "/gpfs/runtime/opt/glew/2.1.0/lib64/libGLEW.so")
  #find_package(GLEW REQUIRED)
  list(APPEND NGP_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})
  list(APPEND NGP_LIBRARIES GL ${GLEW_LIBRARIES} $<TARGET_OBJECTS:glfw_objects>)
  ```

  - Then you can follow normal build instructions from the Instant-NGP repository.

+ For hand pose estimation: Install hamer & EasyMocap

  ```shell
  cd hamer
  conda create --name hp_estim python=3.10
  conda activate hp_estim
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install -e .[all]
  pip install -v -e third-party/ViTPose
  bash fetch_demo_data.sh
  ```

  Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de/) and register to get access to the downloads section. We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder. Also copy the weights to the EasyMocap directory:
  
  ```shell
  hp_estimation$ mkdir EasyMocap/data/smplx/smplh
  hp_estimation$ cp hamer/_DATA/data/mano/MANO_RIGHT.pkl EasyMocap/data/smplx/smplh
  ```
  
  Also change a few lines in `EasyMocap/easymocap/smplmodel/body_model.py:177-179` to:
  
  ```python
          # self.num_pca_comps = kwargs['num_pca_comps']
          # self.use_pca = kwargs['use_pca']
          # self.use_flat_mean = kwargs['use_flat_mean']
          self.num_pca_comps = 6            
          self.use_pca = True            
          self.use_flat_mean = True
  ```

## Calibration

- Activate the `colmap` environment
- Rename the calibration directory as `calib` . The directory structure of the `<root_dir>` as follows.

```
root_dir
|   calib
|   |   *cam*
|   |   |   frame_*.jpg
|   sequence_name
```

- For obtaining the camera parameters run the following command. The generated `params.txt` file are under `calib` directory.

```shell
python src/colmap_calib.py -r <root_dir>
```



## Segmentation & Optimization

+ Activate the `instant-ngp` & `sam` environment
+ For obtaining the optimized camera parameters run the following command. The generated `optim_params.txt` file are under `calib` directory. `$INGP_PATH` refer to the path where you install the instant-ngp.

```shell
# Step 0: Exatrct Frames
python scripts/extract_frames.py -r $ROOT_DIR -s $SEQUENCE --out_path $ROOT_DIR 

# Step 1: OPTIMIZE EXTRINICS
bash optimize_extrinsics.sh $ROOT_DIR $ROOT_DIR "arm and hand" $SEQUENCE $INGP_PATH
```



## Hand Pose Estimation

After calibration and the target sequence are in `ROOT_DIR/SEQUENCE` directory.

```shell
ROOT_DIR="../data/"
SEQUENCE="use_mouse"
STRIDE=2
# STEP 2: Keypoints Extraction
python scripts/keypoints_3d_hamer.py -r $ROOT_DIR -s $SEQUENCE --undistort --start 0 --end -1 --stride $STRIDE --use_optim_params
python scripts/filter_poses.py -r $ROOT_DIR -s $SEQUENCE --bin_size 1

# STEP 3: MANO FIT
python scripts/mano_em.py -r $ROOT_DIR -s $SEQUENCE --model manor --body handr --vis_smpl --undistort --use_filtered --use_optim_param
```

Data Structure after processing:

```
root_directory
    |_calib (if all sequences use same calibration)
    |_sequence
    	|_ synced
        |_ images
            |_ segment_sam (from step 1)
            |_ camera_check (from step 1)
            |_ image (from step 0)
        |_ mano (from step 3) # rendered images after easymocap mano fitting
        |_ mesh
            |_ ngp_mesh (from step 1)
        |_ bboxes (from step 2)
        |_ keypoints_2d (from step 2)
        |_ keypoints_3d (from step 2)
        |_ chosen_frames.json (from step 1)
...
```
