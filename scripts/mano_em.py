import os
import cv2
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.utils.reader import Reader
import src.utils.params as param_utils
from src.utils.parser import add_common_args

sys.path.append("./EasyMocap")
from easymocap.mytools import Timer
from easymocap.dataset import CONFIG
from easymocap.visualize.renderer import Renderer
from easymocap.mytools.file_utils import get_bbox_from_pose
from easymocap.mytools.vis_base import plot_bbox, plot_keypoints, merge
from easymocap.pipeline import smpl_from_keypoints3d2d, smpl_from_keypoints3d
from easymocap.smplmodel import check_keypoints, load_model, select_nf

# parser = load_parser()
parser = argparse.ArgumentParser("Mano Fitting Argument Parser")
parser.add_argument("--separate_calib", action="store_true")
parser.add_argument("--use_optim_params", action="store_true")
parser.add_argument("--use_filtered", action="store_true", help="Whether to use only filtered keypoints (binned)")
# Easy Mocap Arguments
parser.add_argument('--cfg_model', type=str, default=None)
parser.add_argument('--body', type=str, default='body25', choices=['body15', 'body25', 'h36m', 'bodyhand', 'bodyhandface', 'handl', 'handr', 'total'])
parser.add_argument('--model', type=str, default='smpl', choices=['smpl', 'smplh', 'smplx', 'manol', 'manor'])
parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
# optimization control
recon = parser.add_argument_group('Reconstruction control')
recon.add_argument('--robust3d', action='store_true')
# visualization
output = parser.add_argument_group('Output control')
output.add_argument('--vis_det', action='store_true')
output.add_argument('--vis_repro', action='store_true')
output.add_argument('--vis_smpl', action='store_true')
output.add_argument('--write_smpl_full', action='store_true')
parser.add_argument('--write_vertices', action='store_true')
output.add_argument('--vis_mask', action='store_true')
output.add_argument('--undis', action='store_true')
output.add_argument('--save_mesh', action='store_true')
output.add_argument('--sub_vis', type=str, nargs='+', default=[], help='the sub folder lists for visualization')
# debug
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--save_origin', action='store_true')
parser.add_argument('--opts', help="Modify config options using the command-line", 
    default={}, nargs='+')

add_common_args(parser)
args = parser.parse_args()

base_path = os.path.join(args.root_dir, args.seq_path)

# -------------------- Visualization Functions -------------------- #
def vis_smpl(vertices, faces, images, nf, cameras, mode='smpl', extra_data=[], add_back=True):
    outname = os.path.join(base_path, 'mano', '{:08d}.jpg'.format(nf))
    render_data = {}
    assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
    pid = 0
    render_data[pid] = {'vertices': vertices, 'faces': faces, 
        'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
    writer_vis_smpl(render_data, images, cameras, outname, add_back=add_back)

def writer_vis_smpl(render_data, images, cameras, outname, add_back):
    outdir = os.path.dirname(outname)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    render = Renderer(height=1024, width=1024, faces=None)
    render_results = render.render(render_data, cameras, images, add_back=add_back)
    image_vis = merge(render_results, resize=not args.save_origin)
    cv2.imwrite(outname, image_vis)
    return image_vis

# project 3d keypoints from easymocap/mytools/reconstruction.py
def projectN3(kpts3d, cameras):
    # kpts3d: (N, 3)
    nViews = len(cameras)
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = cameras[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    if kpts3d.shape[-1] == 4:
        kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.)
    return kp2ds

# visualize reprojection from easymocap/dataset/mv1pmf.py
def vis_repro(images, kpts_repro, nf, config, to_img=True, mode='repro'):
    lDetections = []
    for nv in range(len(images)):
        det = {
            'id': -1,
            'keypoints2d': kpts_repro[nv],
            'bbox': get_bbox_from_pose(kpts_repro[nv], images[nv])
        }
        lDetections.append([det])
    return vis_detections(images, lDetections, nf, config, mode=mode)

# visualize detections from easymocap/mytools/writer.py
def vis_detections(images, lDetections, nf, config, mode='detec', to_img=True):
    outdir = os.path.join(base_path, 'mano_keypoints')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outname = os.path.join(outdir, '{:06d}.jpg'.format(nf))
    return vis_keypoints2d_mv(images, lDetections, config, outname=outname, vis_id=True)

# visualize 2d keypoints from easymocap/mytools/writer.py
def vis_keypoints2d_mv(images, lDetections, config, outname=None, vis_id=True):
    images_vis = []
    for nv, image in enumerate(images):
        img = image.copy()
        for det in lDetections[nv]:
            pid = det['id']
            if 'keypoints2d' in det.keys():
                keypoints = det['keypoints2d']
            else:
                keypoints = det['keypoints']
            if 'bbox' not in det.keys():
                bbox = get_bbox_from_pose(keypoints, img)
            else:
                bbox = det['bbox']
            plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
            plot_keypoints(img, keypoints, pid=pid, config=config, use_limb_color=False, lw=2)
        images_vis.append(img)
    if len(images_vis) > 1:
        images_vis = merge(images_vis, resize=not args.save_origin)
    else:
        images_vis = images_vis[0]
    if outname is not None:
        cv2.imwrite(outname, images_vis)
    return images_vis
# ----------------------------------------------------------------- #

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

# loads the camera matrices
projs = []
intrs = []
rot = []
trans = []
dist_intrs = []
dists = []
for i in range(len(params)):
    extr = param_utils.get_extr(params[i])
    intr, dist = param_utils.get_intr(params[i])
    r, t = param_utils.get_rot_trans(params[i])

    if args.undis:
        dist_intrs.append(intr.copy())

    projs.append(intr @ extr)
    intrs.append(intr)
    rot.append(r)
    trans.append(t)
    dists.append(dist)

projs = np.asarray(projs)

if args.undis:
    cameras = { 'K': np.asarray(dist_intrs),
                'R': np.asarray(rot), 
                'T': np.asarray(trans) }
else:
    cameras = { 'K': np.asarray(intrs),
                'R': np.asarray(rot), 
                'T': np.asarray(trans) }

# loads the selected frames
if args.use_filtered:
    chosen_path = os.path.join(base_path, "chosen_frames.json")
    with open(chosen_path, "r") as f:
        chosen_frames = set(json.load(f))
else:
    chosen_frames = range(args.start, args.end, args.stride)

# Loads keypoints and bounding boxes
keypoints3d, keypoints2d, bboxes = [], [], []
for idx in tqdm(sorted(chosen_frames), total=len(chosen_frames)):
    frame = f"{idx:08d}"

    keypoints2d_frame = []
    bbox_frame = []
    image_frame = []

    for cam in cam_names:
        # loads the 2d keypoints
        keypoint2d_path = os.path.join(base_path, "keypoints_2d", cam, f"{frame}.json")
        with open(keypoint2d_path, "r") as f:
            keypoints2d_frame.append(json.load(f))

        # loads the bounding boxes
        bbox_path = os.path.join(base_path, "bboxes", cam, f"{frame}.json")
        with open(bbox_path, "r") as f:
            bbox = json.load(f)[0]
            bbox.append(1.0)
            bbox_frame.append(bbox)

    # loads the 3d keypoints
    keypoints3d_path = os.path.join(base_path, "keypoints_3d", f"{frame}.json")
    with open(keypoints3d_path, "r") as f:
        keypoints3d.append(json.load(f))

    keypoints2d.append(keypoints2d_frame)
    bboxes.append(bbox_frame)

keypoints3d = np.asarray(keypoints3d)
keypoints2d = np.asarray(keypoints2d)
bboxes = np.asarray(bboxes)

keypoints3d = check_keypoints(keypoints3d, 1)

# loads the mano model
with Timer('Loading {}, {}'.format(args.model, args.gender), not False):
    body_model = load_model(gender=args.gender, model_type=args.model, model_path="EasyMocap/data/smplx")

# fits the mano model
dataset_config = CONFIG[args.body]
params = smpl_from_keypoints3d2d(body_model, keypoints3d, keypoints2d, bboxes, projs, 
    config=dataset_config, args=args, weight_shape={'s3d': 5000., 'reg_shapes': 0.1}, weight_pose=None)

# visualize model
if args.vis_smpl or args.save_mesh or args.vis_repro:
    import trimesh

    nf = 0
    image_dir = os.path.join(base_path, "synced")
    reader = Reader("video", image_dir)
    for frames, idx in tqdm(reader(chosen_frames), total=len(chosen_frames)):
        images = []
        for cam in cam_names:
            image = frames[cam]
            if args.undis:
                image = param_utils.undistort_image(intrs[i], dist_intrs[i], dists[i], image)
            images.append(image)
        param = select_nf(params, nf)
        # visualizing the model
        if args.vis_smpl:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            vis_smpl(vertices=vertices[0], faces=body_model.faces, images=images, nf=nf, cameras=cameras, add_back=True)

        # save the mesh
        if args.save_mesh:
            vertices = body_model(return_verts=True, return_tensor=False, return_smpl_joints=False, **param)
            mesh = trimesh.Trimesh(vertices=vertices[0], faces=body_model.faces)
            outdir = os.path.join(base_path, 'meshes')
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outname = os.path.join(outdir, '{:08d}.obj'.format(nf))
            mesh.export(outname)

        # visualize keypoint reprojection
        if args.vis_repro:
            keypoints = body_model(return_verts=False, return_tensor=False, **param)[0]
            kpts_repro = projectN3(keypoints, projs)
            vis_repro(images, kpts_repro, config=dataset_config, nf=nf, mode='repro_smpl')

        nf += 1
