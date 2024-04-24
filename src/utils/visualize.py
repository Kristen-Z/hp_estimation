import numpy as np
import cv2
from src.IK.skeleton import KinematicChain
from typing import Tuple

point_pairs = [[0, 1], [1, 2], [2, 3], [3, 4],
               [0, 5], [5, 6], [6, 7], [7, 8],
               [0, 9], [9, 10], [10, 11], [11, 12],
               [0, 13], [13, 14], [14, 15], [15, 16],
               [0, 17], [17, 18], [18, 19], [19, 20]]


def project(keypoints3d: np.ndarray, P: np.ndarray):
    """
    Project keypoints to 2D using

    Inputs -
        keypoints3d (N, 3): 3D keypoints
        P (V,3,4): Projection matrices
    Outputs -
        keypoints2d (V, N, 2): Projected 2D keypoints
    """
    hom = np.hstack((keypoints3d, np.ones((keypoints3d.shape[0], 1))))
    projected = np.matmul(P, hom.T).transpose(0, 2, 1)  # (V, N, 2)
    projected = (projected / projected[:, :, -1:])[:, :, :-1]
    return projected


def plot_keypoints_2d(
    joints: np.ndarray,
    image: np.ndarray,
    chain: KinematicChain,
    proj_mat,
    point_color: Tuple[int] = (0, 0, 255),
    bone_color: Tuple[int] = (255, 0, 0),
    plot_bones: bool = True,
) -> np.ndarray:
    res = image.copy()
    joint_radius = min(*image.shape[:2]) // 150
    if proj_mat is None:
        if plot_bones:
            for pair in point_pairs:
                partA = pair[0]
                partB = pair[1]
                
                cv2.line(res, (int(joints[partA][0]),int(joints[partA][1])), (int(joints[partB][0]),int(joints[partB][1])), bone_color, joint_radius // 2)
                cv2.circle(res, (int(joints[partA][0]),int(joints[partA][1])), joint_radius, point_color, -1)
        else:
            for keypoint in joints:
                cv2.circle(res, (int(keypoint[0]), int(keypoint[1])), joint_radius, point_color, -1)
        return res
    else:
        keypoints_2d = project(joints, np.asarray([proj_mat]))[0]

    for keypoint in keypoints_2d:
        cv2.circle(
            res, (int(keypoint[0]), int(keypoint[1])
                  ), joint_radius, point_color, -1
        )

    if plot_bones:
        for (
            bone,
            parent,
        ) in chain.kintree.items():
            parent_id = parent + 1
            bone_id = int(bone) + 1
            cv2.line(
                res,
                (int(keypoints_2d[bone_id][0]), int(keypoints_2d[bone_id][1])),
                (int(keypoints_2d[parent_id][0]),
                 int(keypoints_2d[parent_id][1])),
                bone_color,
                joint_radius // 2,
            )

    return res

def plot_keypoints_ransac(
    joints: np.ndarray,
    image: np.ndarray,
    point_color: Tuple[int] = (0, 0, 255),
    bone_color: Tuple[int] = (255, 0, 0),
    plot_bones: bool = True,
) -> np.ndarray:
    res = image.copy()
    joint_radius = min(*image.shape[:2]) // 150
    if plot_bones:
        for pair in point_pairs:
            partA = pair[0]
            partB = pair[1]
            
            cv2.line(res, (int(joints[partA][0]),int(joints[partA][1])), (int(joints[partB][0]),int(joints[partB][1])), bone_color, joint_radius // 2)
            cv2.circle(res, (int(joints[partA][0]),int(joints[partA][1])), joint_radius, point_color, -1)
        cv2.circle(res, (int(joints[-1][0]),int(joints[-1][1])), joint_radius, point_color, -1)
    else:
        for keypoint in joints:
            cv2.circle(res, (int(keypoint[0]), int(keypoint[1])), joint_radius, point_color, -1)

    return res