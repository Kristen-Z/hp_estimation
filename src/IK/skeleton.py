import numpy as np
import torch
from tqdm import trange
from torch.optim import Adam
from torch.nn.parameter import Parameter
from src.utils.transforms import (
    euler_angles_to_matrix,
    get_keypoints,
    get_pose_wrt_root,
)


class KinematicChain:
    def __init__(self, bones, device=torch.device("cpu")):
        self.device = device
        self.bones = bones
        self.parents = [-1 for _ in range(len(bones))]
        # Temporary, rename if stuff works
        self.rest_matrices = [np.eye(4) for _ in range(len(bones))]
        self.kintree = {}
        self.tails = torch.zeros(
            (len(self.bones), 3), dtype=torch.float32, device=device
        )
        self.heads = torch.zeros(
            (len(self.bones), 3), dtype=torch.float32, device=device
        )
        for _, bone in bones.items():
            bone_idx = bone["idx"]
            self.rest_matrices[bone_idx] = bone["rest_matrix"]
            self.parents[bone_idx] = bone["parent_idx"]
            self.kintree[str(bone_idx)] = bone["parent_idx"]
            self.tails[bone_idx] = torch.tensor(bone["tail"])
            self.heads[bone_idx] = torch.tensor(bone["head"])
        self.rest_matrices = torch.from_numpy(np.asarray(self.rest_matrices)).to(device).float()

        self.dof = torch.zeros(
            (len(self.bones) + 1, 3), dtype=torch.bool
        )  # num_joint, 3 (x, y, z)

        self.limits = torch.zeros(
            (len(self.bones) + 1, 3, 2), dtype=torch.float, device=device
        )  # num_joint, 3 (x, y, z), 2 (low, high)

        self.limits[:, :, 0] = -torch.pi
        self.limits[:, :, 1] = torch.pi

        xz = [True, False, True]  # useful for dof

        # DOF taken from http://www.iri.upc.edu/files/academic/pfc/74-PFC.pdf#page=31
        # RU
        self.dof[0, :] = True  # all

        # CMC*
        self.dof[1, xz] = True  # xz
        self.dof[2, xz] = True  # xz
        # TODO: Figure out limits on thumbs
        # self.limits[1, 0, 0] = -torch.pi / 6
        # self.limits[1, 0, 1] = torch.pi / 6
        # self.limits[1, 2, 0] = -torch.pi / 2
        # self.limits[1, 2, 1] = 0

        # CMC
        # self.dof[5:18:4, 2] = True  # z
        # self.limits[5:18:4, 2, 0] = -torch.pi / 2
        # self.limits[5:18:4, 2, 1] = 0

        # MCP
        self.dof[3, xz] = True  # xz
        self.dof[6:19:4, xz] = True
        self.limits[6:19:4, 0, 0] = -torch.pi / 6
        self.limits[6:19:4, 0, 1] = torch.pi / 6
        self.limits[6:19:4, 2, 0] = -torch.pi / 2
        self.limits[6:19:4, 2, 1] = torch.pi / 9

        # PIP
        self.dof[4, 2] = True
        self.dof[7:20:4, 2] = True
        self.limits[7:20:4, 2, 0] = -torch.pi / 2
        self.limits[7:20:4, 2, 1] = 0

        # DIP
        self.dof[8:21:4, 2] = True
        self.limits[8:21:4, 2, 0] = -torch.pi / 2
        self.limits[8:21:4, 2, 1] = 0

    def plot_skeleton(self, trans, angles, target=None):
        """Debug function"""
        import polyscope as ps
        import trimesh

        ps.init()

        (
            _,
            heads,
            tails,
        ) = self.forward(trans, angles)
        heads = heads.detach().cpu().numpy()
        tails = tails.detach().cpu().numpy()
        pt_cloud = ps.register_point_cloud("heads", heads)
        ps.register_point_cloud("tails", tails)

        pt_cloud.add_vector_quantity(
            "bones",
            tails - heads,
            color=(0, 0, 0),
            enabled=True,
            vectortype="ambient",
            radius=0.004,
        )

        if not target is None:
            ps.register_point_cloud("target", target)

        pcd = trimesh.PointCloud(heads)
        _ = pcd.export("points.ply")

        ps.show()

    def loss(self, trans_params, angle_params, target, to_use, constraint, limit=False):
        predicted, _, __ = self.forward(trans_params, angle_params, constraint)
        keypoint_loss = predicted - target
        keypoint_loss = torch.square(torch.linalg.norm(keypoint_loss, axis=1))
        keypoint_loss = keypoint_loss[to_use].mean()
        loss = {"keypoint_loss": keypoint_loss}
        if limit:
            if constraint:
                limits = self.limits.reshape(-1, 2)[self.dof.flatten()]
            else:
                limits = self.limits.reshape(-1, 2)
            limit_loss_lo = limits[:, 0] - angle_params
            limit_loss_lo = torch.where(limit_loss_lo > 0, limit_loss_lo, 0)
            limit_loss = torch.sum(torch.square(limit_loss_lo))
            limit_loss_hi = limits[:, 1] - angle_params
            limit_loss_hi = torch.where(limit_loss_hi < 0, limit_loss_hi, 0)
            limit_loss += torch.sum(torch.square(limit_loss_hi))
            loss["limit_loss"] = limit_loss
        return loss

    def forward(self, trans_params, angle_params, constraint=False):
        angles = torch.zeros((len(self.bones) + 1) * 3, device=self.device)
        if constraint:
            angles[self.dof.flatten()] = angle_params
        else:
            angles = angle_params

        angles = angles.reshape(-1, 3)

        pose_matrices = euler_angles_to_matrix(angles, "XYZ", intrinsic=True)
        global_translation = trans_params.unsqueeze(0)
        matrix = get_pose_wrt_root(
            self.rest_matrices,
            pose_matrices[1:].unsqueeze(0),
            pose_matrices[:1],
            global_translation,
            self.kintree,
        )

        heads = get_keypoints(matrix, self.rest_matrices, self.heads).squeeze(0)
        tails = get_keypoints(matrix, self.rest_matrices, self.tails).squeeze(0)

        scaled_tails = torch.zeros_like(tails)
        scaled_heads = torch.zeros_like(heads)
        for i in range(len(self.kintree)):
            parent = self.kintree[str(i)]
            if parent == -1:
                scaled_heads[i] = heads[i]
            else:
                scaled_heads[i] = scaled_tails[parent]
            dir_vec = tails[i] - heads[i]
            dir_vec = dir_vec / torch.linalg.norm(dir_vec)
            scaled_tails[i] = scaled_heads[i] + dir_vec * self.bones[f"bone_{i}"]["len"]

        keypoints = torch.vstack([scaled_heads[:1], scaled_tails])
        return keypoints, scaled_heads, scaled_tails

    def update_bone_lengths(self, keypoints: np.ndarray):
        for bone_name, bone in self.bones.items():
            curr_id = bone["idx"] + 1
            parent_id = bone["parent_idx"] + 1
            bone_vecs = keypoints[:, curr_id] - keypoints[:, parent_id]
            to_use = ~torch.logical_or(
                torch.isclose(
                    keypoints[:, curr_id, 3], torch.tensor(0.0, device=self.device)
                ),
                torch.isclose(
                    keypoints[:, parent_id, 3], torch.tensor(0.0, device=self.device)
                ),
            )
            if not torch.count_nonzero(to_use):
                raise ValueError(f"No frame has length of bone {bone_name}")
            bone_lens = torch.linalg.norm(bone_vecs[:, :3], axis=1)[to_use]
            self.bones[bone_name]["len"] = bone_lens.mean().item()

    def IK(
        self,
        target,
        to_use,
        constraint,
        limit,
        lr=1e-1,
        trans_init=None,
        angles_init=None,
        max_iter=10000,
        threshold=1e-6,
    ):
        if trans_init is None:
            trans_init = torch.zeros(3, device=self.device)
        trans_params = Parameter(trans_init)

        if angles_init is None:
            # 20 Joint angles + Global rotation
            angles_init = torch.zeros(
                (len(self.bones) + 1, 3), device=self.device
            ).flatten()

        if constraint:
            angle_params = Parameter(angles_init[self.dof.flatten()])
        else:
            angle_params = Parameter(angles_init)

        optimizer = Adam([trans_params, angle_params], lr=lr)

        pbar = trange(max_iter)
        least_loss = 1e10
        least_iter = 0
        least_params = angle_params
        for i in pbar:
            optimizer.zero_grad()
            loss = self.loss(
                trans_params, angle_params, target, to_use, constraint, limit
            )
            total_loss = loss["keypoint_loss"] + loss["limit_loss"] if limit else 0
            total_loss.backward()
            optimizer.step()
            pbar.set_description(
                f"loss: {total_loss:.6f}"
                + f", keypoint_loss: {loss['keypoint_loss']:.6f}"
                + f", limit_loss: {loss['limit_loss']:.6f}"
                if limit
                else ""
            )

            # Early stopping
            if total_loss < least_loss:
                least_loss = total_loss
                least_iter = i
                least_params = angle_params.detach().clone()
            if i - least_iter > 10 and abs(total_loss - least_loss) < threshold:
                break

        if constraint:
            to_return = torch.zeros(
                (len(self.bones) + 1, 3), device=self.device
            ).flatten()
            to_return[self.dof.flatten()] = least_params
        else:
            to_return = least_params

        return trans_params, to_return
