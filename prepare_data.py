
import argparse
import os

import numpy as np
import torch
import glob
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from tqdm import tqdm
from utils import utils_transform

def _syn_acc(v, smooth_n=4):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0 and v.shape[0] > smooth_n * 2:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def generate_input_features(bdata_poses, body_pose_world, bm):
    output_aa = torch.Tensor(bdata_poses[:, :66]).reshape(-1, 3)
    output_6d = utils_transform.aa2sixd(output_aa).reshape(
        bdata_poses.shape[0], -1
    )
    rotation_local_full_gt_list = output_6d[1:]

    rotation_local_matrot = aa2matrot(
        torch.tensor(bdata_poses).reshape(-1, 3)
    ).reshape(bdata_poses.shape[0], -1, 9)
    rotation_global_matrot = local2global_pose(
        rotation_local_matrot, bm.kintree_table[0].long()
    )  # rotation of joints relative to the origin
    
    head_rotation_global_matrot = rotation_global_matrot[:, [15], :, :]

    rotation_global_6d = utils_transform.matrot2sixd(
        rotation_global_matrot.reshape(-1, 3, 3)
    ).reshape(rotation_global_matrot.shape[0], -1, 6)
    input_rotation_global_6d = rotation_global_6d[1:, [15, 20, 21, 4, 5, 0], :]

    rotation_velocity_global_matrot = torch.matmul(
        torch.inverse(rotation_global_matrot[:-1]),
        rotation_global_matrot[1:],
    )
    rotation_velocity_global_6d = utils_transform.matrot2sixd(
        rotation_velocity_global_matrot.reshape(-1, 3, 3)
    ).reshape(rotation_velocity_global_matrot.shape[0], -1, 6)
    input_rotation_velocity_global_6d = rotation_velocity_global_6d[
        :, [15, 20, 21, 4, 5, 0], :
    ]

    position_global_full_gt_world = body_pose_world.Jtr[
        :, :22, :
    ]  # position of joints relative to the world origin

    position_head_world = position_global_full_gt_world[
        :, 15, :
    ]  # world position of head

    head_global_trans = torch.eye(4).repeat(
        position_head_world.shape[0], 1, 1
    )
    head_global_trans[:, :3, :3] = head_rotation_global_matrot.squeeze()
    head_global_trans[:, :3, 3] = position_global_full_gt_world[:, 15, :]

    head_global_trans_list = head_global_trans[1:]

    num_frames = position_global_full_gt_world.shape[0] - 1

    # provide the hand representations in the head space
    hands_rotation_mat_in_head_space = rotation_global_matrot[
        :, 15:16, :, :].transpose(2, 3).matmul(
            rotation_global_matrot[:, [20, 21], :, :])
    hands_rotation_in_head_space_r6d = utils_transform.matrot2sixd(
        hands_rotation_mat_in_head_space.reshape(-1, 3, 3)
    ).reshape(hands_rotation_mat_in_head_space.shape[0], -1, 6)[1:]
    
    rotation_velocity_handsinheadspace = torch.matmul(
        torch.inverse(hands_rotation_mat_in_head_space[:-1]),
        hands_rotation_mat_in_head_space[1:],
    )
    rotation_velocity_handsinheadspace_r6d = utils_transform.matrot2sixd(
        rotation_velocity_handsinheadspace.reshape(-1, 3, 3)
    ).reshape(rotation_velocity_handsinheadspace.shape[0], -1, 6)

    hands_position_in_head_space = (position_global_full_gt_world[:, [20, 21], :] - position_global_full_gt_world[:, 15:16, :]).double().bmm(rotation_global_matrot[:, 15])
    
    foot_accs = _syn_acc(body_pose_world.v[:, [1176, 4662, 3021]])[1:, :, :]

    hmd_position_global_full_gt_list = torch.cat(
        [
            input_rotation_global_6d.reshape(num_frames, -1),
            input_rotation_velocity_global_6d.reshape(num_frames, -1),
            position_global_full_gt_world[1:, [15, 20, 21], :].reshape(
                num_frames, -1
            ),
            position_global_full_gt_world[1:, [15, 20, 21], :].reshape(
                num_frames, -1
            )
            - position_global_full_gt_world[:-1, [15, 20, 21], :].reshape(
                num_frames, -1
            ),
            hands_rotation_in_head_space_r6d.reshape(num_frames, -1),
            rotation_velocity_handsinheadspace_r6d.reshape(num_frames, -1),
            hands_position_in_head_space[1:].reshape(num_frames, -1),
            hands_position_in_head_space[1:].reshape(num_frames, -1) - hands_position_in_head_space[:-1].reshape(num_frames, -1),
            foot_accs.reshape(num_frames, -1),
        ],
        dim=-1,
    )
    data = {}
    data["rotation_local_full_gt_list"] = rotation_local_full_gt_list
    data["rotation_global_full_gt_list"] = rotation_global_6d[1:, :22].reshape(num_frames, -1).cpu().float()
    data[
        "hmd_position_global_full_gt_list"
    ] = hmd_position_global_full_gt_list
    data["head_global_trans_list"] = head_global_trans_list
    data["position_global_full_gt_world"] = (
        position_global_full_gt_world[1:].cpu().float()
    )
    return data

def process_protocol1(args, bm_male, bm_female):
    for dataroot_subset in ["BioMotionLab_NTroje", "CMU", "MPI_HDM05"]:
        print(dataroot_subset)
        for phase in ["train", "test"]:
            print(phase)
            savedir = os.path.join(args.save_dir, dataroot_subset, phase)
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            split_file = os.path.join(
                "prepare_data/data_split", dataroot_subset, phase + "_split.txt"
            )

            with open(split_file, "r") as f:
                filepaths = [line.strip() for line in f]

            idx = 0
            data_total = []
            for filepath in tqdm(filepaths):
                bdata = np.load(
                    os.path.join(args.root_dir, filepath), allow_pickle=True
                )

                if "mocap_framerate" in bdata:
                    framerate = bdata["mocap_framerate"]
                else:
                    continue

                if framerate == 120:
                    stride = 2
                elif framerate == 60:
                    stride = 1
                else:
                    raise AssertionError(
                        "Please check your AMASS data, should only have 2 types of framerate, either 120 or 60!!!"
                    )

                bdata_poses = bdata["poses"][::stride, ...]
                bdata_trans = bdata["trans"][::stride, ...]
                bdata_betas = bdata["betas"]
                subject_gender = bdata["gender"]

                bm = bm_male if subject_gender == 'male' else bm_female

                body_parms = {
                    "root_orient": torch.Tensor(
                        bdata_poses[:, :3]
                    ),  # .to(comp_device), # controls the global root orientation
                    "pose_body": torch.Tensor(
                        bdata_poses[:, 3:66]
                    ),  # .to(comp_device), # controls the body
                    "trans": torch.Tensor(
                        bdata_trans
                    ),  # .to(comp_device), # controls the global body position
                    'betas': torch.Tensor(
                        bdata_betas
                    ).repeat(bdata_poses.shape[0], 1),
                }

                body_parms_list = body_parms

                body_pose_world = bm(
                    **{
                        k: v
                        for k, v in body_parms.items()
                        if k in ["pose_body", "root_orient", "trans", 'betas']
                    }
                )

                data = generate_input_features(bdata_poses, body_pose_world, bm)
                data["body_parms_list"] = body_parms_list
                data["framerate"] = 60
                data["gender"] = subject_gender
                data["filepath"] = filepath
                data_total.append(data)
                idx += 1
            torch.save(data_total, os.path.join(savedir, "{}.pt".format(idx)))

def process_protocol2(args, bm_male, bm_female):
    train_datasets = ["ACCAD", "BioMotionLab_NTroje", "BMLmovi", \
        "CMU", "EKUT", "Eyes_Japan_Dataset", "KIT", "MPI_HDM05", \
        "MPI_Limits", "MPI_mosh", "SFU", "TotalCapture"]
    test_datasets = ["HumanEva", "Transitions_mocap"]
    phase_all = ["train"]*len(train_datasets) + ["test"]*len(test_datasets)
    datasets_all = train_datasets + test_datasets
    for dataset_idx, dataroot_subset in enumerate(datasets_all):
        print(dataroot_subset)
        phase = phase_all[dataset_idx]
        savedir = os.path.join(args.save_dir, dataroot_subset, phase)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        filepaths = glob.glob(os.path.join(args.root_dir, dataroot_subset, '*/*_poses.npz'))

        idx = 0
        data_total = []
        for filepath in tqdm(filepaths):
            data = {}
            bdata = np.load(
                filepath, allow_pickle=True
            )

            if "mocap_framerate" in bdata:
                framerate = bdata["mocap_framerate"]
            else:
                continue
            
            if framerate == 120:
                stride = 2
            elif framerate == 60:
                stride = 1
            else:
                stride = round(framerate/60.0)

            bdata_poses = bdata["poses"][::stride, ...]
            bdata_trans = bdata["trans"][::stride, ...]
            bdata_betas = bdata["betas"]
            subject_gender = bdata["gender"]

            if bdata_poses.shape[0] < 10:
                continue

            bm = bm_male if subject_gender == 'male' else bm_female

            body_parms = {
                "root_orient": torch.Tensor(
                    bdata_poses[:, :3]
                ),  # .to(comp_device), # controls the global root orientation
                "pose_body": torch.Tensor(
                    bdata_poses[:, 3:66]
                ),  # .to(comp_device), # controls the body
                "trans": torch.Tensor(
                    bdata_trans
                ),  # .to(comp_device), # controls the global body position
                'betas': torch.Tensor(
                    bdata_betas
                ).repeat(bdata_poses.shape[0], 1),
            }

            body_parms_list = body_parms

            body_pose_world = bm(
                **{
                    k: v
                    for k, v in body_parms.items()
                    if k in ["pose_body", "root_orient", "trans", 'betas']
                }
            )

            data = generate_input_features(bdata_poses, body_pose_world, bm)
            data["body_parms_list"] = body_parms_list
            data["framerate"] = 60
            data["gender"] = subject_gender
            data["filepath"] = filepath
            data_total.append(data)
            idx += 1
        torch.save(data_total, os.path.join(savedir, "{}.pt".format(idx)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default=None,
        help="=dir where you put your smplh and dmpls dirs",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="=dir where you want to save your generated data",
    )
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default=None, 
        help="=dir where you put your AMASS/HMD data",
    )
    parser.add_argument(
        "--protocol", 
        type=str, 
        choices=['protocol1', 'protocol2'],
        default='protocol1',
        help="=protocol1 or protocol2",
    )
    args = parser.parse_args()

    # Different from the AvatarPoser paper, we use male/female model for each sequence
    bm_fname_male = os.path.join(args.support_dir, "smplh/{}/model.npz".format("male"))
    dmpl_fname_male = os.path.join(
        args.support_dir, "dmpls/{}/model.npz".format("male")
    )

    bm_fname_female = os.path.join(args.support_dir, 'smplh/{}/model.npz'.format('female'))
    dmpl_fname_female = os.path.join(args.support_dir, 'dmpls/{}/model.npz'.format('female'))

    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    bm_male = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname_male,
    )

    bm_female = BodyModel(
        bm_fname=bm_fname_female, 
        num_betas=num_betas, 
        num_dmpls=num_dmpls, 
        dmpl_fname=dmpl_fname_female
    )
    if args.protocol == 'protocol1':
        process_protocol1(args, bm_male, bm_female)
    elif args.protocol == 'protocol2':
        process_protocol2(args, bm_male, bm_female)
    else:
        raise NotImplementedError("Protocol {} is not implemented".format(args.protocol))