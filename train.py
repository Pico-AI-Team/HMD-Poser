import torch
import os
import argparse
import random
import numpy as np
from utils.utils_config import load_config, add_log, LearningRateLambda
from dataset.dataloader import load_data, TrainDataset, TestDataset
from torch.utils.data import DataLoader
from model.hmd_imu_model import HMDIMUModel
from model.loss import PoseJointLoss
from runner.training_loop import train_loop

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    configs = load_config(args.config)
    out_dir, LOG, train_summary_writer = add_log(configs)
    dst_file = os.path.join(out_dir, os.path.basename(args.config))
    os.system('cp {} {}'.format(args.config, dst_file))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    '''
    # ----------------------------------------
    # seed
    # ----------------------------------------
    '''
    seed = configs.manual_seed
    if seed is None:
        seed = random.randint(1, 10000)
    LOG.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # creat dataloader
    # ----------------------------------------
    '''
    LOG.info("creating train dataloader...")
    train_datas = load_data(configs.dataset_path, "train", 
        input_motion_length=configs.input_motion_length)
    train_dataset = TrainDataset(train_datas, configs.compatible_inputs, 
        configs.input_motion_length, configs.train_dataset_repeat_times)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=configs.train_config.batch_size,
        shuffle=True,
        num_workers=configs.train_config.num_workers,
        drop_last=True,
        persistent_workers=False,
    )
    LOG.info("There are {} video sequence, {} item and {} batch in the training set.".format(
        len(train_datas),
        len(train_dataset),
        len(train_dataloader)
        )
    )

    LOG.info("creating test dataloader...")
    test_datas = load_data(configs.dataset_path, "test", 
        input_motion_length=configs.input_motion_length)
    test_dataset = TestDataset(test_datas, configs.compatible_inputs,
        configs.input_motion_length, 1)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )
    LOG.info("There are {} video sequence, {} item and {} batch in the test set.".format(
        len(test_datas),
        len(test_dataset),
        len(test_dataloader)
        )
    )

    LOG.info("creating model...")
    model = HMDIMUModel(configs, device)
    if configs.resume_model is not None:
        model.load(configs.resume_model)
        LOG.info(f"successfully resume checkpoint in {configs.resume_model}")
    

    loss_func = PoseJointLoss(configs.loss.loss_type,
        configs.loss.root_orientation_weight,
        configs.loss.local_pose_weight,
        configs.loss.global_pose_weight,
        configs.loss.joint_position_weight,
        configs.loss.smooth_loss_weight,
        configs.loss.shape_loss_weight
    ).to(device)

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
        lr=configs.lr_config.lr, 
        betas=(configs.optimizer_config.momentum, configs.optimizer_config.beta2), 
        weight_decay=configs.optimizer_config.weight_decay, 
        eps=configs.optimizer_config.adam_eps, 
        amsgrad=configs.optimizer_config.amsgrad
    )
    
    training_batches_per_epoch = len(train_dataloader)
    last_epoch = 0
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            LearningRateLambda(
                [s * training_batches_per_epoch for s in configs.lr_config.lr_decay],
                decay_factor=configs.lr_config.lr_decay_factor,
                decay_epochs=configs.lr_config.lr_decay_epochs * training_batches_per_epoch,
                warm_up_start_epoch=configs.lr_config.lr_warm_up_start_epoch * training_batches_per_epoch,
                warm_up_epochs=configs.lr_config.lr_warm_up_epochs * training_batches_per_epoch,
                warm_up_factor=configs.lr_config.lr_warm_up_factor,
                warm_restart_schedule=[
                    r * training_batches_per_epoch for r in configs.lr_config.lr_warm_restarts],
                warm_restart_duration=configs.lr_config.lr_warm_restart_duration * training_batches_per_epoch,
            ),
        ],
        last_epoch=last_epoch * training_batches_per_epoch - 1,
    )
    
    train_loop(configs, device, model, loss_func, optimizer, lr_scheduler, train_dataloader, test_dataloader, LOG, train_summary_writer, out_dir)


if __name__ == "__main__":
    main()