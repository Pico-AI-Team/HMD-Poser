import torch
import os
import time
from tqdm import tqdm
import numpy as np
from utils import utils_transform
import math

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

def train_loop(configs, device, model, loss_func, optimizer, \
        lr_scheduler, train_loader, test_loader, LOG, train_summary_writer, out_dir):
    global_step = 0
    best_train_loss, best_test_loss, best_position, best_local_pose = float("inf"), float("inf"), float("inf"), float("inf")
    for epoch in range(configs.train_config.epochs):
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        LOG.info(f'Training epoch: {epoch + 1}/{configs.train_config.epochs}, LR: {current_lr:.6f}')

        one_epoch_root_loss, one_epoch_lpose_loss, one_epoch_gpose_loss, one_epoch_joint_loss, \
            one_epoch_acc_loss, one_epoch_shape_loss, one_epoch_total_loss = [], [], [], [], [], [], []
        for batch_idx, (input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas) in enumerate(train_loader):
            global_step += 1
            optimizer.zero_grad()
            batch_start = time.time()
            input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas = input_feat.to(device).float(), \
                gt_local_pose.to(device).float(), gt_global_pose.to(device).float(), \
                    gt_positions.to(device).float(), gt_betas.to(device).float()
            if len(input_feat.shape) == 4:
                input_feat = torch.flatten(input_feat, start_dim=0, end_dim=1)
                gt_local_pose = torch.flatten(gt_local_pose, start_dim=0, end_dim=1)
                gt_global_pose = torch.flatten(gt_global_pose, start_dim=0, end_dim=1)
                gt_positions = torch.flatten(gt_positions, start_dim=0, end_dim=1)
                gt_betas = torch.flatten(gt_betas, start_dim=0, end_dim=1)

            pred_local_pose, pred_betas, rotation_global_r6d, pred_joint_position = model(input_feat)
            pred_joint_position_head_centered = pred_joint_position - pred_joint_position[:, :, 15:16] + gt_positions[:, :, 15:16]
            gt_positions_head_centered = gt_positions
            root_orientation_loss, local_pose_loss, global_pose_loss, joint_position_loss, accel_loss, shape_loss, total_loss = \
                loss_func(pred_local_pose[:, :, :6], pred_local_pose[:, :, 6:], rotation_global_r6d, pred_joint_position_head_centered, pred_betas, \
                    gt_local_pose[:, :, :6], gt_local_pose[:, :, 6:], gt_global_pose, gt_positions_head_centered, gt_betas)
            
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            lr = optimizer.param_groups[0]['lr']

            one_epoch_root_loss.append(root_orientation_loss.item())
            one_epoch_lpose_loss.append(local_pose_loss.item())
            one_epoch_gpose_loss.append(global_pose_loss.item())
            one_epoch_joint_loss.append(joint_position_loss.item())
            one_epoch_acc_loss.append(accel_loss.item())
            one_epoch_shape_loss.append(shape_loss.item())
            one_epoch_total_loss.append(total_loss.item())
            batch_time = time.time() - batch_start
            
            if batch_idx % configs.train_config.log_interval == 0:
                batch_info = {
                    'type': 'train',
                    'epoch': epoch + 1, 'batch': batch_idx, 'n_batches': len(train_loader),
                    'time': round(batch_time, 5),
                    'lr': round(lr, 8),
                    'loss_total': round(float(total_loss.item()), 3),
                    'root_orientation_loss': round(float(root_orientation_loss.item()), 3),
                    'local_pose_loss': round(float(local_pose_loss.item()), 3),
                    'global_pose_loss': round(float(global_pose_loss.item()), 3),
                    'joint_3d_loss': round(float(joint_position_loss.item()), 3),
                    'smooth_loss': round(float(accel_loss.item()), 3),
                    'shape_loss': round(float(shape_loss.item()), 3)
                }
                LOG.info(batch_info)
        one_epoch_root_loss = torch.tensor(one_epoch_root_loss).mean().item()
        one_epoch_lpose_loss = torch.tensor(one_epoch_lpose_loss).mean().item()
        one_epoch_gpose_loss = torch.tensor(one_epoch_gpose_loss).mean().item()
        one_epoch_joint_loss = torch.tensor(one_epoch_joint_loss).mean().item()
        one_epoch_acc_loss = torch.tensor(one_epoch_acc_loss).mean().item()
        one_epoch_shape_loss = torch.tensor(one_epoch_shape_loss).mean().item()
        one_epoch_total_loss = torch.tensor(one_epoch_total_loss).mean().item()

        train_summary_writer.add_scalar(
            'train_epoch_loss/loss_total', one_epoch_total_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/root_orientation_loss', one_epoch_root_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/local_pose_loss', one_epoch_lpose_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/global_pose_loss', one_epoch_gpose_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/joint_3d_loss', one_epoch_joint_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/smooth_loss', one_epoch_acc_loss, epoch)
        train_summary_writer.add_scalar(
            'train_epoch_loss/shape_loss', one_epoch_shape_loss, epoch)

        epoch_info = {
            'type': 'train',
            'epoch': epoch + 1,
            'loss_total': round(float(one_epoch_total_loss), 3),
            'root_orientation_loss': round(float(one_epoch_root_loss), 3),
            'local_pose_loss': round(float(one_epoch_lpose_loss), 3),
            'global_pose_loss': round(float(one_epoch_gpose_loss), 3),
            'joint_3d_loss': round(float(one_epoch_joint_loss), 3),
            'smooth_loss': round(float(one_epoch_acc_loss), 3),
            'shape_loss': round(float(one_epoch_shape_loss), 3)
        }
        LOG.info(epoch_info)

        if one_epoch_total_loss < best_train_loss:
            LOG.info("Saving model with best train loss in epoch {}".format(epoch+1))
            filename = os.path.join(out_dir, "epoch_with_best_trainloss.pt")
            if os.path.exists(filename):
                os.remove(filename)
            model.save(epoch, filename)
            best_train_loss = one_epoch_total_loss
        
        if epoch % configs.train_config.val_interval == 0:
            model.eval()
            one_epoch_root_loss, one_epoch_lpose_loss, one_epoch_gpose_loss, one_epoch_joint_loss, \
                one_epoch_acc_loss, one_epoch_shape_loss, one_epoch_total_loss = [], [], [], [], [], [], []
            position_error_, local_pose_error_ = [], []
            with torch.no_grad():
                for _, (input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas) in tqdm(enumerate(test_loader)):
                    input_feat, gt_local_pose, gt_global_pose, gt_positions, gt_betas = input_feat.to(device).float(), \
                        gt_local_pose.to(device).float(), gt_global_pose.to(device).float(), \
                            gt_positions.to(device).float(), gt_betas.to(device).float()
                    if len(input_feat.shape) == 4:
                        input_feat = torch.flatten(input_feat, start_dim=0, end_dim=1)
                        gt_local_pose = torch.flatten(gt_local_pose, start_dim=0, end_dim=1)
                        gt_global_pose = torch.flatten(gt_global_pose, start_dim=0, end_dim=1)
                        gt_positions = torch.flatten(gt_positions, start_dim=0, end_dim=1)
                        gt_betas = torch.flatten(gt_betas, start_dim=0, end_dim=1)

                    pred_local_pose, pred_betas, rotation_global_r6d, pred_joint_position = model(input_feat)
                    
                    pred_joint_position_head_centered = pred_joint_position - pred_joint_position[:, :, 15:16] + gt_positions[:, :, 15:16]
                    gt_positions_head_centered = gt_positions
                    root_orientation_loss, local_pose_loss, global_pose_loss, joint_position_loss, accel_loss, shape_loss, total_loss = \
                        loss_func(pred_local_pose[:, :, :6], pred_local_pose[:, :, 6:], rotation_global_r6d, pred_joint_position_head_centered, pred_betas, \
                            gt_local_pose[:, :, :6], gt_local_pose[:, :, 6:], gt_global_pose, gt_positions_head_centered, gt_betas)
                    
                    pos_error_ = torch.mean(torch.sqrt(
                        torch.sum(
                            torch.square(gt_positions_head_centered-pred_joint_position_head_centered),axis=-1
                        )
                    ))
                    position_error_.append(pos_error_.item() * METERS_TO_CENTIMETERS)

                    pred_local_pose_aa = utils_transform.sixd2aa(pred_local_pose.reshape(-1, 6).detach()).reshape(-1, 22*3)
                    gt_local_pose_aa = utils_transform.sixd2aa(gt_local_pose.reshape(-1, 6).detach()).reshape(-1, 22*3)
                    diff = gt_local_pose_aa - pred_local_pose_aa
                    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
                    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
                    rot_error = torch.mean(torch.absolute(diff))
                    local_pose_error_.append(rot_error.item() * RADIANS_TO_DEGREES)

                    one_epoch_root_loss.append(root_orientation_loss.item())
                    one_epoch_lpose_loss.append(local_pose_loss.item())
                    one_epoch_gpose_loss.append(global_pose_loss.item())
                    one_epoch_joint_loss.append(joint_position_loss.item())
                    one_epoch_acc_loss.append(accel_loss.item())
                    one_epoch_shape_loss.append(shape_loss.item())
                    one_epoch_total_loss.append(total_loss.item())
            
            one_epoch_root_loss = torch.tensor(one_epoch_root_loss).mean().item()
            one_epoch_lpose_loss = torch.tensor(one_epoch_lpose_loss).mean().item()
            one_epoch_gpose_loss = torch.tensor(one_epoch_gpose_loss).mean().item()
            one_epoch_joint_loss = torch.tensor(one_epoch_joint_loss).mean().item()
            one_epoch_acc_loss = torch.tensor(one_epoch_acc_loss).mean().item()
            one_epoch_shape_loss = torch.tensor(one_epoch_shape_loss).mean().item()
            one_epoch_total_loss = torch.tensor(one_epoch_total_loss).mean().item()
            position_error_ = torch.tensor(position_error_).mean().item()
            local_pose_error_ = torch.tensor(local_pose_error_).mean().item()

            train_summary_writer.add_scalar(
                'val_epoch_loss/loss_total', one_epoch_total_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/root_orientation_loss', one_epoch_root_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/local_pose_loss', one_epoch_lpose_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/global_pose_loss', one_epoch_gpose_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/joint_3d_loss', one_epoch_joint_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/smooth_loss', one_epoch_acc_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/shape_loss', one_epoch_shape_loss, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/position_error', position_error_, epoch)
            train_summary_writer.add_scalar(
                'val_epoch_loss/local_pose_error', local_pose_error_, epoch)

            epoch_info = {
                'type': 'test',
                'epoch': epoch + 1,
                'loss_total': round(float(one_epoch_total_loss), 3),
                'root_orientation_loss': round(float(one_epoch_root_loss), 3),
                'local_pose_loss': round(float(one_epoch_lpose_loss), 3),
                'global_pose_loss': round(float(one_epoch_gpose_loss), 3),
                'joint_3d_loss': round(float(one_epoch_joint_loss), 3),
                'smooth_loss': round(float(one_epoch_acc_loss), 3),
                'shape_loss': round(float(one_epoch_shape_loss), 3),
                'MPJPE': round(float(position_error_), 3),
                'MPJRE_with_Root': round(float(local_pose_error_), 3)
            }
            LOG.info(epoch_info)
            model.train()

            if one_epoch_total_loss < best_test_loss:
                LOG.info("Saving model with lowest test loss in epoch {}".format(epoch+1))
                filename = os.path.join(out_dir, "epoch_with_best_testloss.pt")
                if os.path.exists(filename):
                    os.remove(filename)
                model.save(epoch, filename)
                best_test_loss = one_epoch_total_loss
            
            if position_error_ < best_position:
                best_position = position_error_
                LOG.info("Lowest MPJPE {} in epoch {}".format(best_position, epoch+1))
            
            if local_pose_error_ < best_local_pose:
                best_local_pose = local_pose_error_
                LOG.info("Lowest MPJRE_(including root) {} in epoch {}".format(best_local_pose, epoch+1))
