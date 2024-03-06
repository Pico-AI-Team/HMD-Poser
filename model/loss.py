"""
    loss definition
"""
import torch
from torch import nn

class PoseJointLoss(nn.Module):
    def __init__(self, loss_type, root_orientation_weight=1.0,  
                local_pose_weight=1.0, 
                global_pose_weight=1.0, 
                joint_position_weight=1.0,
                acc_loss_weight=0.5,
                shape_loss_weight=0.5) -> None:
        super().__init__()
        self.root_orientation_weight = root_orientation_weight
        self.local_pose_weight = local_pose_weight
        self.global_pose_weight = global_pose_weight
        self.joint_position_weight = joint_position_weight
        self.acc_loss_weight = acc_loss_weight
        self.shape_loss_weight = shape_loss_weight
        if loss_type == 'l1':
            self.lossfn = nn.L1Loss()
        elif loss_type == 'l2':
            self.lossfn = nn.MSELoss()
        elif loss_type == 'l2sum':
            self.lossfn = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(loss_type))

    def forward(self, pred_root_orientation, 
                    pred_local_rotation, 
                    pred_global_rotation,
                    pred_joint_position,
                    pred_shape,
                    gt_root_orientation, 
                    gt_local_rotation, 
                    gt_global_rotation,
                    gt_joint_position,
                    gt_shape
                ):
        assert pred_root_orientation.shape == gt_root_orientation.shape and \
                pred_local_rotation.shape == gt_local_rotation.shape and \
                pred_global_rotation.shape == gt_global_rotation.shape and \
                pred_joint_position.shape == gt_joint_position.shape and \
                pred_shape.shape == gt_shape.shape, "GT and Pred have different shape"
        
        root_orientation_loss = self.lossfn(pred_root_orientation, gt_root_orientation)
        local_pose_loss = self.lossfn(pred_local_rotation, gt_local_rotation)
        global_pose_loss = self.lossfn(pred_global_rotation, gt_global_rotation)
        joint_position_loss = self.lossfn(pred_joint_position, gt_joint_position)
        shape_loss = self.lossfn(pred_shape, pred_shape.mean(dim=1, keepdim=True).repeat(1, pred_shape.shape[1], 1))
        
        accel_gt = gt_joint_position[:,:-2,:] - 2 * gt_joint_position[:,1:-1,:] + gt_joint_position[:,2:,:]
        accel_pose = pred_joint_position[:,:-2,:] - 2 * pred_joint_position[:,1:-1,:] + pred_joint_position[:,2:,:]
        accel_loss = self.lossfn(accel_pose, accel_gt)


        total_loss = (self.root_orientation_weight * root_orientation_loss + 
        self.local_pose_weight * local_pose_loss +
        self.global_pose_weight * global_pose_loss +
        self.joint_position_weight * joint_position_loss +
        self.acc_loss_weight * accel_loss +
        self.shape_loss_weight * shape_loss
        )

        return root_orientation_loss, local_pose_loss, global_pose_loss, joint_position_loss, accel_loss, shape_loss, total_loss