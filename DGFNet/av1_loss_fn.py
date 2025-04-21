from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import gpu, to_long
from utils.loss_utils import topo_loss


class LossFunc(nn.Module):
    def __init__(self, config, device):
        super(LossFunc, self).__init__()
        self.config = config
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")
        # self.topo_loss = topo_loss() # // 新加

    def forward(self, out_sc, data, epoch):

        loss_out = self.pred_loss(
                                  out_sc,
                                  gpu(data["TRAJS_FUT"], self.device),
                                  to_long(gpu(data["PAD_FUT"], self.device)),
                                  gpu(data["TRAJS_FUT_ORI"], self.device), # dgf
                                  epoch
                                  )
        # print(len(data["TRAJS_FUT"]))
        # print(data["TRAJS_FUT_ORI"].type)
        loss_out["loss"] = loss_out["cls_loss"] + loss_out["reg_loss"] + loss_out["reg_loss_final"] + loss_out["topo_loss"] # dgf
        return loss_out


    def pred_loss(self, out_sc: Dict[str, List[torch.Tensor]], gt_preds: List[torch.Tensor], pad_flags: List[torch.Tensor], gt_preds_sc: List[torch.Tensor], epoch):
        cls, reg, reg_final= map(lambda x: torch.cat(x, 0), out_sc[:3])  # dgf
        # print("cls",cls) # (56,6) tensor([[0.1659, 0.1625, 0.1701, 0.1694, 0.1671, 0.1649],
        topo_pred, braids, actor_topo_mask = out_sc[3], out_sc[4], out_sc[5] # (1,14,14,1) (1,14,14,1,) (1,1,14)
        # topo_pred = topo_pred.squeeze(-1)
        # topo_pred = F.softmax(topo_pred, dim=-1)
        # topo_pred = topo_pred.unsqueeze(-1) # 0410 0418
        # topo_pred = topo_pred[:, 0, :, :]
        # braids = braids[:, 0, :, :]
        print("topo_pred",topo_pred) # tensor([[[[-0.1253],
        print("braids",braids) # tensor([[[[False],
        # print("actor_topo_mask",actor_topo_mask) # (2,1,42) tensor([[[ True,  True,  True, False,  True, False, False, False, False, False,

        gt_preds = torch.cat(gt_preds, 0)  # dgf (56,30,2)
        has_preds = torch.cat(pad_flags, 0).bool()  # dgf (56,30)
        # print("gt_preds",gt_preds) # tensor([[[ 2.0900e-01, -2.4069e-02],
        # print("has_preds",has_preds) # tensor([[ True,  True,  True,  ...,  True,  True,  True],

        N = actor_topo_mask.shape[2]
        actor_topo_mask = actor_topo_mask.expand(-1, N, -1)  # 使用广播机制扩展 (2,42,42)
        # print("actor_topo_mask",actor_topo_mask) #  tensor([[[ True,  True,  True,  ..., False, False, False],

        loss_out = dict()
        num_modes = self.config["g_num_modes"]
        num_preds = 30

        mask, last_idcs = self.create_mask(has_preds, num_preds)  # dgf
        cls, reg, reg_final, gt_preds, has_preds, last_idcs = map(lambda x: x[mask], [cls, reg, reg_final, gt_preds, has_preds, last_idcs])  # dgf
        # print("cls",cls) # tensor([[0.1659, 0.1625, 0.1701, 0.1694, 0.1671, 0.1649],
        # print("reg",reg) # (56,6,30,2) tensor([[[[-0.8462, -0.7651],
        # print("reg_final",reg_final) # (56,6,30,2) tensor([[[[ 8.5566e-01,  6.9115e-01],
        # print("gt_preds",gt_preds) # tensor([[[ 2.0900e-01, -2.4069e-02],
        # print("has_preds",has_preds) # tensor([[ True,  True,  True,  ...,  True,  True,  True],
        # print("reg[..., 0:2]",reg[..., 0:2]) # (56,6,30,2) tensor([[[[-0.8462, -0.7651],

        dist, min_dist, min_idcs = self.get_min_distance_indices(reg[..., 0:2].clone(), gt_preds, last_idcs, num_modes)  # dgf

        topo_loss = self.topo_loss(topo_pred, braids.detach(), actor_topo_mask.float().detach()) # (2,) 归一化之后的topo_pred_ 0410 braids.detach(), actor_topo_mask.float().detach()
        print("topo_loss",topo_loss) # tensor(0.1350, device='cuda:0', grad_fn=<MeanBackward0>)  tensor(0.2686, device='cuda:0', grad_fn=<MeanBackward0>)
        # 是归一化之后的拓扑预测还是需要传给模型的拓扑预测  根据reg是模型得出的结果来进行损失
        # 这儿只能传递初始值，因为，拓扑损失里面会做归一化 tensor(0.1028, device='cuda:0', grad_fn=<MeanBackward0>)现在这是对的损失
        # topo_loss tensor(0.1034, device='cuda:0', grad_fn=<MeanBackward0>)  这是最合理的改进版了
        cls_loss = self.calculate_classification_loss(cls, min_idcs, mask, dist, min_dist)  # dgf
        reg_loss = self.calculate_regression_loss(reg, min_idcs, gt_preds, has_preds)  # dgf
        # print("reg_loss",reg_loss) # reg_loss tensor(2.2638, device='cuda:0', grad_fn=<DivBackward0>)
        reg_loss_final = self.calculate_regression_loss(reg_final[..., 0:2].clone(), min_idcs, gt_preds, has_preds)  # dgf
        # print("reg_loss_final",reg_loss_final) # reg_loss_final tensor(2.2179, device='cuda:0', grad_fn=<DivBackward0>)

        loss_out["cls_loss"] = self.config["cls_coef"] * cls_loss
        loss_out["reg_loss"] = self.config["reg_coef"] * reg_loss
        loss_out["reg_loss_final"] = self.config["reg_coef_final"] * reg_loss_final  # dgf

        # if epoch < 20:
        #     loss_out["cls_loss"] = 0.0 * loss_out["cls_loss"]
        #     loss_out["reg_loss"] = 0.0 * loss_out["reg_loss"]
        #     loss_out["reg_loss_final"] = 0.0 * loss_out["reg_loss_final"]

        loss_out["topo_loss"] = 10 * topo_loss # 师兄给的10?  我设的0.2效果比之差很小

        return loss_out

    def create_mask(self, has_preds, num_preds):  # dgf
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds) # (56,30)
        # print("last",last) # tensor([[1.0000, 1.0033, 1.0067,  ..., 1.0900, 1.0933, 1.0967],
        max_last, last_idcs = last.max(1) # (56,) (56,)
        # print("max_last",max_last) # tensor([1.0967, 1.0967, 1.0967, 1.0500, 1.0967, 1.0700, 1.0533, 1.0433, 1.0700,
        # print("last_idcs",last_idcs) # tensor([29, 29, 29, 15, 29, 21, 16, 13, 21, 29, 29,  6, 22, 27, 29, 29, 29, 29,
        mask = max_last > 1.0 # (56,)
        # print("mask",mask) # tensor([True, True, True, True, True, True, True, True, True, True, True, True,
        return mask, last_idcs

    def get_min_distance_indices(self, reg, gt_preds, last_idcs, num_modes):  # dgf
        row_idcs = torch.arange(len(last_idcs)).long().to(self.device) # (56,)
        # print("row_idcs",row_idcs) # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        dist = [torch.sqrt(((reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs]) ** 2).sum(1)) for j in range(num_modes)] # list:6 这是总共有6个模态
        # print("dist",dist) # [tensor([ 4.2949, 31.3454, 23.8799, 22.0468,  1.6730,  0.7869,  1.5256,  1.5218,
        dist = torch.stack(dist, dim=1) # (56,6)
        # print("dist",dist) # tensor([[4.2949e+00, 3.8511e+00, 5.2269e+00, 5.2230e+00, 4.1749e+00, 3.9620e+00], 按照每个张量的第一个元素进行排列
        min_dist, min_idcs = dist.min(1) # (56,) (56,)
        # print("min_dist",min_dist) # tensor([3.8511e+00, 3.1090e+01, 2.3612e+01, 2.2034e+01, 1.2781e+00, 7.7322e-01, 找出56个在6个模态中距离最小的
        # print("min_idcs",min_idcs) # 最小的对应的位置
        return dist, min_dist, min_idcs
    
    def calculate_classification_loss(self, cls, min_idcs, mask, dist, min_dist):  # dgf
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device) # (56,)
        # print("row_idcs",row_idcs) # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls # (56,6)
        # print("mgn",mgn) # tensor([[-3.3426e-03,  0.0000e+00, -7.6143e-03, -6.9099e-03, -4.5182e-03,
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1) # (56,1)
        # print("mask0",mask0) # tensor([[False],
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"] # # (56,6)
        # print("mask1",mask1) # tensor([[ True, False,  True,  True,  True, False],
        mgn = mgn[mask0 * mask1] # (130,)
        # print("mgn",mgn) # tensor([-3.9722e-03, -2.4380e-03, -3.4943e-03, -9.7004e-03, -4.0440e-04,
        mask = mgn < self.config["mgn"] # # (130,)
        # print("mask",mask) # tensor([True, True, True, True, True, True, True, True, True, True, True, True,
        num_cls = mask.sum().item() # 130

        cls_loss = (self.config["mgn"] * mask.sum() - mgn[mask].sum()) / (num_cls + 1e-10)
        # print("cls_loss",cls_loss) # cls_loss tensor(0.1985, device='cuda:0', grad_fn=<DivBackward0>)
        return cls_loss

    def calculate_regression_loss(self, reg, min_idcs, gt_preds, has_preds):  # dgf
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)
        reg = reg[row_idcs, min_idcs]
        num_reg = has_preds.sum().item()
        reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) / (num_reg + 1e-10)
        return reg_loss

    def focal_loss(self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "none",
    ) -> torch.Tensor:
        """
        Original implementation from
        https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
        p = torch.sigmoid(inputs) # (2,42,42,1)
        # print("p",p) # tensor([[[[0.4687],
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # (2,42,42,1)
        # print("ce_loss",ce_loss) # tensor([[[[0.6325],
        p_t = p * targets + (1 - p) * (1 - targets) # (2,42,42,1)
        # print("p_t",p_t) # tensor([[[[0.5313],
        loss = ce_loss * ((1 - p_t) ** gamma) # (2,42,42,1)
        # print("loss1",loss) # tensor([[[[0.1390],

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets) # (2,42,42,1)
            # print("alpha_t",alpha_t) # tensor([[[[0.7500],
            loss = alpha_t * loss # (2,42,42,1)
            # print("loss2",loss) # tensor([[[[0.1042],

        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def topo_loss(self,
            prediction, targets, valid_mask,
            top_k=False, top_k_ratio=1.):
        """
        build the top-k CE loss for BeTop reasoning
        preds, targets, valid_mask: [b, src, tgt]
        return: loss [b]
        """
        b, s, t, step = prediction.shape # 2 42 42 1
        targets = targets.float()

        loss = self.focal_loss(
            prediction,
            targets,
            reduction='none',
        )
        # print("loss3",loss) # tensor([[[[0.1042],

        loss = loss * valid_mask[..., None] # (2,42,42,1)  loss4已经做了掩码了
        # print("loss4",loss) # tensor([[[[0.1042],
        # loss = loss.view(b, s * t, step) # (2,1764,1)  相当于对半对折了  这一步就有问题了  我需要只每个批次的提取第一块 braids_gt = braids_gt[:,0,:,:]
        loss = loss[:,0,:,:] # 0421
        # print("loss5",loss) # tensor([[[0.1042],
        # valid_mask = valid_mask.reshape(b, s * t) # (2,1764)
        valid_mask = valid_mask[:,0,:] # 0421
        # print("valid_mask",valid_mask) # tensor([[1., 1., 1.,  ..., 0., 0., 0.],

        mask = torch.sum(valid_mask, dim=-1) # (2,) 沿着列方向求和
        # print("mask",mask) # tensor([ 168., 1386.], device='cuda:0')
        mask = mask + (mask == 0).float() # tensor([ 168., 1386.], device='cuda:0')
        # print("mask",mask) # tensor([ 168., 1386.], device='cuda:0')

        return (torch.sum(loss.mean(-1), dim=1) / mask).mean()

