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
        topo_pred, braids, actor_topo_mask = out_sc[3], out_sc[4], out_sc[5] # (1,14,14,1) (1,14,14,1,) (1,1,14)
        topo_pred = topo_pred.squeeze(-1)
        topo_pred = F.softmax(topo_pred, dim=-1)
        topo_pred = topo_pred.unsqueeze(-1) # 0410 0418
        gt_preds = torch.cat(gt_preds, 0)  # dgf
        has_preds = torch.cat(pad_flags, 0).bool()  # dgf

        N = actor_topo_mask.shape[2]
        actor_topo_mask = actor_topo_mask.expand(-1, N, -1)  # 使用广播机制扩展

        loss_out = dict()
        num_modes = self.config["g_num_modes"]
        num_preds = 30

        mask, last_idcs = self.create_mask(has_preds, num_preds)  # dgf
        cls, reg, reg_final, gt_preds, has_preds, last_idcs = map(lambda x: x[mask], [cls, reg, reg_final, gt_preds, has_preds, last_idcs])  # dgf
        # print("cls",cls)

        dist, min_dist, min_idcs = self.get_min_distance_indices(reg[..., 0:2].clone(), gt_preds, last_idcs, num_modes)  # dgf

        topo_loss = self.topo_loss(topo_pred, braids.detach(), actor_topo_mask.float().detach()) # (2,) 归一化之后的topo_pred_ 0410 braids.detach(), actor_topo_mask.float().detach()
        # print("topo_loss",topo_loss) # topo_loss tensor(0.1350, device='cuda:0', grad_fn=<MeanBackward0>)
        # 是归一化之后的拓扑预测还是需要传给模型的拓扑预测  根据reg是模型得出的结果来进行损失

        cls_loss = self.calculate_classification_loss(cls, min_idcs, mask, dist, min_dist)  # dgf
        # print("cls_loss",cls_loss) # cls_loss tensor(0.1985, device='cuda:0', grad_fn=<DivBackward0>)
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
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0
        return mask, last_idcs

    def get_min_distance_indices(self, reg, gt_preds, last_idcs, num_modes):  # dgf
        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)
        dist = [torch.sqrt(((reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs]) ** 2).sum(1)) for j in range(num_modes)]
        dist = torch.stack(dist, dim=1)
        min_dist, min_idcs = dist.min(1)
        return dist, min_dist, min_idcs
    
    def calculate_classification_loss(self, cls, min_idcs, mask, dist, min_dist):  # dgf
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        num_cls = mask.sum().item()
        cls_loss = (self.config["mgn"] * mask.sum() - mgn[mask].sum()) / (num_cls + 1e-10)
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
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

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
        b, s, t, step = prediction.shape
        targets = targets.float()

        loss = self.focal_loss(
            prediction,
            targets,
            reduction='none',
        )

        loss = loss * valid_mask[..., None]
        loss = loss.view(b, s * t, step)
        valid_mask = valid_mask.reshape(b, s * t)

        mask = torch.sum(valid_mask, dim=-1)
        mask = mask + (mask == 0).float()

        return (torch.sum(loss.mean(-1), dim=1) / mask).mean()

