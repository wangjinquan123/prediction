import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

class TrajPredictionEvaluator():
    ''' Return evaluation results for batched data '''

    def __init__(self, config, save_dir=None):
        super(TrajPredictionEvaluator, self).__init__()
        self.config = config

    def evaluate(self, post_out, data):
        traj_pred = post_out['traj_pred']
        prob_pred = post_out['prob_pred']
        # traj_pred:    batch x n_mod x pred_len x 2
        # prob_pred:    batch x n_mod

        # braids = post_out['braids']
        # braids_gt = data['braids']
        braids = post_out['topo_pred'] # 现在概率值
        braids_gt = post_out['braids']
        braids_gt = braids_gt[:,0,:,:] # 0420
        # print("braids_gt",braids_gt)

        # topo_pred = braids.squeeze(-1)  # (2,14,14) 0407 不能少
        # braids_probs = F.softmax(topo_pred_, dim=-1)  # (2,14,14) 0407
        braids = torch.sigmoid(braids)
        braids_probs = braids[:,0,:,:] # 0420
        # print("braids_probs", braids_probs) # tensor([[[0.0239, 0.0239, 0.0238,  ..., 0.0238, 0.0238, 0.0238],
        # tensor([[[0.4687, 0.4687, 0.4677,  ..., 0.4683, 0.4683, 0.4683],

        # 确保数据是CPU上的numpy数组
        # braids_probs_ = braids_probs.unsqueeze(-1)
        # print("braids_probs_1", braids_probs_)
        # braids_probs_ = braids_probs_.cpu().detach().numpy().flatten() # 展平为1D 0407  还是不太能理解他的这个意义何在
        # print("braids_probs_2", braids_probs_)
        # braids_gt_ = braids_gt.cpu().numpy().flatten().astype(int) # 0407

        eval_out = {}
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thr in thresholds:
            # braids_pred = braids > thr
            braids_pred = braids_probs > thr # 0407
            # print("braids_pred", braids_pred.squeeze(-1))
            # braids_pred = braids_pred.unsqueeze(-1) # 0407 0418
            # print("braids_pred", braids_pred)
            tp = torch.sum(braids_pred & braids_gt).float() #  预测正确 真实为Braids，模型也预测为Braids
            # print("tp",tp)
            fp = torch.sum(braids_pred & ~braids_gt).float() #   错预测 真实不是Braids，模型错误预测为Braids  这个应该很大
            # print("fp",fp)
            fn = torch.sum(~braids_pred & braids_gt).float() #   漏预测 真实是Braids，模型错误预测为非Braids
            # print("fn",fn)

            precision = tp / (tp + fp + 1e-8) # 10个车，6个为braids，总共将8个判断为True，5个预测正确，1个漏预测，3个错预测，precision=5/(5+3)=0.625    6个label
            # print("precision",precision)
            recall = tp / (tp + fn + 1e-8)  # 10个车，6个为braids,总共将8个判断为True，5个预测正确，1个漏预测，3个错预测，recall=5/(5+1)=0.83
            # print("recall",recall)

            F1 = 2 * precision * recall / (precision + recall + 1e-8) # 0407 这个用于判断topo_pred_mask = topo_pred_mask > 0.5  中取合适的阈值
            # print("F1",F1)


            thr_str = f"{int(thr * 10):02d}"
            eval_out[f'braids_precision_{thr_str}'] = precision.item()
            eval_out[f'braids_recall_{thr_str}'] = recall.item()

            eval_out[f'braids_f1_{thr_str}'] = F1.item() # 0407

        # # y_true: 真实标签（0或1）
        # # y_score: 模型预测的概率值（非二值化结果）
        # # 检查标签类别数量
        # if len(np.unique(braids_gt_)) >= 2: # 0407
        #     auc = roc_auc_score(braids_gt_, braids_probs_)
        #     eval_out['braids_auc'] = auc
        # else:
        #     eval_out['braids_auc'] = float('nan') # 单类情况处理（例如跳过或设为默认值）

        if self.config['data_ver'] == 'av1':
            # for av1
            traj_fut = torch.stack([traj[0, :, 0:2] for traj in data['TRAJS_FUT']])  # batch x fut x 2
        elif self.config['data_ver'] == 'av2':
            # for av2
            traj_fut = torch.stack([x['TRAJS_POS_FUT'][0] for x in data["TRAJS"]])  # batch x fut x 2
        else:
            assert False, 'Unknown data_ver: {}'.format(self.config['data_ver'])

        # to np.ndarray
        traj_pred = np.asarray(traj_pred.cpu().detach().numpy()[:, :, :, :2], np.float32)
        prob_pred = np.asarray(prob_pred.cpu().detach().numpy(), np.float32)
        traj_fut = np.asarray(traj_fut.numpy(), np.float32)

        seq_id_batch = data['SEQ_ID']
        batch_size = len(seq_id_batch)

        pred_dict = {}
        gt_dict = {}
        prob_dict = {}
        for j in range(batch_size):
            seq_id = seq_id_batch[j]
            pred_dict[seq_id] = traj_pred[j]
            gt_dict[seq_id] = traj_fut[j]
            prob_dict[seq_id] = prob_pred[j]

        # # Max #guesses (K): 1
        res_1 = get_displacement_errors_and_miss_rate(
            pred_dict, gt_dict, 1, self.config['g_pred_len'], miss_threshold=self.config['miss_thres'], forecasted_probabilities=prob_dict)
        # # Max #guesses (K): 6
        res_k = get_displacement_errors_and_miss_rate(
            pred_dict, gt_dict, 6, self.config['g_pred_len'], miss_threshold=self.config['miss_thres'], forecasted_probabilities=prob_dict)


        eval_out['minade_1'] = res_1['minADE']
        eval_out['minfde_1'] = res_1['minFDE']
        eval_out['mr_1'] = res_1['MR']
        eval_out['brier_fde_1'] = res_1['brier-minFDE']

        eval_out['minade_k'] = res_k['minADE']
        eval_out['minfde_k'] = res_k['minFDE']
        eval_out['mr_k'] = res_k['MR']
        eval_out['brier_fde_k'] = res_k['brier-minFDE']

        return eval_out
