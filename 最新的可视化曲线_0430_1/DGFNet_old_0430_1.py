from typing import Dict, List, Tuple, Optional
import math
from fractions import gcd

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

from utils.utils import gpu, init_weights
from utils.topo_utils import generate_behavior_braids
from DGFNet.av1_loss_fn import LossFunc
import copy
from DGFNet.topo_decoder import TopoFuser,TopoDecoder

import os
from datetime import datetime


def create_batched_combination_trajs(src_trajs, tgt_trajs): # 这一步的意义？
    """
    src trajs: [b, 1, T, d]
    return [b, a, a, 2, T, d]
    """
    b, a, t, d = src_trajs.shape
    blank_traj = torch.zeros_like(src_trajs)
    blank_tgt_traj = torch.zeros_like(tgt_trajs)
    src = torch.stack([src_trajs, blank_traj], dim=2)[:, :, None, :, :]
    tgt = torch.stack([blank_tgt_traj, tgt_trajs], dim=2)[:, None, :, :, :]
    res = src + tgt
    res = res[:,:,:,:,-1,0:2]
    return res # b,a,a,2,2


class DGFNet(nn.Module):
    # Initialization
    def __init__(self, cfg, device):
        super(DGFNet, self).__init__()
        self.device = device
        
        self.g_num_modes = 6
        self.g_pred_len = 30
        self.g_num_coo = 2
        self.d_embed = 128
        self.decoder_layer_topo = 2

        self.fuse_layer = TopoFuser(device=self.device,
                                    input_dim=cfg['d_embed'],
                                    dim=cfg['d_embed']) # // 需要设定
        self.decoder_layer = TopoDecoder(device=self.device,
                                        dim=cfg['d_embed']) # // 需要设定

        self.actor_net_ac = Actor_Encoder(device=self.device,
                                  n_in=cfg['in_actor'], # 3
                                  hidden_size=cfg['d_actor'], # 128
                                  n_fpn_scale=cfg['n_fpn_scale'])

        self.actor_net_sc = Actor_Encoder(device=self.device,
                                  n_in=cfg['in_actor'],
                                  hidden_size=cfg['d_actor'],
                                  n_fpn_scale=cfg['n_fpn_scale'])

        self.lane_net_ac = Map_Encoder(device=self.device,
                                in_size=cfg['in_lane'], # 10
                                hidden_size=cfg['d_lane'], # 128
                                dropout=cfg['dropout'])

        self.lane_net_sc = Map_Encoder(device=self.device,
                                in_size=cfg['in_lane_sc'], # 8
                                hidden_size=cfg['d_lane'],
                                dropout=cfg['dropout'])

        self.ac_am_fusion = FusionNet(device=self.device,
                                    config=cfg)


        self.interaction_module_sc_am = Interaction_Module_SC_AM(device=self.device,
                                                               hidden_size = cfg['d_lane'],
                                                               depth = cfg["n_interact"])

        self.interaction_module_af = Interaction_Module_FE(device=self.device,   # aa
                                                         hidden_size = cfg['d_lane'],
                                                         depth = cfg["n_interact"])

        self.interaction_module_al = Interaction_Module_FE(device=self.device,  #  al
                                                         hidden_size = cfg['d_lane'],
                                                         depth = cfg["n_interact"])

        self.trajectory_decoder_fe = Trajectory_Decoder_Future_Enhanced(device=self.device,
                                                                       hidden_size = cfg['d_lane']) # 128

        self.trajectory_decoder_final = Trajectory_Decoder_Final(device=self.device,
                                                                hidden_size = cfg["d_decoder_F"]) # 256  hidden_size=cfg["d_decoder_F"] + 2  # 增加残差特征

        self.rft_encoder = Reliable_Future_Trajectory_Encoder()

        # self.build_topo_layers = Build_Top_Layer(device=self.device,  # top
        #     d_model = cfg["D_MODEL"],
        #     map_d_model=cfg["MAP_D_MODEL"],
        #     actor_d_model=cfg["ACTOR_D_MODEL"],
        #     dropout=cfg['dropout'],
        #     num_decoder_layers=cfg["NUM_DECODER_LAYERS"])

        if cfg["init_weights"]:
            self.apply(init_weights)

    def apply_topo_reasoning(  # 17 ## 0503
        self,
        query_feat, kv_feat,
        fuse_layer,
        decoder_layer,
        ):
        """
        performing synergistic Topology reasoning
        Args:
            query_feat, kv_feat  [M, B, D], [B, N, D] M：模态数（如预测的多个可能轨迹）
            prev_topo_feat, [B, M, N, D]
            fuse_layer, decoder layer: Topo decoders
            center_gt_positive_idx / full_preds:
            Efficient decoding for train-time reasoning
        """
        # query_feat = query_feat.permute(1, 0, 2) # 0420  原版这儿需要permute
        b = query_feat.shape[0]

        src = query_feat
        tgt = kv_feat

        topo_feat = fuse_layer(src, tgt,
                               # prev_topo_feat
                               ) # 变成[B,1,N,D]  最关键的，不用管prev_topo_feat src和tgt可以输入成一样，
        topo_pred = decoder_layer(topo_feat) # 变成[B,1,N,1]  这个输出出来就是一个N×N的概率  将这个做loss 这个可以预测出topo_brabids  跟他车交互的概率

        return topo_pred  # 所以这儿的拓扑预测是single_topo_pred
    # [B,M,N,D] [B,1,N,1] [B,M,N,1]

    def forward(self, data, data_ori): # tuple:7
        actors_ac, actor_idcs, lanes_ac, lane_idcs, rpe,  actors_sc, lanes_sc, trajs_obs, actors_future, actor_future_mask = data # simpl
        # print(type(actors_future))
        # print(len(actors_future))
        # print(type(actors_future[0]))
        # print(actors_future[0])
        print("rpe1:",rpe) # list:2 rpe [{'scene': tensor([[[ 1.0000e+00,  2.2155e-01,  4.3499e-01,  ...,  4.5422e-01,

        # print(type(actor_future_mask))
        # print(actor_future_mask)

        # actors_future = actors_future[0].to(self.device) ##
        # actors_future_mask = torch.stack(actor_future_mask, dim=0).to(self.device) ## ##

        agent_lengths = []  # 15 adapt
        for actor_id in actor_idcs:  # (15,) adapt
            agent_lengths.append(actor_id.shape[0] if actor_id is not None else 0)

        lane_lengths = []  # 197
        for lane_id in lane_idcs:  # (197,)
            lane_lengths.append(lane_id.shape[0] if lane_id is not None else 0)

        # print(actors_ac[0])
        batch_size = len(actor_idcs)  # 1
        # 计算每个样本的 Agent 数量及最大值
        agent_lengths = [len(actor_id) if actor_id is not None else 0 for actor_id in actor_idcs]
        max_agent_num = max(agent_lengths)  # 15
        max_lane_num = max(lane_lengths)  # 197

        # ac actors/lanes encoding
        actors_ac = self.actor_net_ac(actors_ac) # (15,128) simpl
        lanes_ac = self.lane_net_ac(lanes_ac) # (197,128) simpl

        # ac feature fusion
        actors_ac, _ , _ = self.ac_am_fusion(actors_ac, actor_idcs, lanes_ac, lane_idcs, rpe) # (15,128) simpl
        # print("actors_ac:",actors_ac)

        # sc actors/lanes encoding
        actors_sc = self.actor_net_sc(actors_sc) # (15,128)
        lanes_sc = self.lane_net_sc(lanes_sc) # (197,128)

        # actors_aa,_ ,_ = self.ac_am_fusion(actors_sc, actor_idcs, lanes_sc, lane_idcs, rpe) # (15,128) 0405 如果用actor_ac没用的话，再试试这个
        # print("actors_aa:",actors_aa)

        actors_batch_sc = torch.zeros(batch_size, max_agent_num, self.d_embed, device=self.device) # (2,42,128)
        actors_batch_ac = torch.zeros(batch_size, max_agent_num, self.d_embed, device=self.device) # (2,42,128)
        lanes_batch_sc = torch.zeros(batch_size, max_lane_num, self.d_embed, device=self.device) # (1,197,128)

        # actors_batch_aa = torch.zeros(batch_size, max_agent_num, self.d_embed, device=self.device) # (1,15,128) 0406

        # 修复1：正确填充逻辑
        padded_actors_future = torch.zeros(batch_size, max_agent_num, 30, 2, device=self.device)
        for i in range(batch_size):
            original_agents = actors_future[i].to(self.device)  # [原始agent数, 30, 2]
            num_agents = original_agents.shape[0]
            padded_actors_future[i, :num_agents] = original_agents

        padded_actors_future_mask = torch.zeros(batch_size, max_agent_num, 30, device=self.device)
        for i in range(batch_size):
            original_mask = actor_future_mask[i].to(self.device)  # [原始agent数, 30]
            num_agents = original_mask.shape[0]
            padded_actors_future_mask[i, :num_agents] = original_mask

        actors_batch_fu = torch.zeros(batch_size, max_agent_num, 30, 2,device=self.device)  # 扩展目标形状
        actors_batch_history = torch.zeros(batch_size, max_agent_num, 20, 2, device=self.device)
        actors_batch_fu_mask = torch.zeros(batch_size, max_agent_num, 30, device=self.device)

        for i, actor_ids in enumerate(actor_idcs):
            # num_agents = actor_ids.shape[0] # 15 ##

            num_agents = agent_lengths[i]  # 使用预先计算的长度

            actors_batch_sc[i, :num_agents] = actors_sc[actor_ids[0] : actor_ids[-1] + 1]
            actors_batch_ac[i, :num_agents] = actors_ac[actor_ids[0] : actor_ids[-1] + 1]

            # actors_batch_aa[i, :num_agents] = actors_aa[actor_ids[0] : actor_ids[-1] + 1] # 0406

            pads_obs = data_ori['PAD_OBS'][i].to(self.device) # [14,20] # [实际agent数, 20]
            pads_fut = data_ori['PAD_FUT'][i].to(self.device) # [实际agent数, 30]

            # 获取当前样本的轨迹数据，假设trajs_obs是按样本组织的列表
            sample_trajs = trajs_obs[i]  # 形状应为 [num_agents_in_sample, 20, 2]
            # 填充到max_agent_num
            if num_agents > 0:
                actors_batch_history[i, :num_agents] = sample_trajs[:num_agents]

            # 修复3：使用正确的索引方式
            actors_batch_fu[i, :num_agents] = padded_actors_future[i, :num_agents]  # 直接使用填充后数据
            actors_batch_fu_mask[i, :num_agents] = padded_actors_future_mask[i, :num_agents]

            # for i in range(batch_size):
            #     # 获取自车未来轨迹
            #     ego_future = actors_batch_fu[i, 0]  # (30, 2)
            #     orig_ego = ego_future[0].clone()  # 原点：第一个点
            #
            #     # 计算方向向量（未来轨迹前两个点）
            #     if ego_future.shape[0] >= 2:
            #         vec_ego = ego_future[1] - ego_future[0]
            #     else:
            #         vec_ego = torch.tensor([1.0, 0.0], device=self.device)
            #
            #     # 计算旋转矩阵（负角度旋转）
            #     theta = torch.atan2(vec_ego[1], vec_ego[0])
            #     cos_theta = torch.cos(theta)  # 注意：直接使用theta，而非 -theta
            #     sin_theta = torch.sin(theta)
            #     rot_matrix = torch.tensor(
            #         [[cos_theta, sin_theta],
            #          [sin_theta, -cos_theta]],  # 修正矩阵结构
            #         device=self.device
            #     )
            #
            #     # 对所有代理的轨迹进行变换
            #     for j in range(max_agent_num):
            #         traj_global = actors_batch_fu[i, j]  # (30, 2)
            #
            #         # 平移
            #         traj_local = traj_global - orig_ego
            #
            #         # 旋转（注意矩阵转置）
            #         traj_transformed = torch.mm(traj_local, rot_matrix)
            #
            #         actors_batch_fu[i, j] = traj_transformed


            # 修复4：保持维度一致性
            valid_mask = (pads_obs.sum(-1) >= 15) & (pads_fut.sum(-1) >= 30)
            valid_mask = valid_mask.float().unsqueeze(-1)  # [实际agent数, 1]

            # actors_batch_fu[i, :num_agents] = actors_future[actor_ids[0] : actor_ids[-1] + 1] ##
            # actors_batch_history[i, :num_agents] = trajs_obs[actor_ids[0]: actor_ids[-1] + 1] # b,a,20,2 ## ##
            # actors_batch_history[i, :num_agents] = torch.stack(trajs_obs[actor_ids[0]: actor_ids[-1] + 1], dim=0) ##
            # actors_batch_fu_mask[i, :num_agents] = actors_future_mask[actor_ids[0] : actor_ids[-1] + 1]* valid_mask.float() ## ##

            actors_batch_fu_mask[i, :num_agents] *= valid_mask
        # print(f'\n\nSequence ID: {data_ori["SEQ_ID"][0]}')
        # print("src:", actors_batch_fu)

        history_matrics = create_batched_combination_trajs(actors_batch_history,actors_batch_history) # 1,a,a,2,2 (1,14,14,2,2)
        braids = generate_behavior_braids(actors_batch_fu, actors_batch_fu, actors_batch_fu_mask, actors_batch_fu_mask,1)  # (2,42,42,1)
        # print("braids:",braids)
        actor_topo_mask = torch.any(actors_batch_fu_mask, dim=-1)[:, None, :] # (1,1,14)

        # data_org=data_org
        # vis.draw_once_new(self, data=data_org, braids=braids, history_matrics=history_matrics) # 本身post_out:dict:3 data:dict:21
        for i, lane_ids in enumerate(lane_idcs):
            num_lanes = lane_ids.shape[0] # 197
            lanes_batch_sc[i, :num_lanes] = lanes_sc[lane_ids[0] : lane_ids[-1] + 1] # lanes_batch_sc:维度？

        topo_pred = self.apply_topo_reasoning(actors_batch_ac, actors_batch_ac, self.fuse_layer, self.decoder_layer) # (2,42,42,1) 以为直接输出的是概率  这个拓扑预测还需要斟酌
        # print("topo_pred:",topo_pred.squeeze(-1)) #  tensor([[[-0.1253, -0.1253, -0.1294,  ..., -0.1269, -0.1269, -0.1269],

        topo_pred_ = torch.sigmoid(topo_pred) # 0419
        # print("topo_pred",topo_pred.squeeze(-1)) # tensor([[[0.4687, 0.4687, 0.4677,  ..., 0.4683, 0.4683, 0.4683],

        # topo_pred_mask = topo_pred.squeeze(-1) # (2,14,14) 0407 不能少
        # topo_pred_mask = F.softmax(topo_pred_mask, dim=-1) # (2,14,14) 0407 sigmoid
        # topo_pred_ = topo_pred_mask.unsqueeze(-1) # 0417 0418
        # print("topo_pred_mask:", topo_pred_mask) #  tensor([[[0.0239, 0.0239, 0.0238,  ..., 0.0238, 0.0238, 0.0238],
        topo_pred_mask = topo_pred_ > 0.4 # 0407 0417_2 概率的判断必须要先归一化 # 178 需根据F1.score来判断阈值  具体取多少阈值仍需商榷
        topo_pred_mask = topo_pred_mask.squeeze(-1) # 0417 0418 0419
        # print("topo_pred_mask:", topo_pred_mask) # (2,42,42) tensor([[[False, False, False,  ..., False, False, False],

        masks, _ = get_masks(agent_lengths, lane_lengths,  self.device) # list:4 01
        # print("mask",masks)
        masks[-4] = topo_pred_mask.float()  # //0407 新加 (1,14,14,1) (2,42,42) .float()
        # print("mask[-4]",masks)
        agent_states, lane_states = self.interaction_module_sc_am(actors_batch_sc, lanes_batch_sc, masks)  # four (1,15,128) (1,197,128) 02 lane_states无用

        #reliable future trajectory generate
        predictions_fe = self.trajectory_decoder_fe(agent_states)   # (1,15,6,30,2) 特征增强 03 三个MLP

        final_positions = predictions_fe[:, :, :, -1, :] # (1,15,6,2) 是把倒数第二列截掉
        mean_final_positions = final_positions.mean(dim=2) # (1,15,2) 是在第三列上截掉
        deviations = torch.sqrt(((final_positions - mean_final_positions.unsqueeze(2)) ** 2).sum(dim=-1)) # (1,15,6)
        mean_deviation = deviations.mean(dim=-1) # (1,15)

        mask = (mean_deviation <= 5).float() # (1,15)
        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (1,15,1,1,1)
        masked_predictions = predictions_fe * mask # (1,15,6,30,2)

        # print("masked_predictions_1:",masked_predictions[0][5][0])
        # print("masked_predictions_2:", masked_predictions[0][7][0])

        future_states = self.rft_encoder(masked_predictions)  # Reliable Future Trajectory (1,15,128)

        # 有困难的再做一次local attention
        #future feature interaction
        masks_af = get_mask(agent_lengths, agent_lengths, self.device) # list:1  AA 这个判断合不合理，futre_states全程应该都是在场景为中心下的
        # masks_af[0] = topo_pred_mask.int()
        actors_batch_af, _ = self.interaction_module_af(actors_batch_ac, future_states, masks_af)  # (1,15,128)

        masks_al = get_mask(agent_lengths, lane_lengths, self.device) # list:1 AL
        actors_batch_al, _ = self.interaction_module_al(actors_batch_af, lanes_batch_sc, masks_al)  # (1,15,128) 场景级别和自车级别坐标系做交互？

        agent_fusion = torch.cat((actors_batch_al, agent_states), dim = 2) # (1,15,256)  只在第三列上相加

        predictions_final, logits_final = self.trajectory_decoder_final(agent_fusion) # (1,15,6,30,2) (1,15,6)  重新再走一遍self.trajectory_decoder_fe

        expan_predictions_fe = torch.zeros((actors_sc.shape[0], self.g_num_modes, self.g_pred_len, self.g_num_coo), device=self.device) # (15,6,30,2)
        expan_predictions_final = torch.zeros((actors_sc.shape[0], self.g_num_modes, self.g_pred_len, self.g_num_coo), device=self.device) # (15,6,30,2)
        expan_logits_final = torch.zeros((actors_sc.shape[0], self.g_num_modes), device=self.device) # (15,6)
        for i, actor_ids in enumerate(actor_idcs):
            num_agents = actor_ids.shape[0] # 15
            expan_predictions_fe[actor_ids[0]:actor_ids[-1] + 1] = predictions_fe[i, :num_agents]
            expan_predictions_final[actor_ids[0]:actor_ids[-1] + 1] = predictions_final[i, :num_agents]
            expan_logits_final[actor_ids[0]:actor_ids[-1] + 1] = logits_final[i, :num_agents]

        res_reg_fe, res_reg_final , res_cls_final = [], [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            res_reg_fe.append(expan_predictions_fe[idcs])
            res_reg_final.append(expan_predictions_final[idcs])
            res_cls_final.append(expan_logits_final[idcs])
        
        out_sc = [res_cls_final, res_reg_final, res_reg_fe, topo_pred, braids, actor_topo_mask, history_matrics ] # list:3 topo_pred传回的是初始值
        # 看betopNet怎么预测label的

        return out_sc # list:3

    def pre_process(self, data): # dict:21
        '''
            Send to device
            'BATCH_SIZE', 'SEQ_ID', 'CITY_NAME',
            'ORIG', 'ROT',
            'TRAJS_OBS', 'TRAJS_FUT', 'PAD_OBS', 'PAD_FUT', 'TRAJS_CTRS', 'TRAJS_VECS',
            'LANE_GRAPH',
            'RPE',
            'ACTORS', 'ACTOR_IDCS', 'LANES', 'LANE_IDCS','LANES_SC','ACTORS_SC'
        '''

        actors = gpu(data['ACTORS'], self.device) # (15,3,20)
        actors_sc = gpu(data['ACTORS_SC'], self.device) # (15,3,20)
        actor_idcs = gpu(data['ACTOR_IDCS'], self.device) # list:1
        lanes = gpu(data['LANES'], self.device) # (197,10,10)
        lanes_sc = gpu(data['LANES_SC'], self.device) # (197,10,8)
        lane_idcs = gpu(data['LANE_IDCS'], self.device)  # list:1
        rpe = gpu(data['RPE'], self.device)  # list:1

        # 新增加载历史轨迹（观测轨迹）
        trajs_obs = gpu(data['TRAJS_OBS_ORI'], self.device)
        actor_future = gpu(data['TRAJS_FUT_ORI'], self.device) # 列表
        actor_future_mask = gpu(data['PAD_FUT'], self.device)

        return actors, actor_idcs, lanes, lane_idcs, rpe , actors_sc, lanes_sc, trajs_obs, actor_future, actor_future_mask


    def post_process(self, out): # list:3
        post_out = dict()
        res_cls = out[0]
        res_reg = out[1]

        topo_pred = out[3]
        braids = out[4]
        history_matrics = out[6]

        # get prediction results for target vehicles only
        reg = torch.stack([trajs[0] for trajs in res_reg], dim=0) # (1,6,30,2)
        cls = torch.stack([probs[0] for probs in res_cls], dim=0) # (1,6)

        post_out['out_sc'] = out
        post_out['traj_pred'] = reg  # batch x n_mod x pred_len x 2
        post_out['prob_pred'] = cls  # batch x n_mod

        post_out['topo_pred'] = topo_pred
        post_out['braids'] = braids
        post_out['history_matrics'] = history_matrics

        return post_out


class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(
            int(kernel_size) - 1) // 2, stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):  # (15,128,5)  (15,64,10) (15,32,20)   (15,128,5) (15,64,10) (15,32,20)
        out = self.conv(x)  # (15,128,5) (15,128,10) (15,128,20)   (15,128,5) (15,128,10) (15,128,20)
        out = self.norm(out) # (15,128,5) (15,128,10) (15,128,20)   (15,128,5) (15,128,10) (15,128,20)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):    # (15,3,20) (15,32,20) (15,32,20)  (15,64,10) (15,64,10) (15,128,5) (15,128,20)   (15,3,20) (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,20)
        out = self.conv1(x)  # (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)   (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)
        out = self.bn1(out)  # (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)   (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)
        out = self.relu(out)  # (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)   (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)
        out = self.conv2(out)  # (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)   (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)
        out = self.bn2(out)  # (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)   (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)

        if self.downsample is not None:
            x = self.downsample(x) # (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5)  _   (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) _ (15,128,20)

        out += x  # (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)   (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)
        if self.act:
            out = self.relu(out)  # (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)   (15,32,20) (15,32,20) (15,64,10) (15,64,10) (15,128,5) (15,128,5) (15,128,20)
        return out


class Actor_Encoder(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, device, n_in=3, hidden_size=128, n_fpn_scale=4):
        super(Actor_Encoder, self).__init__()
        self.device = device
        norm = "GN"
        ng = 1

        n_out = [2**(5 + s) for s in range(n_fpn_scale)]  # [32, 64, 128]
        blocks = [Res1d] * n_fpn_scale
        num_blocks = [2] * n_fpn_scale

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], hidden_size, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(hidden_size, hidden_size, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor: # (15,3,20)   (15,3,20)
        out = actors # (15,3,20)   (15,3,20)

        outputs = [] # list:3
        for i in range(len(self.groups)):
            out = self.groups[i](out) # (15,32,20) (15,64,10) (15,128,5)   (15,32,20) (15,64,10) (15,128,5)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1]) # (15,128,5)   (15,128,5)
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False) # (15,128,10) (15,128,20)   (15,128,10) (15,128,20)
            out += self.lateral[i](outputs[i]) # (15,128,10) (15,128,20)   (15,128,10) (15,128,20)

        out = self.output(out)[:, :, -1] # (15,128)   (15,128)
        return out

class PointFeatureAggregator(nn.Module):
    def __init__(self, hidden_size: int, aggre_out: bool, dropout: float = 0.1) -> None:
        super(PointFeatureAggregator, self).__init__()
        self.aggre_out = aggre_out

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def _global_maxpool_aggre(self, feat):
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).permute(0, 2, 1)

    def forward(self, x_inp):  # (197,10,128) (197,10,128)   (197,10,128) (197,10,128)
        x = self.fc1(x_inp)  # [N_{lane}, 10, hidden_size]  (197,10,128) (197,10,128)   (197,10,128) (197,10,128)
        x_aggre = self._global_maxpool_aggre(x)  # (197,1,128) (197,1,128)   (197,1,128) (197,1,128)
        x_aggre = torch.cat([x, x_aggre.repeat([1, x.shape[1], 1])], dim=-1)  # (197,10,256) (197,10,256)   (197,10,256) (197,10,256)

        out = self.norm(x_inp + self.fc2(x_aggre))  # (197,10,128) (197,10,128)   (197,10,128) (197,10,128)
        if self.aggre_out:
            return self._global_maxpool_aggre(out).squeeze() # _
        else:
            return out


class Map_Encoder(nn.Module):
    def __init__(self, device, in_size=10, hidden_size=128, dropout=0.1):
        super(Map_Encoder, self).__init__()
        self.device = device

        self.proj = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.aggre1 = PointFeatureAggregator(hidden_size=hidden_size, aggre_out=False, dropout=dropout)
        self.aggre2 = PointFeatureAggregator(hidden_size=hidden_size, aggre_out=True, dropout=dropout)
    '''
    def forward(self, feats):
        outs = []
        for feat in feats:
            x = self.proj(feat)  # [N_{lane}, 10, hidden_size]
            x = self.aggre1(x)
            x = self.aggre2(x)  # [N_{lane}, hidden_size]
            outs.extend(x)
        return outs
    '''
    def forward(self, feats):  # (197,10,10)    (197,10,8)
        x = self.proj(feats)  # [N_{lane}, 10, hidden_size] (197,10,128)   (197,10,128)
        x = self.aggre1(x) # (197,10,128)   (197,10,128)
        x = self.aggre2(x)  # [N_{lane}, hidden_size] (197,128)   (197,128)
        return x


class Spatial_Feature_Layer(nn.Module):
    def __init__(self,
                 device,
                 d_edge: int = 128,
                 d_model: int = 128,
                 d_ffn: int = 2048,
                 n_head: int = 8,
                 dropout: float = 0.1,
                 update_edge: bool = True) -> None:
        super(Spatial_Feature_Layer, self).__init__()
        self.device = device
        self.update_edge = update_edge

        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge),
                nn.LayerNorm(d_edge),
                nn.ReLU(inplace=True)
            )
            self.norm_edge = nn.LayerNorm(d_edge)

        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=False)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self,
                node: Tensor,
                edge: Tensor,
                edge_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                node:       (N, d_model)  (212,128)
                edge:       (N, N, d_model)  (212,212,128)
                edge_mask:  (N, N)
        '''
        # update node
        x, edge, memory = self._build_memory(node, edge)  # (1,212,128) (212,212,128) (212,212,128) +1 +1 +1
        x_prime, _ = self._mha_block(x, memory, attn_mask=None, key_padding_mask=edge_mask)  # (1,212,128) +1 +1 +1
        x = self.norm2(x + x_prime).squeeze() # (212,128) +1 +1 +1
        x = self.norm3(x + self._ff_block(x)) # (212,128) +1 +1 +1
        return x, edge, None

    def _build_memory(self,
                      node: Tensor,
                      edge: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        '''
            input:
                node:   (N, d_model)
                edge:   (N, N, d_edge)
            output:
                :param  (1, N, d_model)
                :param  (N, N, d_edge)
                :param  (N, N, d_model)
        '''
        n_token = node.shape[0]

        # 1. build memory
        src_x = node.unsqueeze(dim=0).repeat([n_token, 1, 1])  # (N, N, d_model)
        tar_x = node.unsqueeze(dim=1).repeat([1, n_token, 1])  # (N, N, d_model)
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # (N, N, d_model)
        # 2. (optional) update edge (with residual)
        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (N, N, d_edge)

        return node.unsqueeze(dim=0), edge, memory # edge是RPE, memory是C

    # multihead attention block
    def _mha_block(self,
                   x: Tensor,
                   mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                x:                  [1, N, d_model]
                mem:                [N, N, d_model]
                attn_mask:          [N, N]
                key_padding_mask:   [N, N]
            output:
                :param      [1, N, d_model]
                :param      [N, N]
        '''
        x, _ = self.multihead_attn(x, mem, mem,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False)  # return average attention weights
        return self.dropout2(x), None

    # feed forward block
    def _ff_block(self,
                  x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SymmetricFusionTransformer(nn.Module):
    def __init__(self,
                 device,
                 d_model: int = 128,
                 d_edge: int = 128,
                 n_head: int = 8,
                 n_layer: int = 6,
                 dropout: float = 0.1,
                 update_edge: bool = True):
        super(SymmetricFusionTransformer, self).__init__()
        self.device = device

        fusion = []
        for i in range(n_layer):
            need_update_edge = False if i == n_layer - 1 else update_edge
            fusion.append(Spatial_Feature_Layer(device=device,
                                   d_edge=d_edge,
                                   d_model=d_model,
                                   d_ffn=d_model*2,
                                   n_head=n_head,
                                   dropout=dropout,
                                   update_edge=need_update_edge))
        self.fusion = nn.ModuleList(fusion)

    def forward(self, x: Tensor, edge: Tensor, edge_mask: Tensor) -> Tensor:
        '''
            x: (N, d_model)  (212,128)
            edge: (d_model, N, N)  (212,212,128)
            edge_mask: (N, N) None
        '''
        # attn_multilayer = []
        for mod in self.fusion:
            x, edge, _ = mod(x, edge, edge_mask) # (212,128) (212,212,128) None
            # attn_multilayer.append(attn)
        return x, None


class FusionNet(nn.Module):
    def __init__(self, device, config):
        super(FusionNet, self).__init__()
        self.device = device

        d_embed = config['d_embed']
        dropout = config['dropout']
        update_edge = config['update_edge']

        self.proj_actor = nn.Sequential(
            nn.Linear(config['d_actor'], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_lane = nn.Sequential(
            nn.Linear(config['d_lane'], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_rpe_scene = nn.Sequential(
            nn.Linear(config['d_rpe_in'], config['d_rpe']),
            nn.LayerNorm(config['d_rpe']),
            nn.ReLU(inplace=True)
        )

        self.fuse_scene = SymmetricFusionTransformer(self.device,
                                                     d_model=d_embed,
                                                     d_edge=config['d_rpe'],
                                                     n_head=config['n_scene_head'],
                                                     n_layer=config['n_scene_layer'],
                                                     dropout=dropout,
                                                     update_edge=update_edge)

    def forward(self,
                actors: Tensor,
                actor_idcs: List[Tensor],
                lanes: Tensor,
                lane_idcs: List[Tensor],
                rpe_prep: Dict[str, Tensor]): # list:1
        # print('actors: ', actors.shape)  (15,128)
        # print('actor_idcs: ', [x.shape for x in actor_idcs]) list:1
        # print('lanes: ', lanes.shape)  (197,128)
        # print('lane_idcs: ', [x.shape for x in lane_idcs]) list:1

        # projection
        actors = self.proj_actor(actors)  # (15,128)
        lanes = self.proj_lane(lanes)  # (197,128)

        actors_new, lanes_new = list(), list()
        for a_idcs, l_idcs, rpes in zip(actor_idcs, lane_idcs, rpe_prep): # (15,) (197,) dict:2
            # * fusion - scene
            _actors = actors[a_idcs]  # (15,128)
            _lanes = lanes[l_idcs]  # (197,128)
            tokens = torch.cat([_actors, _lanes], dim=0)  # (N, d_model)  (212,128)
            rpe = self.proj_rpe_scene(rpes['scene'].permute(1, 2, 0))  # (N, N, d_rpe)  (212,212,128)
            # print("rpe2:",rpe) # tensor([[[1.4250, 0.0000, 0.0000,  ..., 1.9236, 0.0000, 0.1095],
            out, _ = self.fuse_scene(tokens, rpe, rpes['scene_mask']) # (212,128)

            actors_new.append(out[:len(a_idcs)]) # list:1
            lanes_new.append(out[len(a_idcs):]) # list:1

        # print('actors: ', [x.shape for x in actors_new])
        # print('lanes: ', [x.shape for x in lanes_new])
        actors = torch.cat(actors_new, dim=0) # (15,128)
        lanes = torch.cat(lanes_new, dim=0) # (197,128)
        # print('actors: ', actors.shape)
        # print('lanes: ', lanes.shape)
        return actors, lanes, None


class Interaction_Module_SC_AM(nn.Module):  #
    def __init__(self, device, hidden_size, depth=3):
        super(Interaction_Module_SC_AM, self).__init__()

        self.depth = depth

        self.AA = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.AL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.LL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.LA = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])

    def forward(self, agent_features, lane_features, masks): # (1,15,128) (1,197,128)

        for layer_index in range(self.depth):
            # === Lane to Agent ===
            lane_features = self.LA[layer_index](lane_features, agent_features, attn_mask=masks[-1]) # (1,197,128)   (1,197,128)   (1,197,128)
            # === === ===

            # === Lane to Lane ===
            lane_features = self.LL[layer_index](lane_features, attn_mask=masks[-2]) # (1,197,128)   (1,197,128)   (1,197,128)
            # === ==== ===

            # === Agent to Lane ===
            agent_features = self.AL[layer_index](agent_features, lane_features, attn_mask=masks[-3]) # (1,15,128)   (1,15,128)   (1,15,128) (1,14,14,1)
            # === ==== ===

            # === Agent to Agent ===
            agent_features = self.AA[layer_index](agent_features, attn_mask=masks[-4]) # (1,15,128)   (1,15,128)   (1,15,128)
            # === ==== ===

        return agent_features, lane_features
# masks[-1], masks[-2], masks[-3], masks[-4] 分别是用于不同交互方向的掩码。


class Interaction_Module_FE(nn.Module):  #
    def __init__(self, device, hidden_size, depth=3):
        super(Interaction_Module_FE, self).__init__()

        self.depth = depth
        self.AL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])

    def forward(self, agent_features, lane_features, masks): # (1,15,128) (1,15,128)   (1,15,128) (1,197,128)

        for layer_index in range(self.depth):

            # === Agent to Lane ===
            agent_features = self.AL[layer_index](agent_features, lane_features, attn_mask=masks[0]) # (1,15,128) (1,15,128) (1,15,128)   (1,15,128) (1,15,128) (1,15,128)
            # === ==== ===

        return agent_features, lane_features


class Attention_Block(nn.Module):  #
    def __init__(self, hidden_size, num_heads=8, p_drop=0.1):
        super(Attention_Block, self).__init__()
        self.multiheadattention = Attention(hidden_size, num_heads, p_drop)

        self.ffn_layer = MLP(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)


    def forward(self, query, key_value=None, attn_mask=None): # (1,197,128) (1,15,128) (1,197,15)   (1,197,128) (1,197,128) (1,197,197)    (1,15,128) (1,197,128) (1,15,197)   (1,15,128) (1,15,128) (1,15,15)
        if key_value is None:
            key_value = query

        attn_output = self.multiheadattention( # (1,197,128)   (1,197,128)   (1,15,128)   (1,15,128)
            query, key_value, attention_mask=attn_mask)

        query = self.norm1(attn_output + query) # (1,197,128)   (1,197,128)   (1,15,128)   (1,15,128)
        query_temp = self.ffn_layer(query) # (1,197,128)   (1,197,128)   (1,15,128)   (1,15,128)
        query = self.norm2(query_temp + query) # (1,197,128)   (1,197,128)   (1,15,128)   (1,15,128)

        return query


class MLP(nn.Module):  #
    def __init__(self, input_dim, output_dim, p_drop=0.0, hidden_dim=None, residual=False):
        super(MLP, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layer2_dim = hidden_dim
        if residual:
            layer2_dim = hidden_dim + input_dim

        self.residual = residual
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(layer2_dim, output_dim) #
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout1(out)
        if self.residual:
            out = self.layer2(torch.cat([out, x], dim=-1))
        else:
            out = self.layer2(out)

        out = self.dropout2(out)
        return out


class Attention(nn.Module):  #
    def __init__(self, hidden_size, num_attention_heads, p_drop):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.last_projection = nn.Linear(self.all_head_size, hidden_size)
        self.attention_drop = nn.Dropout(p_drop)

    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1) # (1,1,14,14,1)
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.logical_not() * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_value_states, attention_mask):
        mixed_query_layer = self.query(query_states) # (1,14,128)
        mixed_key_layer = F.linear(key_value_states, self.key.weight) # (1,14,128)
        mixed_value_layer = self.value(key_value_states) # (1,14,128)

        query_layer = self.transpose_for_scores(mixed_query_layer) # (1,8,14,16)
        key_layer = self.transpose_for_scores(mixed_key_layer) # (1,8,14,16)
        value_layer = self.transpose_for_scores(mixed_value_layer) # (1,8,14,16)
        attention_scores = torch.matmul( # (1,8,14,14)
            query_layer/math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if attention_mask is not None:
            attention_scores = attention_scores + \
                self.get_extended_attention_mask(attention_mask) # (1,8,198,14) (1,1,14,14,1)

        attention_probs = nn.Softmax(dim=-1)(attention_scores) # (1,8,198,14)
        attention_probs = self.attention_drop(attention_probs)

        assert torch.isnan(attention_probs).sum() == 0

        context_layer = torch.matmul(attention_probs, value_layer) # (1,8,198,16)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (1,198,8,16)

        new_context_layer_shape = context_layer.size()[ # (1,198,128)
                                  :-2] + (self.all_head_size,)
        # context_layer.shape = (batch, max_vector_num, all_head_size)
        context_layer = context_layer.view(*new_context_layer_shape) # (1,198,128)
        context_layer = self.last_projection(context_layer)
        return context_layer


class Trajectory_Decoder_Future_Enhanced(nn.Module):  #
    def __init__(self, device, hidden_size):
        super(Trajectory_Decoder_Future_Enhanced, self).__init__()
        self.endpoint_predictor = MLP(hidden_size, 6*2, residual=True)

        self.get_trajectory = MLP(hidden_size + 2, 29*2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + 2, 2, residual=True)

    def forward(self, agent_features): # (1,15,128)
        # agent_features.shape = (N, M, 128)
        N = agent_features.shape[0] # 1
        M = agent_features.shape[1] # 15
        D = agent_features.shape[2] # 128

        # endpoints.shape = (N, M, 6, 2)
        endpoints = self.endpoint_predictor(agent_features).view(N, M, 6, 2) # (1,15,6,2) MLPinit

        # prediction_features.shape = (N, M, 6, 128)
        agent_features_expanded = agent_features.unsqueeze(dim=2).expand(N, M, 6, D) # (1,15,6,128)

        # offsets.shape = (N, M, 6, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features_expanded, endpoints.detach()], dim=-1)) # (1,15,6,2)
        endpoints += offsets # (1,15,6,2)

        # agent_features_expanded.shape = (N, M, 6, 128 + 2)
        agent_features_expanded = torch.cat([agent_features_expanded, endpoints.detach()], dim=-1) # (1,15,6,130)

        predictions = self.get_trajectory(agent_features_expanded).view(N, M, 6, 29, 2) # (1,15,6,29,2)
        #logits = self.get_prob(agent_features_expanded).view(N, M, 6)

        predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2) # (1,15,6,30,2)

        assert predictions.shape == (N, M, 6, 30, 2)

        return predictions


class Trajectory_Decoder_Final(nn.Module):  #
    def __init__(self, device, hidden_size):
        super(Trajectory_Decoder_Final, self).__init__()
        self.endpoint_predictor = MLP(hidden_size, 6*2, residual=True)

        self.get_trajectory = MLP(hidden_size + 2, 29*2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + 2, 2, residual=True)

        self.cls = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size + 2),
            nn.LayerNorm(hidden_size+ 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size + 2, hidden_size + 2),
            nn.LayerNorm(hidden_size + 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size + 2, 1)
        )

    def forward(self, agent_features): # (1,15,256)
        # agent_features.shape = (N, M, 128)
        N = agent_features.shape[0] # 1
        M = agent_features.shape[1] # 15
        D = agent_features.shape[2] # 256

        # endpoints.shape = (N, M, 6, 2)
        endpoints = self.endpoint_predictor(agent_features).view(N, M, 6, 2) # (1,15,6,2)

        # prediction_features.shape = (N, M, 6, 128)
        agent_features_expanded = agent_features.unsqueeze(dim=2).expand(N, M, 6, D) # (1,15,6,256)

        # offsets.shape = (N, M, 6, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features_expanded, endpoints.detach()], dim=-1)) # (1,15,6,2)
        endpoints += offsets # (1,15,6,2)

        # agent_features_expanded.shape = (N, M, 6, 128 + 2)
        agent_features_expanded = torch.cat([agent_features_expanded, endpoints.detach()], dim=-1) # (1,15,6,258)

        predictions = self.get_trajectory(agent_features_expanded).view(N, M, 6, 29, 2) # (1,15,6,29,2)

        #logits = self.get_prob(agent_features_expanded).view(N, M, 6)

        logits = self.cls(agent_features_expanded).view(N, M, 6) # (1,15,6)
        logits = F.softmax(logits * 1.0, dim=2)  # e.g., [159, 6] (1,15,6)

        predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2) # (1,15,6,30,2)

        assert predictions.shape == (N, M, 6, 30, 2)

        return predictions, logits

class Reliable_Future_Trajectory_Encoder(nn.Module):  #
    def __init__(self):
        super(Reliable_Future_Trajectory_Encoder, self).__init__()
        self.get_encoder = MLP(360, 128, residual=True)  # just MLP

    def forward(self, agent_features): # (1,15,6,30,2)
        # agent_features.shape = (N, M, 128)
        N, M, _, _, _ = agent_features.shape

        flattened_input = agent_features.view(N, M, -1) # (1,15,360)

        future_features = self.get_encoder(flattened_input).view(N, M, 128) # (1,15,128)

        return future_features


def get_mask(agent_lengths, lane_lengths, device):
    max_lane_num = max(lane_lengths) # 15?   197
    max_agent_num = max(agent_lengths) # 15   15
    batch_size = len(agent_lengths) # 1   1

    # === Agent - Lane Mask ===
    # query: agent, key-value: lane
    AL_mask = torch.zeros(  # (1,15,15)   (1,15,197)
        batch_size, max_agent_num, max_lane_num, device=device)

    for i, (agent_length, lane_length) in enumerate(zip(agent_lengths, lane_lengths)):
        AL_mask[i, :agent_length, :lane_length] = 1

    masks = [AL_mask] # list:1   list:1

    # === === === === ===
    return masks


def get_masks(agent_lengths, lane_lengths, device):
    max_lane_num = max(lane_lengths) # 197
    max_agent_num = max(agent_lengths) # 15
    batch_size = len(agent_lengths) # 1

    # === === Mask Generation Part === ===
    # === Agent - Agent Mask ===
    # query: agent, key-value: agent
    # === 用 actor_topo 替换 AA_mask ===
    # if actor_topo is not None:
    #     AA_mask = actor_topo.to(device)  # 直接使用外部传入的拓扑掩码
    # else:
    #     # 保留原有逻辑（如果未提供 actor_topo，则生成默认掩码）
    #     AA_mask = torch.zeros(batch_size, max_agent_num, max_agent_num, device=device)
    #     for i, agent_length in enumerate(agent_lengths):
    #         AA_mask[i, :agent_length, :agent_length] = 1

    AA_mask = torch.zeros( # (1,15,15)
        batch_size, max_agent_num, max_agent_num, device=device)

    for i, agent_length in enumerate(agent_lengths):
        AA_mask[i, :agent_length, :agent_length] = 1
    # === === ===
    # print("AA_mask",AA_mask.shape) # torch.Size([2, 42, 42])
    # print("AA_mask",AA_mask)

    # === Agent - Lane Mask ===
    # query: agent, key-value: lane
    AL_mask = torch.zeros( # (1,15,197)
        batch_size, max_agent_num, max_lane_num, device=device)

    for i, (agent_length, lane_length) in enumerate(zip(agent_lengths, lane_lengths)):
        AL_mask[i, :agent_length, :lane_length] = 1
    # === === ===

    # === Lane - Lane Mask ===
    # query: lane, key-value: lane
    LL_mask = torch.zeros( # (1,197,197)
        batch_size, max_lane_num, max_lane_num, device=device)

    QL_mask = torch.zeros( # (1,6,197)
        batch_size, 6, max_lane_num, device=device)

    for i, lane_length in enumerate(lane_lengths):
        LL_mask[i, :lane_length, :lane_length] = 1

        QL_mask[i, :, :lane_length] = 1

    # === === ===

    # === Lane - Agent Mask ===
    # query: lane, key-value: agent
    LA_mask = torch.zeros( # (1,197,15)
        batch_size, max_lane_num, max_agent_num, device=device)

    for i, (lane_length, agent_length) in enumerate(zip(lane_lengths, agent_lengths)):
        LA_mask[i, :lane_length, :agent_length] = 1

    # === === ===

    masks = [AA_mask, AL_mask, LL_mask, LA_mask] # list:4

    # === === === === ===

    return masks, QL_mask




# class Build_Top_Layer(nn.Module):
#     def __init__(self, device, d_model, map_d_model,  actor_d_model, dropout, num_decoder_layers):
#         super(Build_Top_Layer, self).__init__()
#
#         self.d_model = d_model
#         self.map_d_model = map_d_model
#         self.actor_d_model = actor_d_model
#         self.dropout = dropout
#         self.num_decoder_layers = num_decoder_layers
#         self.actor_topo_fusers = nn.ModuleList(
#             [TopoFuser(actor_d_model, actor_d_model // 2, dropout) for _ in range(num_decoder_layers)]  # 10
#         )
#         self.actor_topo_decoders = nn.ModuleList(  # 12 调用topodecoder
#             [TopoDecoder(actor_d_model // 2, dropout, self.multi_step) for _ in range(num_decoder_layers)]
#         )
#
#     def forward(self,):
#         fuse_layer = self.actor_topo_fusers()
#         decoder_layer = self.actor_topo_decoders()
#
#         return  fuse_layer, decoder_layer
