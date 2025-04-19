import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.utils import from_numpy


class ArgoDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 mode: str,
                 obs_len: int = 20,
                 pred_len: int = 30,
                 aug: bool = False,
                 verbose: bool = False):
        self.mode = mode
        self.aug = aug
        self.verbose = verbose

        self.dataset_files = []
        self.dataset_len = -1
        self.prepare_dataset(dataset_dir)

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        if self.verbose:
            print('[Dataset] Dataset Info:')
            print('-- mode: ', self.mode)
            print('-- total frames: ', self.dataset_len)
            print('-- obs_len: ', self.obs_len)
            print('-- pred_len: ', self.pred_len)
            print('-- seq_len: ', self.seq_len)
            print('-- aug: ', self.aug)

    def prepare_dataset(self, feat_path):
        if self.verbose:
            print("[Dataset] preparing {}".format(feat_path))

        if isinstance(feat_path, list):
            for path in feat_path:
                sequences = os.listdir(path)
                sequences = sorted(sequences)
                for seq in sequences:
                    file_path = f"{path}/{seq}"
                    self.dataset_files.append(file_path)
        else:
            sequences = os.listdir(feat_path)
            sequences = sorted(sequences)
            for seq in sequences:
                file_path = f"{feat_path}/{seq}"
                self.dataset_files.append(file_path)

        self.dataset_len = len(self.dataset_files)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        df = pd.read_pickle(self.dataset_files[idx])
        '''
            "SEQ_ID", "CITY_NAME",
            "ORIG", "ROT",
            "TIMESTAMP", "TRAJS", "TRAJS_CTRS", "TRAJS_VECS", "PAD_FLAGS",
            "LANE_GRAPH"
        '''

        data = self.data_augmentation(df)

        seq_id = data['SEQ_ID']
        city_name = data['CITY_NAME']
        orig = data['ORIG']
        rot = data['ROT']

        trajs = data['TRAJS'] # ()
        trajs_ori = data['TRAJS_ORI']  # dgf  是以自车为中心的
        trajs_obs = trajs[:, :self.obs_len]
        trajs_fut = trajs[:, self.obs_len:]
        trajs_obs_ori = trajs_ori[:, :self.obs_len]  # dgf
        trajs_fut_ori = trajs_ori[:, self.obs_len:]  # dgf

        pad_flags = data['PAD_FLAGS']  # (num_agents, seq_len)
        pad_obs = pad_flags[:, :self.obs_len]
        pad_fut = pad_flags[:, self.obs_len:]

        trajs_ctrs = data['TRAJS_CTRS']
        trajs_vecs = data['TRAJS_VECS']

        graph = data['LANE_GRAPH']

        lane_ctrs = graph['lane_ctrs']
        lane_vecs = graph['lane_vecs']

        rpes = dict()
        scene_ctrs = torch.cat([torch.from_numpy(trajs_ctrs), torch.from_numpy(lane_ctrs)], dim=0)
        scene_vecs = torch.cat([torch.from_numpy(trajs_vecs), torch.from_numpy(lane_vecs)], dim=0)
        rpes['scene'], rpes['scene_mask'] = self._get_rpe(scene_ctrs, scene_vecs)

        data = {}
        data['SEQ_ID'] = seq_id
        data['CITY_NAME'] = city_name
        data['ORIG'] = orig
        data['ROT'] = rot
        data['TRAJS_OBS'] = trajs_obs
        data['TRAJS_FUT'] = trajs_fut
        data['TRAJS_OBS_ORI'] = trajs_obs_ori  # dgf
        data['TRAJS_FUT_ORI'] = trajs_fut_ori  # dgf
        data['PAD_OBS'] = pad_obs
        data['PAD_FUT'] = pad_fut
        data['TRAJS_CTRS'] = trajs_ctrs
        data['TRAJS_VECS'] = trajs_vecs
        data['LANE_GRAPH'] = graph
        data['RPE'] = rpes

        return data

    def _get_cos(self, v1, v2):
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        cos_angle = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)
        return cos_angle

    def _get_sin(self, v1, v2):
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        sin_angle = (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
        return sin_angle

    def _get_rpe(self, ctrs, vecs, radius=100.0):
        d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
        d_pos = d_pos * 2 / radius
        pos_rpe = d_pos.unsqueeze(0)

        def compute_cos_sin(vec1, vec2):  # dgf
            cos_ag = self._get_cos(vec1, vec2)
            sin_ag = self._get_sin(vec1, vec2)
            return cos_ag, sin_ag

        cos_ag1, sin_ag1 = compute_cos_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))

        v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
        cos_ag2, sin_ag2 = compute_cos_sin(vecs.unsqueeze(0), v_pos)

        ang_rpe = torch.stack([cos_ag1, sin_ag1, cos_ag2, sin_ag2])
        rpe = torch.cat([ang_rpe, pos_rpe], dim=0)

        return rpe, None
    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        batch = from_numpy(batch)
        data = dict()
        data['BATCH_SIZE'] = len(batch)

        # # 关键修改点：计算当前batch最大Actor数量并填充
        # max_actors = max(len(item['TRAJS_OBS']) for item in batch)  # 获取最大Actor数
        #
        #
        # # 填充处理
        # for i in range(len(batch)):
        #     item = batch[i]
        #     pad_size = max_actors - len(item['TRAJS_OBS'])
        #
        #     if pad_size > 0:
        #         # 使用PyTorch张量进行填充
        #         # 历史轨迹填充 (obs_len=20)
        #         trajs_obs_pad = np.zeros((pad_size, self.obs_len, 2))
        #         item['TRAJS_OBS'] = np.concatenate([item['TRAJS_OBS'], trajs_obs_pad], axis=0)
        #
        #         trajs_fut_pad = torch.zeros((pad_size, self.pred_len, 2), dtype=item['TRAJS_FUT'].dtype)
        #         item['TRAJS_FUT'] = torch.cat([item['TRAJS_FUT'], trajs_fut_pad], dim=0)
        #
        #         # Mask填充（填充部分标记为无效）
        #         pad_obs_pad = np.ones((pad_size, self.obs_len))
        #         item['PAD_OBS'] = np.concatenate([item['PAD_OBS'], pad_obs_pad], axis=0)
        #
        #         pad_fut_pad = torch.ones((pad_size, self.pred_len), dtype=item['PAD_FUT'].dtype)
        #         item['PAD_FUT'] = torch.cat([item['PAD_FUT'], pad_fut_pad], dim=0)
        #
        #         # 填充TRAJS_OBS_ORI
        #         trajs_obs_ori_pad = np.zeros((pad_size, self.obs_len, 2))
        #         item['TRAJS_OBS_ORI'] = np.concatenate([item['TRAJS_OBS_ORI'], trajs_obs_ori_pad], axis=0)
        #
        #         # 填充TRAJS_FUT_ORI
        #         trajs_fut_ori_pad = np.zeros((pad_size, self.pred_len, 2))
        #         item['TRAJS_FUT_ORI'] = np.concatenate([item['TRAJS_FUT_ORI'], trajs_fut_ori_pad], axis=0)
        #
        # # for item in batch:
        # #     item['TRAJS_OBS'] = from_numpy(item['TRAJS_OBS'].float())
        # #     item['PAD_OBS'] = from_numpy(item['PAD_OBS'].float())
        # #     item['TRAJS_FUT'] = from_numpy(item['TRAJS_FUT'].float())
        # #     item['PAD_FUT'] = from_numpy(item['PAD_FUT'].float())
        # #     item['TRAJS_OBS_ORI'] = from_numpy(item['TRAJS_OBS_ORI'].float())  # 新增
        # #     item['TRAJS_FUT_ORI'] = from_numpy(item['TRAJS_FUT_ORI'].float())  # 新增
        #
        # tensor_fields = ['TRAJS_OBS', 'TRAJS_FUT', 'TRAJS_OBS_ORI', 'TRAJS_FUT_ORI', 'PAD_OBS', 'PAD_FUT']
        # for item in batch:
        #     for key in tensor_fields:
        #         # 先转换numpy数组为tensor，再进行类型转换
        #         if isinstance(item[key], np.ndarray):  # 新增类型判断
        #             item[key] = from_numpy(item[key])  # 先转换为tensor
        #         item[key] = item[key].float()  # 再进行类型转换


        for key in batch[0].keys():
            data[key] = [x[key] for x in batch]
        '''
            Keys:
            'BATCH_SIZE', 'SEQ_ID', 'CITY_NAME',
            'ORIG', 'ROT',
            'TRAJS_OBS', 'PAD_OBS', 'TRAJS_FUT', 'PAD_FUT', 'TRAJS_CTRS', 'TRAJS_VECS', 'TRAJS_OBS_ORI' , 'TRAJS_FUT_ORI'
            'LANE_GRAPH',
            'RPE'
        '''

        actors, actor_idcs = self.actor_gather(data['BATCH_SIZE'], data['TRAJS_OBS'], data['PAD_OBS'])
        # print(data['TRAJS_OBS'].size)
        actors_sc, _ = self.actor_gather_sc(data['BATCH_SIZE'], data['TRAJS_OBS_ORI'], data['PAD_OBS'])  # dgf
        lanes, lane_idcs, lanes_sc  = self.graph_gather(data['BATCH_SIZE'], data["LANE_GRAPH"])  # dgf

        data['ACTORS'] = actors
        data['ACTORS_SC'] = actors_sc  # dgf
        data['ACTOR_IDCS'] = actor_idcs
        data['LANES'] = lanes
        data['LANES_SC'] = lanes_sc  # dgf
        data['LANE_IDCS'] = lane_idcs

        return data

    # 把轨迹进行处理了
    def actor_gather(self, batch_size, actors, pad_flags):
        num_actors = [len(x) for x in actors]
        actor_idcs = []
        act_feats = []
        count = 0
        for i in range(batch_size):
            act_feats.append(torch.cat([actors[i], pad_flags[i].unsqueeze(2)], dim=2))  # dgf (15,20,3)
        act_feats = [x.transpose(1, 2) for x in act_feats]
        actors = torch.cat(act_feats, 0)

        for i in range(batch_size):
            idcs = torch.arange(count, count + num_actors[i])
            actor_idcs.append(idcs)
            count += num_actors[i]
        return actors, actor_idcs

    def actor_gather_sc(self, batch_size, actors, pad_flags):  # dgf
        num_actors = [len(x) for x in actors]
        idx_actor = []
        features_act = []
        count = 0

        for i in range(batch_size):
            features_act.append(torch.cat([actors[i], pad_flags[i].unsqueeze(2)], dim=2))
        features_act = [x.transpose(1, 2) for x in features_act]
        actors = torch.cat(features_act, 0)
          
        for i in range(batch_size):
            idx = torch.arange(count, count + num_actors[i])
            idx_actor.append(idx)
            count += num_actors[i]

        return actors, idx_actor

    def graph_gather(self, batch_size, graphs):
        lane_idx = []
        lane_count = 0
        
        for i in range(batch_size):
            l_idcs = torch.arange(lane_count, lane_count + graphs[i]["num_lanes"])
            lane_idx.append(l_idcs)
            lane_count += graphs[i]["num_lanes"]

        graph = {key: torch.cat([x[key] for x in graphs], 0) for key in ["node_ctrs", "node_vecs", "turn", "control", "intersect", "left", "right"]}  # dgf
        graph_sc = {key: torch.cat([x[key] for x in graphs], 0) for key in ["sc_vecs", "turn", "control", "intersect", "left", "right"]}  # dgf

        graph.update({key: [x[key] for x in graphs] for key in ["lane_ctrs", "lane_vecs"]})  # dgf

        lanes = torch.cat([
            graph['node_ctrs'],
            graph['node_vecs'],
            graph['turn'],
            graph['control'].unsqueeze(2),
            graph['intersect'].unsqueeze(2),
            graph['left'].unsqueeze(2),
            graph['right'].unsqueeze(2)
        ], dim=-1) 

        lanes_sc = torch.cat([  # dgf
            graph_sc['sc_vecs'],
            graph_sc['turn'],
            graph_sc['control'].unsqueeze(2),
            graph_sc['intersect'].unsqueeze(2),
            graph_sc['left'].unsqueeze(2),
            graph_sc['right'].unsqueeze(2)
        ], dim=-1) 

        return lanes, lane_idx, lanes_sc  # dgf


    def rpe_gather(self, rpes):
        rpe = dict()
        for key in list(rpes[0].keys()):
            rpe[key] = [x[key] for x in rpes]
        return rpe

    def data_augmentation(self, df):

        data = {key: df[key].values[0] for key in df.keys()}  # dgf

        is_aug = random.choices([True, False], weights=[0.3, 0.7])[0]
        if not (self.aug and is_aug):
            return data

        def flip_vertical(coords):  # dgf
            coords[..., 1] *= -1

        flip_vertical(data['TRAJS_CTRS'])  # dgf
        flip_vertical(data['TRAJS_VECS'])  # dgf
        flip_vertical(data['TRAJS'])  # dgf

        lane_graph_keys = ['lane_ctrs', 'lane_vecs', 'node_ctrs', 'node_vecs']  # dgf
        for key in lane_graph_keys:  # dgf
            flip_vertical(data['LANE_GRAPH'][key])

        data['LANE_GRAPH']['left'], data['LANE_GRAPH']['right'] = (  # dgf
            data['LANE_GRAPH']['right'], data['LANE_GRAPH']['left']
        )

        return data