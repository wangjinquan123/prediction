import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.vis_utils import ArgoMapVisualizer


class Visualizer():
    def __init__(self, save_dir=None):
        self.map_vis = ArgoMapVisualizer()

        if save_dir is None or save_dir == "":
            save_dir = "/home/wangjinquan/Wjinquan/DGFNet-main/visualizations"

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def draw_once(self, post_out, data, eval_out, braids=None, history_matrics=None, show_map=False, test_mode=False, split='val'):
        batch_size = len(data['SEQ_ID'])  # data(1,15,15,1)?

        city_name = data['CITY_NAME'][0]
        orig = data['ORIG'][0]  #
        rot = data['ROT'][0]
        seq_id = data['SEQ_ID'][0]
        trajs_obs = data['TRAJS_OBS'][0]
        trajs_fut = data['TRAJS_FUT'][0]
        pads_obs = data['PAD_OBS'][0]
        pads_fut = data['PAD_FUT'][0]
        trajs_ctrs = data['TRAJS_CTRS'][0]  # torch.Size([14, 2])
        trajs_vecs = data['TRAJS_VECS'][0]  # torch.Size([14, 2])
        lane_graph = data['LANE_GRAPH'][0]

        res_cls = post_out['out_sc'][0]
        res_reg = post_out['out_sc'][1]

        _, ax = plt.subplots(figsize=(12, 12))
        ax.axis('equal')
        ax.set_title('{}-{}'.format(seq_id, city_name))

        # print("orig:", orig)  # orig: tensor([1774.1851,  390.5243])
        # print("rot:", rot)  # rot: tensor([[ 0.7632, -0.6462],[ 0.6462,  0.7632]])

        if show_map:
            self.map_vis.show_surrounding_elements(ax, city_name, orig, rot=rot)
        else:
            rot = torch.eye(2)
            orig = torch.zeros(2)

        # ax.plot(0, 0, marker='s', color='black', markersize=10, label='Ego Car')   # 显示自车  +1
        # ax.legend()

        # trajs
        for i, (traj_obs, pad_obs, ctr, vec) in enumerate(zip(trajs_obs, pads_obs, trajs_ctrs, trajs_vecs)):  # i:13
            zorder = 10
            if i == 0:
                clr = 'r'
                zorder = 20
            elif i == 1:
                clr = 'cornflowerblue'
            else:
                clr = 'royalblue'

            # if torch.sum(pad_obs) < 15:
            if torch.sum(pads_obs[i]) < 15 or torch.sum(pads_fut[i]) < 30:
                clr = 'grey'

            theta = np.arctan2(vec[1], vec[0])
            act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

            # print("traj_obs_1:",traj_obs) # tensor([[-1.0531e+01,  0.0000e+00], [-9.9194e+00, -9.0704e-03],...[-3.7523e-01, -2.0396e-02],[ 0.0000e+00,  0.0000e+00]])
            traj_obs = torch.matmul(traj_obs, act_rot.T) + ctr
            # print("traj_obs_2:",traj_obs) # 后面都是跟1不一样的 tensor([[-1.0531e+01,  0.0000e+00],[-9.9194e+00, -9.0704e-03],...[-3.7523e-01, -2.0396e-02], [ 0.0000e+00,  0.0000e+00]])
            traj_obs = torch.matmul(traj_obs, rot.T) + orig
            # print("traj_obs_3:",traj_obs) # tensor([[1766.1477,  383.7194],[1766.6205,  384.1078],...[1773.9119,  390.2662],[1774.1851,  390.5243]])

            # traj_obs_np = traj_obs.cpu().numpy() - orig.cpu().numpy()  # 平移到自车中心 +1
            # traj_obs_np = traj_obs_np.dot(rot.cpu().numpy().T)  # 旋转坐标系对齐
            # print("traj_obs：",traj_obs) # 就是自车的历史轨迹

            ax.plot(traj_obs[:, 0], traj_obs[:, 1], marker='.', alpha=0.5, color=clr,zorder=zorder)  # 显示自车历史轨迹 显示他车历史轨迹
            # print("traj_obs[-1,0]:", traj_obs[-1, 0])  # 自车0的最后一个点x坐标 tensor(1774.1851) 他车1的最后一个x坐标tensor(1773.1509) tensor(1797.7736) tensor(1816.0104) tensor(1780.6898) tensor(1773.0310) tensor(1776.5281)
            # print("traj_obs[-1,1]:", traj_obs[-1, 1])  # 自车0的最后一个点y坐标 tensor(390.5243) 他车1的最后一个y坐标tensor(397.2153) tensor(416.9710) tensor(425.3833) tensor(389.4348) tensor(411.6733) tensor(419.9353)
            ax.plot(traj_obs[-1, 0], traj_obs[-1, 1], marker='o', alpha=0.5, color=clr, zorder=zorder, markersize=10)  # 历史轨迹末端

            # 新增轨迹编号标注
            label_pos = traj_obs[0]  # 在轨迹起点标注
            ax.text(label_pos[0], label_pos[1], f'#{i}',
                    fontsize=8, color=clr,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        if history_matrics is not None and braids is not None:
            braids_np = braids.cpu().numpy() if isinstance(braids, torch.Tensor) else braids
            history_np = history_matrics.cpu().numpy() if isinstance(history_matrics, torch.Tensor) else history_matrics
            batch_size = braids.shape[0]
            his_agent_num = history_matrics.shape[1]
            # print("braids_np:", braids_np) # braids_np: [[[[False]
            # print("history_np:", history_np) #  [[[[[ 0.0000000e+00  0.0000000e+00]
            # print("batch_size:", batch_size) # 1
            # print("his_agent_num:", his_agent_num) # 18

            orig_np = orig.cpu().numpy()  # [1774.185, 390.52426]
            rot_np = rot.cpu().numpy()
            # print("orig:", orig_np)  # 应与现有轨迹的 orig 一致 # orig:[ 427.89792 1142.9688 ] [1739.7024  455.2432]
            # print("rot:", rot_np)  # 应与现有轨迹的 rot 一致
            # rot: [[-0.99987346 -0.01590915] [[-0.7567467  -0.65370816]
            #  [ 0.01590915 -0.99987346]] [ 0.65370816 -0.7567467 ]]

            # 在循环外打印 trajs_vecs 的维度
            # print("trajs_vecs shape:", trajs_vecs.shape)  # torch.Size([14, 2])
            # print("trajs_ctrs shape:", trajs_ctrs.shape)  # torch.Size([14, 2])

            # 在绘制绿线前添加调试代码
            # print("braids_np 中标记为 True 的代理对：")
            for b in range(batch_size):
                # print(braids_np)  # 只有两个true
                for i in range(his_agent_num):
                    for j in range(his_agent_num):
                        if i >= j:
                            continue
                        if braids_np[b][i][j][0] == True and (i == 0 or j == 0):
                            print(f"Batch={b}, Agent{i} <-> Agent{j}")  # Batch=0, Agent1 <-> Agent2
                            # print("history_np[b][i][j]:", history_np[b][i][j]) #
                            p1_orig = history_np[b][i][j][0][:]  # (1,2)
                            p2_orig = history_np[b][i][j][1][:]
                            print(history_np.size)
                            print("p1_orig:", p1_orig)  #  [0. 0.] p1_orig: [24.310974   5.6481047]  非ORG：p1_orig: [-10.531179   0.      ]
                            print("p2_orig:", p2_orig)  # [16.826515 26.259987] p2_orig: [53.558212   5.0261383]  非ORG：p2_orig: [-2.0777111e+01  1.6242266e-06]

                            # 获取当前代理的 ctr（中心坐标）
                            # 修改后（正确：ctr_i 转换为全局坐标）
                            ctr_i_local = trajs_ctrs[i].cpu().numpy()
                            # ctr_i_global = ctr_i_local @ rot_np.T + orig_np  # 全局坐标
                            print("ctr_i_local:", ctr_i_local) # [0. 0.]
                            # print("ctr_i_global:", ctr_i_global)  # [1739.7024  455.2432] ctr_i_global: [1773.1509  397.2153] 非ORG：ctr_i_global: [1774.185    390.52426]

                            # 新增：获取代理 i 的 ctr 和 vec
                            vec_i = trajs_vecs[i].cpu().numpy()  # vec shape: (2,)
                            print(f"vec at agent={i}:", vec_i)  # [ 1.0000000e+00 -4.3463537e-08] 应输出 [x, y] vec at agent=1: [-0.9999814  0.0060991]
                            print(f"vec shape:", vec_i.shape)  # 应输出 (2,) vec shape: (2,)

                            # 坐标转换
                            theta_i = np.arctan2(vec_i[1], vec_i[0])
                            act_rot_i = np.array([
                                [np.cos(theta_i), -np.sin(theta_i)],
                                [np.sin(theta_i), np.cos(theta_i)]
                            ])
                            print("theta_i:", theta_i)  # -4.3463537e-08 theta_i: -3.1415927
                            print("act_rot:\n", act_rot_i)# [[ 1.0000000e+00  4.3463537e-08] [-4.3463537e-08  1.0000000e+00]]
                            # act_rot:  非ORG：
                            #  [[-0.9999814 -0.0060991]  [[ 1. -0.]
                            #  [ 0.0060991 -0.9999814]]  [ 0.  1.]]

                            # # 双重坐标系转换（与轨迹处理完全一致）
                            p1_transformed = (p1_orig @ act_rot_i.T + ctr_i_local) @ rot_np.T + orig_np
                            p2_transformed = (p2_orig @ act_rot_i.T + ctr_i_local) @ rot_np.T + orig_np

                            print("p1_transformed:",p1_transformed)  #  [1739.7024  455.2432] p1_transformed: [1758.1246   377.28702] 非ORG:p1_transformed: [1766.1477   383.71936] p1_transformed: [1766.1477   383.71936]
                            print("p2_transformed:",p2_transformed)  # [1709.8026   446.37067] p2_transformed: [1760.3217  378.3563] 非ORG：p2_transformed: [1758.328   377.0988] p2_transformed: [1789.0895   410.54382]

                            ax.plot([p1_transformed[0], p2_transformed[0]], [p1_transformed[1], p2_transformed[1]],
                                    linestyle='--', color='green',
                                    linewidth=2,
                                    zorder=100)


        if not test_mode:
            # if not test mode, vis GT trajectories
            for i, (traj_fut, pad_fut, ctr, vec) in enumerate(zip(trajs_fut, pads_fut, trajs_ctrs, trajs_vecs)):
                zorder = 10
                if i == 0:
                    clr = 'deeppink'
                    zorder = 20
                elif i == 1:
                    clr = 'deepskyblue'
                else:
                    clr = 'deepskyblue'

                if torch.sum(pads_obs[i]) < 15 or torch.sum(pads_fut[i]) < 30:
                    continue

                theta = np.arctan2(vec[1], vec[0])
                act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])

                traj_fut = torch.matmul(traj_fut, act_rot.T) + ctr
                traj_fut = torch.matmul(traj_fut, rot.T) + orig
                # traj_fut_np = traj_fut.cpu().numpy() - orig.cpu().numpy()  # 平移到自车中心  +1
                # traj_fut_np = traj_fut_np.dot(rot.cpu().numpy().T)  # 旋转坐标系对齐
                ax.plot(traj_fut[:, 0], traj_fut[:, 1], alpha=0.5, color=clr, zorder=zorder)  # gt

                mk = '*' if torch.sum(pad_fut) == 30 else '*'
                ax.plot(traj_fut[-1, 0], traj_fut[-1, 1], marker=mk, alpha=0.5, color=clr, zorder=zorder,
                        markersize=12)  # 端点

        # traj pred all
        # print('res_reg: ', [x.shape for x in res_reg])
        res_reg = res_reg[0].cpu().detach()  # .numpy()
        res_cls = res_cls[0].cpu().detach()  # .numpy()
        for i, (trajs, probs, ctr, vec) in enumerate(zip(res_reg, res_cls, trajs_ctrs, trajs_vecs)):
            if i == 0:
                clr = 'r'
                zorder = 20
            elif i == 1:
                clr = 'cornflowerblue'
            else:
                clr = 'royalblue'

            if torch.sum(pads_obs[i]) < 15 or torch.sum(pads_fut[i]) < 30:
                continue

            theta = np.arctan2(vec[1], vec[0])
            act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

            for traj, prob in zip(trajs, probs):
                if prob < 0.05 and (not i in [0, 1]):
                    continue
                traj = torch.matmul(traj, act_rot.T) + ctr
                traj = torch.matmul(traj, rot.T) + orig
                # traj = np.dot(traj, act_rot.T) + ctr.cpu().numpy()   # +1
                # traj = np.dot(traj, rot.cpu().numpy().T) + orig.cpu().numpy()

                # traj_np = traj - orig.cpu().numpy()  # 平移到自车中心 +1
                # traj_np = traj_np.dot(rot.cpu().numpy().T)  # 旋转坐标系对齐
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.3, color=clr, zorder=zorder, linestyle='--')  # 显示预测轨迹
                # ax.plot(traj[-1, 0], traj[-1, 1], alpha=0.3, marker='o', color=clr, zorder=zorder, markersize=12)

                # traj_end = traj_np[-1]   # 在轨迹终点标注概率 +1
                # ax.text(traj_end[0], traj_end[1], f'{prob:.2f}', fontsize=8, color=clr, zorder=zorder, ha='center',
                #         va='center')

                ax.arrow(traj[-2, 0],
                         traj[-2, 1],
                         (traj[-1, 0] - traj[-2, 0]),
                         (traj[-1, 1] - traj[-2, 1]),
                         # head_width=0.5,  # 调整箭头大小
                         # head_length=0.7,
                         edgecolor=None,
                         # fc=clr,  #
                         color=clr,
                         alpha=0.3,
                         width=0.2,
                         zorder=zorder)

        # lane graph
        node_ctrs = lane_graph['node_ctrs']  # [196, 10, 2]
        node_vecs = lane_graph['node_vecs']  # [196, 10, 2]
        lane_ctrs = lane_graph['lane_ctrs']  # [196, 2]
        lane_vecs = lane_graph['lane_vecs']  # [196, 2]

        for ctrs_tmp, vecs_tmp, anch_pos, anch_vec in zip(node_ctrs, node_vecs, lane_ctrs, lane_vecs):
            anch_rot = torch.Tensor([[anch_vec[0], -anch_vec[1]],
                                     [anch_vec[1], anch_vec[0]]])
            ctrs_tmp = torch.matmul(ctrs_tmp, anch_rot.T) + anch_pos
            ctrs_tmp = torch.matmul(ctrs_tmp, rot.T) + orig
            ax.plot(ctrs_tmp[:, 0], ctrs_tmp[:, 1], alpha=0.1, linestyle='dotted', color='grey')

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.save_dir, f'{split}_{seq_id}.png')
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")

        plt.close()



