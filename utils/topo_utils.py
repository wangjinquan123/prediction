'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''

import numpy as np 
import torch 


def segments_intersect(line1_start, line1_end, line2_start, line2_end, line1_start_y, line1_end_y, line2_start_y, line2_end_y): # (81,29,2) (81,29,2) (81,29,2) (81,29,2)
    #calculating intersection given arbitary shape

    # Calculating the differences
    dx1 = line1_end[..., 0] - line1_start[..., 0] # (196,29) k1 自车的x轴轨迹的增量
    dy1 = line1_end[..., 1] - line1_start[..., 1] # (196,29)
    dx2 = line2_end[..., 0] - line2_start[..., 0] # (196,29) k2 他车的x轴轨迹的增量
    dy2 = line2_end[..., 1] - line2_start[..., 1] # (196,29)

    du1 = line1_end_y[..., 0] - line1_start_y[..., 0] # (196,29)
    dv1 = line1_end_y[..., 1] - line1_start_y[..., 1]  # (196,29)
    du2 = line2_end_y[..., 0] - line2_start_y[..., 0] # (196,29)
    dv2 = line2_end_y[..., 1] - line2_start_y[..., 1] # (196,29)

    # Calculating determinants
    det = dx1 * dy2 - dx2 * dy1 # (196,29)  # dy2=dy1  这儿判断的是两车在时间步内是否平行
    det_mask = det != 0 # (196,29)
    # print("det:\n", det)
    # print("det_mask:\n", det_mask)

    # # Checking if lines are parallel or coincident
    parallel_mask = torch.logical_not(det_mask) # (196,29)
    # # parallel_mask = torch.logical_and(parallel_mask, close_mask)
    # print("parallel_mask:\n", parallel_mask)

    det_y = du1 * dv2 - du2 * dv1

    # backward_mask = torch.logical_or(dx1 * dx2 <0, du1 * du2 <0)  # 这是针对逆行车辆  用det_T  对自车直行，对向车道左转的车辆会进行误过滤
    # parallel_mask = torch.logical_or(parallel_mask, backward_mask)

    # Calculating intersection parameters
    t1 = ((line2_start[..., 0] - line1_start[..., 0]) * dy2  # (196,29)
          - (line2_start[..., 1] - line1_start[..., 1]) * dx2) / det
    # t2 = ((line2_start[..., 0] - line1_start[..., 0]) * dy1  # (196,29)
    #       - (line2_start[..., 1] - line1_start[..., 1]) * dx1) / det

    t2 = ((line2_start_y[..., 0] - line1_start_y[..., 0]) * dv1
          -(line2_start_y[..., 1] - line1_start_y[..., 1]) * du1)/ det_y

    # Checking intersection conditions
    intersect_mask = torch.logical_and( # (196,29)  这里有问题
        torch.logical_and(t1 >= 0, t1 <= 1),
        torch.logical_and(t2 >= 0, t2 >= 1), # 保留他车在所有时间内在自车前方的情况  t2>=1
    )
    # print("intersect_mask:\n", intersect_mask)

    # intersect_mask = torch.logical_and(intersect_mask,
    #                                    torch.logical_or(line2_start_y[..., 0] - line1_start_y[..., 0] >= 0, line2_end_y[..., 0] - line1_end_y[..., 0] >= 0) )

    intersect_mask = torch.logical_and(intersect_mask,
       torch.logical_or(torch.logical_and(line2_start_y[..., 0] - line1_start_y[..., 0] >= 0, line2_start_y[..., 0] - line1_start_y[..., 0] <= 20 + dv1 * (du2-du1)),
                        torch.logical_and(line2_end_y[..., 0] - line1_end_y[..., 0] >= 0 , line2_end_y[..., 0] - line1_end_y[..., 0] <= 20 + dv1 * (du2-du1)))) # 50?

    # print("intersect_mask:\n", intersect_mask)
    # intersect_mask = torch.logical_and(intersect_mask, torch.abs(du2-du1) < 3 * torch.abs(du1)) # 4  感觉意义不大  现在的这个新版限制会影响braids_precisiion的值，近乎使其置为零

    # Handling parallel or coincident lines
    intersect_mask[parallel_mask] = False # 并行的都置为False

    return intersect_mask


def judge_briad_indicater(src_traj, tar_traj, src_mask, tgt_mask, multi_step=1):
    """
    judge the agent braid indication according to src and ter trajectories:
    src_traj, tar_traj: [b, T, 2]
    return res [b] containing {-1, 0, 1}
    """
    b, t, _ = src_traj.shape # b:81 t:30 _:2
    traj_t = torch.linspace(0, 1, t).to(src_traj.device) # (30,)
    # print("traj_t1:\n", traj_t)
    traj_t = traj_t[None, :].expand(b, -1) # (324,30)  1818得324
    # print("traj_t2:\n", traj_t)

    src_xt = torch.stack([src_traj[:, :, 0], traj_t], dim=-1) # (81,30,2) 0表示y  跟时间traj_t做拼接，得到y-t图
    tar_xt = torch.stack([tar_traj[:, :, 0], traj_t], dim=-1) # (81,30,2)
    # print("src_xt:\n", src_xt)
    # print("tar_xt:\n", tar_xt) # tensor([[[-0.0000e+00,  0.0000e+00], 现在是判断y-t图的相交情况

    src_yt = torch.stack([src_traj[:, :, 1], traj_t], dim=-1) # (81,30,2) 1表示x
    tar_yt = torch.stack([tar_traj[:, :, 1], traj_t], dim=-1) # (81,30,2)
    # print("src_yt:\n", src_yt)
    # print("tar_yt:\n", tar_yt)

    src_start, src_end = src_xt[:, :-1, :2], src_xt[:, 1:, :2] # (81,29,2) (81,29,2)
    tar_start, tar_end = tar_xt[:, :-1, :2], tar_xt[:, 1:, :2] # (81,29,2) (81,29,2)
    # print("src_start:\n", src_start)
    # print("src_end:\n", src_end)
    # print("tar_start:\n", tar_start)
    # print("tar_end:\n", tar_end)

    src_start_y, src_end_y = src_yt[:, :-1, :2], src_yt[:, 1:, :2] # (81,29,2) (81,29,2)
    tar_start_y, tar_end_y = tar_yt[:, :-1, :2], tar_yt[:, 1:, :2] # (81,29,2) (81,29,2)
    # print("src_start_y:\n", src_start_y)
    # print("src_end_y:\n", src_end_y)
    # print("tar_start_y:\n", tar_start_y)
    # # print("tar_end_y:\n", tar_end_y)

    src_start_m, src_end_m = src_mask[:, :-1], src_mask[:, 1:] # (81,29) (81,29)
    src_seg_valid = torch.logical_and(src_start_m, src_end_m) # (81,29)
    # print("src_start_m:\n", src_start_m)
    # print("src_end_m:\n", src_end_m)
    # print("src_seg_valid:\n", src_seg_valid)
    # print("src_seg_valid:", src_seg_valid.shape) #  torch.Size([324, 29])

    tgt_start_m, tgt_end_m = tgt_mask[:, :-1], tgt_mask[:, 1:] # (81,29) (81,29)
    tgt_seg_valid = torch.logical_and(tgt_start_m, tgt_end_m) # (81,29)
    # print("tgt_start_m:\n", tgt_start_m)
    # print("tgt_end_m:\n", tgt_end_m)
    # print("tgt_seg_valid:\n", tgt_seg_valid)
    # print("tgt_seg_valid:", tgt_seg_valid.shape) # torch.Size([81, 29])
    inter_valid = torch.logical_and(src_seg_valid, tgt_seg_valid) # (81,29)
    # print("inter_valid:\n", inter_valid)

    # raw braids :[b, T]
    raw_briad_mask = segments_intersect(src_start, src_end, tar_start, tar_end, src_start_y, src_end_y, tar_start_y, tar_end_y ) # (196,29) True False
    # print("raw_briad_mask1:\n", raw_briad_mask)
    raw_briad_mask = torch.logical_and(raw_briad_mask, inter_valid) # (196,29)
    # print("raw_briad_mask2:\n", raw_briad_mask)
    src_y_start, src_y_end = src_traj[:, :-1, 1], src_traj[:, 1:, 1]
    tar_y_start, tar_y_end = tar_traj[:, :-1, 1], tar_traj[:, 1:, 1]

    dist = torch.sqrt((src_y_start - tar_y_start)**2 + (src_start[..., 0] - tar_start[..., 0])**2)
    # print("src_y_start", src_y_start)
    # print("src_y_end", src_y_end)
    # print("tar_y_start", tar_y_start)
    # print("tar_y_end", tar_y_end)
    # print("dist", dist)

    if multi_step > 1:
        zeros = torch.zeros((raw_briad_mask.shape[0], 1)).to(raw_briad_mask.device)
        raw_briad_mask = torch.cat([raw_briad_mask, zeros], dim=-1)
        t = raw_briad_mask.shape[-1]
        seg_len = t // multi_step
        assert seg_len * multi_step == t
        raw_briad_mask = raw_briad_mask.view(-1, seg_len, multi_step)

    return torch.any(raw_briad_mask, dim=1)


def create_batched_combination_trajs(src_trajs, tgt_trajs):
    """
    src trajs: [b, a, T, d]
    return [b, a, a, 2, T, d]
    """
    b, a, t, d = src_trajs.shape
    blank_traj = torch.zeros_like(src_trajs)
    blank_tgt_traj = torch.zeros_like(tgt_trajs)
    # print("blank_traj:", blank_traj) # tensor([[[[0., 0.],
    # print("blank_tgt_traj:", blank_tgt_traj) # tensor([[[[0., 0.],
    src = torch.stack([src_trajs, blank_traj], dim=2)[:, :, None, :, :] # (1，14，1，2，30，2)
    tgt = torch.stack([blank_tgt_traj, tgt_trajs], dim=2)[:, None, :, :, :] # (1,1,14,2,30,2)
    # print("src:", src)
    # print("tgt:", tgt)
    # print("src:", src.shape) # torch.Size([1, 14, 1, 2, 30, 2])
    res = src + tgt
    return res # (1,14,14,2,30,2)

def generate_behavior_braids(src_trajs, tgt_trajs, src_mask, tgt_mask, multi_step=1): # (1,14,30,2) (1,14,30,2) (1,14,30) (1,14,30)
    """
    generating the behavior_braids label for interacted trajs
    inputs: src trajs: [b, 1, T, d], pos [b, 1, 2], head [b, 1]
    obj_mask: [b, obj]
    tgt_trajs: [b, a, T, d]
    return src_braids: [b, 1, a]
    """
    #make full combinations: [b, 1, a, t]
    mask = src_mask[:,:, None, :] * tgt_mask[:, None, :, :] # (1,18,18,30)
    # print("mask",mask)
    combinated_trajs = create_batched_combination_trajs(src_trajs, tgt_trajs) # (1,14,14,2,30,2)
    # print("combinated_trajs1",combinated_trajs)
    combinated_trajs = combinated_trajs * mask[:, :, :, None, :, None].float() #  torch.Size([1, 18, 18, 2, 30, 2])
    # print("combinated_trajs2",combinated_trajs)# [ 2.1864e+01, -2.3874e+01]
    # print("combinated_trajs:",combinated_trajs.shape) # torch.Size([1, 18, 18, 2, 30, 2])

    # # transformed to ego heading as Y:
    combinated_trajs[..., [0, 1]] = combinated_trajs[..., [1, 0]] # 这里把0和1颠倒了
    combinated_trajs[..., 0] =  -combinated_trajs[..., 0]  # 这个是-y  作为新的x轴 逆时针旋转90°  是将转换到现在的x坐标取反
    # print("combinated_trajs3 ",combinated_trajs )

    # combinated_trajs = combinated_trajs[..., 4::5, :]
    b, a, s, _, t, d = combinated_trajs.shape # b:2 a:42 s:42 _:2 t:30 d:2

    # # 1. 过滤自车后方车辆（纵向位置判断）
    # # 自车轨迹y轴（原x轴）始终大于他车时为后方车辆
    # 1. 后方车辆判断（所有时间步）
    # ego_y = combinated_trajs[:, :, :, 0, :, 1]  # [b, a, a, t] 自车y坐标 [2,42,30] 0323版 ego_y = combinated_trajs[:, :, :, 0, :, 1]
    # obj_y = combinated_trajs[:, :, :, 1, :, 1]  # [b, a, a, t] 他车y坐标 [2,42,42,30] 0323版 obj_y = combinated_trajs[:, :, :, 1, :, 1]
    # # behind_mask = (obj_y < ego_y).all(dim=-1)  # [b, a, a] [2,42,42] 0323版 behind_mask = (obj_y < ego_y).all(dim=-1)
    # print("ego_y:",ego_y) # tensor([[[[ 0.5698,  1.0487,  1.5862,  ..., 12.5883, 12.9042, 13.0780],
    # print("obj_y:",obj_y) # tensor([[[[ 0.5698,  1.0487,  1.5862,  ..., 12.5883, 12.9042, 13.0780],
    # print("obj_y:",obj_y.shape) # torch.Size([1, 18, 18, 30])
    #
    # # 核心条件1：关键时段位置检测
    # # 选取轨迹的25%、50%、75%、100%时点
    # k = ego_y.shape[-1] // 4
    # key_indices = [k, 2 * k, 3 * k, -1]  # 四等分关键点
    #
    # # 构造四维条件张量 [batch, agent, scene, keypoint]
    # key_conditions = obj_y[..., key_indices] < ego_y[..., key_indices] - 1.5  # 1.5m安全余量
    # key_mask = key_conditions.sum(dim=-1) >= 3  # 至少3个关键点满足
    # print("key_conditions:",key_conditions) # tensor([[[[False, False, False, False],
    # print("key_mask:",key_mask) # tensor([[[False, False, False, False, False, False, False, False, False],
    # print("key_mask:",key_mask.shape) #  torch.Size([1, 18, 18])
    #
    # # 核心条件2：速度趋势约束
    # # 计算最后1/3时段的平均速度
    # def compute_velocity(tensor):
    #     """统一速度计算逻辑"""
    #     time_steps = tensor.size(-1)
    #     vel_window = min(5, time_steps - 1)  # 最大窗口5步
    #     return torch.diff(tensor, dim=-1)[..., -vel_window:].mean(dim=-1)
    #
    # ego_vel = compute_velocity(ego_y)  # [b,a,s]
    # obj_vel = compute_velocity(obj_y)  # [b,a,s]
    # vel_mask = obj_vel < ego_vel * 0.85  无依据
    # print("ego_vel:",ego_vel) # tensor([[[ 0.2437,  0.2437,  0.0000,  0.0000,  0.2437,  0.2437,  0.0000,
    # print("obj_vel:",obj_vel) # tensor([[[ 0.2437,  0.2281,  0.0000,  0.0000, -0.0619, -0.0251,  0.0000,
    # print("vel_mask:",vel_mask) # tensor([[[False, False, False, False,  True,  True, False,  True, False,  True,
    # print("vel_mask:",vel_mask.shape) # torch.Size([1, 18, 18])
    #
    # # 核心条件3：动态距离约束
    # min_distance = (ego_y - obj_y).min(dim=-1).values  # 全程最小纵向距离
    # distance_mask_ = min_distance > 0 # -0.5 允许短暂接近但不穿越
    # print("min_distance:",min_distance) #  tensor([[[  0.0000, -11.5891,   0.0000,   0.0000, -15.7031, -16.2530,   0.0000,
    # print("distance_mask_:",distance_mask_) #  tensor([[[False, False, False, False, False, False, False, False, False, False,
    #
    # # 条件组合
    # behind_mask = key_mask & vel_mask & distance_mask_
    # print("behind_mask:",behind_mask) #  tensor([[[False, False, False, False, False, False, False, False, False, False,
    # print("behind_mask:", behind_mask.shape) # torch.Size([1, 18, 18])
    #
    # ego_final = combinated_trajs[:, :, :, 0, -1, 1]  # 自车终点Y坐标 [b, a, s]
    # obj_final = combinated_trajs[:, :, :, 1, -1, 1]  # 他车终点Y坐标
    # print("ego_final:",ego_final) # tensor([[[13.0780, 13.0780,  0.0000,  0.0000, 13.0780, 13.0780,  0.0000,
    # print("obj_final:",obj_final) # tensor([[[13.0780, 17.9721,  0.0000,  0.0000, 15.6695, 16.8248,  0.0000,
    # behind_mask_ = (obj_final < ego_final)
    # print("behind_mask_:",behind_mask_) # tensor([[[False, False, False, False, False, False, False, False, False, False,
    # behind_mask = behind_mask | behind_mask_ # 为true是保留自车后方车辆 并集
    # print("behind_mask:",behind_mask) # tensor([[[False, False, False, False, False, False, False, False, False, False,
    #
    # # 计算他车在变换坐标系中的y方向变化（原始x方向）
    # # 2. 对向车道判断（需扩展到三维）
    # # 获取自车运动方向（转换后坐标系Y轴）
    # ego_direction = src_trajs[:, 0, -1, 1] - src_trajs[:, 0, 0, 1]  # [b]   ?
    # print("src_trajs[:, 0, -1, 1]",src_trajs[:, 0, -1, 1]) # tensor([30.0492], device='cuda:0')
    # print("src_trajs[:, 0, 0, 1]",src_trajs[:, 0, 0, 1]) # tensor([0.1908], device='cuda:0') 开始点
    # print("ego_direction",ego_direction) # tensor([29.8584], device='cuda:0')
    #
    # # 自车位移应取转换后的坐标数据（假设自车索引为0）
    # ego_start_y = combinated_trajs[:, 0, 0, 1, 0, 1]  # [b]
    # ego_end_y = combinated_trajs[:, 0, 0, 1, -1, 1]  # [b]
    # ego_direction_ = ego_end_y - ego_start_y  #  转换后的坐标系Y轴位移  感觉更合理 但问题更多
    # print("ego_start_y ",ego_start_y ) #   tensor([0.5698], device='cuda:0') 开始点是什么时候?  我能不能得到自车的整个轨迹路线，至少在30个时间步内，来判断自车的方向？
    # print("ego_end_y ",ego_end_y ) #  tensor([13.0780], device='cuda:0')这儿的8.8185是前面的自车的最后时刻的y坐标，问题不大
    # print("ego_direction_ ",ego_direction_ ) # tensor([12.5083], device='cuda:0') 我的建议是最好在这儿的未来轨迹起始点作为整个坐标的原点，方向为y轴为正方向
    #
    # # 获取他车运动方向（转换后坐标系Y轴）
    # obj_start_y = combinated_trajs[:, :, :, 1, 0, 1]  # [b, a, s] 0323版  y_start = combinated_trajs[:, :, :, 1, 0, 1]
    # obj_end_y = combinated_trajs[:, :, :, 1,-1, 1]  # [b, a, s] 0323版 y_end = combinated_trajs[:, :, :, 1, -1, 1]
    # obj_displacement = obj_end_y - obj_start_y         # [b, a, s]
    # print("obj_start_y ",obj_start_y ) #  tensor([[[ 0.5698, 12.1588,  0.0000,  0.0000, 16.2729, 16.8227, -0.0000,  开始点的y坐标
    # print("obj_end_y ",obj_end_y ) # tensor([[[13.0780, 17.9721,  0.0000,  0.0000, 15.6695, 16.8248,  0.0000,
    # print("obj_displacement ",obj_displacement ) # tensor([[[ 1.2508e+01,  5.8133e+00,  0.0000e+00,  0.0000e+00, -6.0347e-01,
    #
    # # 双条件融合判断
    # condition_1 = (obj_end_y - obj_start_y) < 0  # 传统对向判断 0323版 oncoming_mask = (y_end - y_start) < 0
    # condition_2 = (torch.sign(ego_direction_).unsqueeze(-1).unsqueeze(-1) * obj_displacement) < 0  # 相对方向判断
    # print("condition_1 ",condition_1 ) # tensor([[[False, False, False, False,  True, False, False,  True, False, False,
    # print("condition_2 ",condition_2 ) # tensor([[[False, False, False, False,  True, False, False,  True, False, False,
    #
    # oncoming_mask = condition_1 | condition_2
    # print("oncoming_mask ",oncoming_mask ) # tensor([[[False, False, False, False,  True, False, False,  True, False, False,  4号和7号逆行？
    #
    # valid_mask = (
    #         ~behind_mask &
    #         ~oncoming_mask
    # )
    # print("valid_mask ",valid_mask ) # tensor([[[False,  True, False, False, False,  True, False, False, False,  True, 保留了2、6、10、11、12、13、15、16号，共8个 最终是6号胜出

    # combinated_trajs = combinated_trajs * valid_mask[:, :, :, None, None, None].float()
    # print("combinated_trajs3 ",combinated_trajs ) #  tensor([[[[[[ -0.0000,   0.0000],
    combinated_trajs = combinated_trajs.view(b*a*s, 2, t, d) # (196,2,30,2)
    # print("combinated_trajs4",combinated_trajs) # tensor([[[[ -0.0000,   0.0000],
    # mask = mask[..., 4::5]
    mask = mask[:, :, :, None, :].repeat(1, 1, 1, 2, 1) # (1,9,9,2,30)
    combined_mask = mask.view(b*a*s, 2, t) # (196,2,30)
    # print("mask2",mask)
    # print("combined_mask:\n", combined_mask)
    src_mask, tgt_mask = combined_mask[:, 0], combined_mask[:, 1] # (196,30) (196,30)
    # print("src_mask:\n", src_mask)
    # print("tgt_mask:\n", tgt_mask)
    # print("src_mask.shape ", src_mask.shape ) #  torch.Size([324, 30])

    src_comb_trajs, tgt_comb_trajs = combinated_trajs[:, 0], combinated_trajs[:, 1]
    # print("src_comb_trajs:\n", src_comb_trajs)
    # print("tgt_comb_trajs:\n", tgt_comb_trajs)
    # print("src_comb_trajs.shape ",src_comb_trajs.shape ) #  torch.Size([324, 30, 2])

    # calculating the braids:
    braids = judge_briad_indicater(src_comb_trajs, tgt_comb_trajs, src_mask, tgt_mask, multi_step) # (196,)
    # print("braids1:\n", braids)
    braids = braids.reshape(b, a, s, multi_step) # (1,14,14,1)
    # print("braids2 ", braids ) # tensor([[[[False],
    return braids


# print("line1_end[..., 0]:",line1_end[..., 0])
    # print("line1_start[..., 0]:", line1_start[..., 0])
    # print("dx1:", dx1)  # tensor([[0., 0., 0.,  ..., 0., 0., 0.],
    #
    # print("line1_end[..., 1]:",line1_end[..., 1])
    # print("line1_start[..., 1]:",line1_start[..., 1])
    # print("dy1:", dy1)  # tensor([[0.0345, 0.0345, 0.0345,  ..., 0.0345, 0.0345, 0.0345],
    #
    # print("line2_end[..., 0]:",line2_end[..., 0])
    # print("line2_start[..., 0]:", line2_start[..., 0])
    # print("dx2:", dx2)  # tensor([[0., 0., 0.,  ..., 0., 0., 0.],
    #
    # print("line2_end[..., 1]:",line2_end[..., 1])  # 与line1_end[..., 1]一致
    # print("line2_start[..., 1]:",line2_start[..., 1]) # 与line1_start[..., 1]一致
    # print("dy2:",dy2) # tensor([[0.0345, 0.0345, 0.0345,  ..., 0.0345, 0.0345, 0.0345], 与dy1一致

 # print("line1_end_y[..., 0]",line1_end_y[..., 0])
    # print("line1_start_y[..., 0]",line1_start_y[..., 0])
    # print("du1:", du1)  # tensor([[0., 0., 0.,  ..., 0., 0., 0.],
    #
    # print("line2_end_y[..., 0]",line2_end_y[..., 0])
    # print("line2_start_y[..., 0]",line2_start_y[..., 0])
    # print("du2:", du2)  # tensor([[0., 0., 0.,  ..., 0., 0., 0.],

# print("line2_start[..., 0]:", line2_start[..., 0])
    # print("line1_start[..., 0]:", line1_start[..., 0])
    # print("(line2_start[..., 0] - line1_start[...,  0]):",(line2_start[..., 0] - line1_start[...,  0]))
    # print("(line2_start[..., 0] - line1_start[...,  0]):* dy2",(line2_start[..., 0] - line1_start[...,  0]) * dy2)
    # print("t1:\n", t1)

    # print("line2_start_y[..., 0]:", line2_start_y[..., 0])
    # print("line1_start_y[..., 0]:", line1_start_y[..., 0])
    # print("(line2_start_y[..., 0] - line1_start_y[...,  0]):",(line2_start_y[..., 0] - line1_start_y[...,  0]))
    # print("t2:\n",t2)



 # stop_mask = dx2 * du2 < 0  # 如果车辆在两个方向上的增量为负，说明他车已经停止了  不完全正确，在现有坐标系下的右上和左下区域存在误判
    # parallel_mask = torch.logical_or(parallel_mask, stop_mask)
    # # print("parallel_mask:\n", parallel_mask)