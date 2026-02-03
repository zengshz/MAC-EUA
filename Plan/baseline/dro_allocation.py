import torch
from Analyze.Utils import calculate_propagation_latency

EL, VL, L, M, H, VH, EH = 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1
omega_dic = {'ML': {"SL": EL, "SM": VL, "SH": VL},
             'MM': {"SL": M, "SM": L, "SH": VL},
             'MH': {"SL": EH, "SM": VH, "SH": H}}


def get_fuzzy_weight(mu, std):
    if mu <= 0.09:
        a = 'ML'
    elif 0.09 < mu <= 0.22:
        a = 'MM'
    else:
        a = 'MH'
    if std <= 0.03:
        b = 'SL'
    elif 0.03 < std <= 0.12:
        b = 'SM'
    else:
        b = 'SH'
    return omega_dic[a][b]


def dro_allocation(
        servers, users, connect, latency,
        gamma=1.5
):
    batch_size, num_users, _ = users.shape
    num_servers = servers.size(1)
    device = users.device

    users_need = users[:, :, 2:6]
    remain_capacity = servers[:, :, 3:7].clone()
    user_allocated = torch.full((batch_size, num_users), -1, dtype=torch.long, device=device)
    server_allocated_user_flag = torch.zeros(batch_size, num_servers, dtype=torch.bool, device=device)

    for i in range(num_users):
        batch_indices = torch.arange(batch_size, device=device)
        connect_cond = connect[batch_indices, i]
        resource_cond = torch.all(remain_capacity >= users_need[:, i:i + 1, :], dim=2)
        combined_cond = connect_cond & resource_cond

        valid_servers = []
        for b in range(batch_size):
            server_ids = torch.nonzero(combined_cond[b], as_tuple=True)[0]
            if server_ids.numel() > 0:
                capacity_used = 1 - remain_capacity[b, :, :4] / servers[b, :, 3:7]
                used_mean = capacity_used.mean(dim=1)
                mu = used_mean.mean().item()
                std = used_mean.std(unbiased=True).item() if len(used_mean) >= 2 else 0.0
                omega_j = get_fuzzy_weight(mu, std)

                C, Bv = [], []
                for sid in server_ids:
                    zi = 0 if not server_allocated_user_flag[b, sid] else 10
                    t, vj = 0, 10
                    c = abs(zi - (t + vj)) * (gamma if zi < t + vj else 1)
                    C.append(c)
                    Bv.append(used_mean[sid].item())

                C, Bv = torch.tensor(C, device=device), torch.tensor(Bv, device=device)
                # 归一化
                Cn = (C - C.min()) / ((C.max() - C.min()).clamp(min=1e-6))
                Bn = (Bv - Bv.min()) / ((Bv.max() - Bv.min()).clamp(min=1e-6))
                # 最终得分：越小越好
                S = omega_j * Cn + (1 - omega_j) * Bn

                best = torch.argmin(S)
                valid_servers.append(server_ids[best])
            else:
                valid_servers.append(torch.tensor(-1, device=device))

        chosen = torch.stack(valid_servers)
        mask = chosen != -1
        if mask.any():
            b_idx = torch.nonzero(mask, as_tuple=True)[0]
            s_idx = chosen[mask]
            remain_capacity[b_idx, s_idx, :] -= users_need[b_idx, i, :]
            user_allocated[b_idx, i] = s_idx
            server_allocated_user_flag[b_idx, s_idx] = True

    allocated_users_num = (user_allocated != -1).sum(dim=1).float()
    allocated_user_ratio = allocated_users_num / num_users
    active_servers_ratio = server_allocated_user_flag.sum(dim=1).float() / num_servers

    # 计算资源利用率：先按资源维度求和，再求资源维度的平均剩余比例
    # 1. 对剩余容量和原始容量应用掩码（排除不活跃服务器）
    masked_cap = remain_capacity.masked_fill(
        ~server_allocated_user_flag.unsqueeze(-1).expand(batch_size, num_servers, 4), value=0)  # 剩余容量，形状[B, S, 4]
    masked_servers = servers[:, :, 3:7].masked_fill(
        ~server_allocated_user_flag.unsqueeze(-1).expand(batch_size, num_servers, 4),
        value=0)  # 原始容量，形状[B, S, 4]

    # 2. 按服务器维度（dim=1）求和，得到每个资源维度的总剩余和总原始容量（形状[B, 4]）
    total_remain_cap = torch.sum(masked_cap, dim=1)  # 总剩余容量：[B, 4]
    total_original_cap = torch.sum(masked_servers, dim=1)  # 总原始容量：[B, 4]

    # 3. 计算每个资源维度的剩余比例（总剩余 / 总原始），处理除零情况
    per_dim_remain_ratio = torch.nan_to_num(
        torch.div(total_remain_cap, total_original_cap),
        nan=0.0  # 当总原始容量为0时，剩余比例视为0
    )

    # 4. 对资源维度（dim=1）求平均，得到平均剩余比例
    mean_remain_ratio = torch.mean(per_dim_remain_ratio, dim=1)  # 形状[B]

    # 5. 资源利用率 = 1 - 平均剩余比例
    capacity_used_ratio = 1 - mean_remain_ratio

    propagation_delay_aver = calculate_propagation_latency(user_allocated, latency)

    return (
        allocated_users_num,
        allocated_user_ratio,
        active_servers_ratio,
        capacity_used_ratio,
        propagation_delay_aver,
        # load_delay_aver,
        # synergy_delay_aver,
    )
