import torch
from Analyze.Utils import calculate_propagation_latency

# 486 482
def mcf_distance_allocation(servers, users, connect, latency):
    # 提取批次信息
    batch_size, num_users, _ = users.shape
    num_servers = servers.size(1)
    device = users.device

    # 提取用户资源需求 (batch_size, num_users, 4)
    users_need = users[:, :, 2:6]

    # 初始化剩余资源 (batch_size, num_servers, 4)
    remain_capacity = servers[:, :, 3:7].clone()

    # 初始化分配记录 (batch_size, num_users)
    user_allocated = torch.full((batch_size, num_users), -1, dtype=torch.long, device=device)

    # 记录边缘服务器分配到的用户数量
    server_allocated_user_flag = torch.zeros(batch_size, num_servers, dtype=torch.bool, device=servers.device)

    # 计算每个用户的总资源需求 (batch_size, num_users)
    total_users_need_demand = torch.sum(users_need, dim=2)
    # 按需求升序排序（需求小的优先）(batch_size, num_users)
    sorted_user_indices = torch.argsort(total_users_need_demand, dim=1)

    for user_order in range(num_users):
        # 获取当前要处理的用户索引 (batch_size,)
        i = sorted_user_indices[:, user_order]

        batch_indices = torch.arange(batch_size, device=device)

        if i.dim() == 0:
            i = i.unsqueeze(0).expand(batch_size)

        # 计算有效服务器条件
        connect_cond = connect[batch_indices, i]  # (batch_size, num_servers)

        resource_cond = torch.all(
            remain_capacity >= users_need[batch_indices, i, :].unsqueeze(1),  # 广播到 (batch_size, num_servers, 4)
            dim=2
        )
        combined_cond = connect_cond & resource_cond  # (batch_size, num_servers)

        # 为每个批次选择一个有效服务器
        valid_servers = []
        for b in range(batch_size):
            server_ids = torch.nonzero(combined_cond[b], as_tuple=True)[0]
            if server_ids.numel() > 0:
                # 计算这些服务器的总剩余资源
                activated = server_allocated_user_flag[b, server_ids].float()
                # 2. 构造排序键（激活状态优先）
                delay = latency[b, i[b], server_ids]  # (n,)
                # 综合激活状态、传播延迟来打分（得分越大越好）
                sort_key = activated.float() * 1e6 - delay
                # 3. 按排序键降序排列
                sorted_indices = torch.argsort(sort_key, descending=True)
                sorted_server_ids = server_ids[sorted_indices]
                chosen = sorted_server_ids[0]
                valid_servers.append(chosen)
            else:
                valid_servers.append(torch.tensor(-1, device=device))
        chosen_server_ids = torch.stack(valid_servers)  # (batch_size,)

        # 标记有效分配
        valid_mask = (chosen_server_ids != -1)  # (batch_size,)

        if valid_mask.any():
            # 有效分配的索引
            valid_batch_indices = torch.nonzero(valid_mask, as_tuple=True)[0]  # (k,)

            # 有效分配的服务器ID
            valid_server_ids = chosen_server_ids[valid_mask]  # (k,)

            # 更新剩余资源
            remain_capacity[valid_batch_indices, valid_server_ids, :] -= users_need[valid_batch_indices, i[valid_mask], :]

            # 更新分配记录
            user_allocated[valid_batch_indices, i[valid_mask]] = valid_server_ids
            server_allocated_user_flag[valid_batch_indices, valid_server_ids] = True

    # 计算已分配的用户比例和已激活的服务器比例
    allocated_users_num = (user_allocated != -1).sum(dim=1).float()
    allocated_user_ratio = allocated_users_num / num_users
    active_servers_ratio = server_allocated_user_flag.sum(dim=1).float() / server_allocated_user_flag.size(1)

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

    # 计算平均传播延迟
    propagation_delay_aver = calculate_propagation_latency(user_allocated, latency)


    return allocated_users_num, allocated_user_ratio, active_servers_ratio, capacity_used_ratio, propagation_delay_aver

