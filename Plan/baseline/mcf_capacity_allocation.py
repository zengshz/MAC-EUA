import torch
from Analyze.Utils import calculate_propagation_latency

# 486 482
def mcf_capacity_allocation(servers, users, connect, latency):
    batch_size, num_users, _ = users.shape
    num_servers = servers.size(1)
    device = users.device

    users_need = users[:, :, 2:6]

    remain_capacity = servers[:, :, 3:7].clone()

    user_allocated = torch.full((batch_size, num_users), -1, dtype=torch.long, device=device)

    server_allocated_user_flag = torch.zeros(batch_size, num_servers, dtype=torch.bool, device=servers.device)

    total_users_need_demand = torch.sum(users_need, dim=2)
    sorted_user_indices = torch.argsort(total_users_need_demand, dim=1)

    for user_order in range(num_users):
        i = sorted_user_indices[:, user_order]

        batch_indices = torch.arange(batch_size, device=device)

        if i.dim() == 0:
            i = i.unsqueeze(0).expand(batch_size)

        connect_cond = connect[batch_indices, i]

        resource_cond = torch.all(
            remain_capacity >= users_need[batch_indices, i, :].unsqueeze(1),
            dim=2
        )
        combined_cond = connect_cond & resource_cond

        valid_servers = []
        for b in range(batch_size):
            server_ids = torch.nonzero(combined_cond[b], as_tuple=True)[0]
            if server_ids.numel() > 0:
                activated = server_allocated_user_flag[b, server_ids].float()
                remaining = remain_capacity[b, server_ids, :].sum(dim=1)
                sort_key = activated * 1e6 + remaining
                sorted_indices = torch.argsort(sort_key, descending=True)
                sorted_server_ids = server_ids[sorted_indices]
                chosen = sorted_server_ids[0]
                valid_servers.append(chosen)
            else:
                valid_servers.append(torch.tensor(-1, device=device))
        chosen_server_ids = torch.stack(valid_servers)
        valid_mask = (chosen_server_ids != -1)

        if valid_mask.any():
            valid_batch_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

            valid_server_ids = chosen_server_ids[valid_mask]

            remain_capacity[valid_batch_indices, valid_server_ids, :] -= users_need[valid_batch_indices, i[valid_mask], :]

            user_allocated[valid_batch_indices, i[valid_mask]] = valid_server_ids
            server_allocated_user_flag[valid_batch_indices, valid_server_ids] = True

    allocated_users_num = (user_allocated != -1).sum(dim=1).float()
    allocated_user_ratio = allocated_users_num / num_users
    active_servers_ratio = server_allocated_user_flag.sum(dim=1).float() / server_allocated_user_flag.size(1)

    propagation_delay_aver = calculate_propagation_latency(user_allocated, latency)


    return allocated_users_num, allocated_user_ratio, active_servers_ratio, propagation_delay_aver

