import torch
from Analyze.Utils import calculate_propagation_latency

def greedy_capacity_allocation(servers, users, connect, latency):

    batch_size, num_users, _ = users.shape
    num_servers = servers.size(1)
    device = users.device

    users_need = users[:, :, 2:6]

    remain_capacity = servers[:, :, 3:7].clone()

    user_allocated = torch.full((batch_size, num_users), -1, dtype=torch.long, device=device)

    server_allocated_user_flag = torch.zeros(batch_size, num_servers, dtype=torch.bool, device=servers.device)

    for i in range(num_users):
        batch_indices = torch.arange(batch_size, device=device)

        connect_cond = connect[batch_indices, i]

        resource_cond = torch.all(
            remain_capacity >= users_need[:, i:i + 1, :],
            dim=2
        )
        combined_cond = connect_cond & resource_cond
        valid_servers = []
        for b in range(batch_size):
            server_ids = torch.nonzero(combined_cond[b], as_tuple=True)[0]
            if server_ids.numel() > 0:
                total_remaining = torch.sum(remain_capacity[b, server_ids, :], dim=1)
                max_index = torch.argmax(total_remaining)
                valid_servers.append(server_ids[max_index])
            else:
                valid_servers.append(torch.tensor(-1, device=device))
        chosen_server_ids = torch.stack(valid_servers)

        valid_mask = (chosen_server_ids != -1)

        if valid_mask.any():
            valid_batch_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

            valid_server_ids = chosen_server_ids[valid_mask]

            remain_capacity[valid_batch_indices, valid_server_ids, :] -= users_need[valid_batch_indices, i, :]

            user_allocated[valid_batch_indices, i] = valid_server_ids

            server_allocated_user_flag[valid_batch_indices, valid_server_ids] = True

    allocated_users_num = (user_allocated != -1).sum(dim=1).float()
    allocated_user_ratio = allocated_users_num / num_users
    active_servers_ratio = server_allocated_user_flag.sum(dim=1).float() / server_allocated_user_flag.size(1)
    propagation_delay_aver = calculate_propagation_latency(user_allocated, latency)

    return allocated_users_num, allocated_user_ratio, active_servers_ratio, propagation_delay_aver