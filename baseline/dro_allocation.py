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


def _argmin_one_per_row(mask: torch.Tensor, score: torch.Tensor, device: torch.device):
    out = torch.full((mask.shape[0],), -1, dtype=torch.long, device=device)
    row_has = mask.any(dim=1)
    if row_has.any():
        masked_score = score[row_has].masked_fill(~mask[row_has], 1e9)
        out[row_has] = masked_score.argmin(dim=1)
    return out

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

        capacity_used = 1 - remain_capacity / servers[:, :, 3:7].clamp_min(1e-6)
        used_mean = capacity_used.mean(dim=-1)
        mu = used_mean.mean(dim=1)
        std = used_mean.std(dim=1, unbiased=False) if used_mean.shape[1] >= 2 else torch.zeros_like(mu)

        is_ml = mu <= 0.09
        is_mm = (mu > 0.09) & (mu <= 0.22)
        is_sl = std <= 0.03
        is_sm = (std > 0.03) & (std <= 0.12)

        omega = torch.full((batch_size,), VH, device=device, dtype=used_mean.dtype)
        omega = torch.where(is_ml & is_sl, torch.full_like(omega, EL), omega)
        omega = torch.where(is_ml & is_sm, torch.full_like(omega, VL), omega)
        omega = torch.where(is_ml & (~is_sl & ~is_sm), torch.full_like(omega, VL), omega)
        omega = torch.where(is_mm & is_sl, torch.full_like(omega, M), omega)
        omega = torch.where(is_mm & is_sm, torch.full_like(omega, L), omega)
        omega = torch.where(is_mm & (~is_sl & ~is_sm), torch.full_like(omega, VL), omega)
        omega = torch.where((~is_ml & ~is_mm) & is_sl, torch.full_like(omega, EH), omega)
        omega = torch.where((~is_ml & ~is_mm) & is_sm, torch.full_like(omega, VH), omega)
        omega = torch.where((~is_ml & ~is_mm) & (~is_sl & ~is_sm), torch.full_like(omega, H), omega)

        zi = torch.where(server_allocated_user_flag, torch.full_like(used_mean, 10.0), torch.zeros_like(used_mean))
        c_raw = (10.0 - zi).abs()
        C = torch.where(zi < 10.0, c_raw * gamma, c_raw)
        Bv = used_mean

        C_masked_min = C.masked_fill(~combined_cond, float("inf"))
        C_masked_max = C.masked_fill(~combined_cond, float("-inf"))
        C_min = C_masked_min.min(dim=1, keepdim=True).values
        C_max = C_masked_max.max(dim=1, keepdim=True).values
        Cn = (C - C_min) / (C_max - C_min).clamp_min(1e-6)

        B_masked_min = Bv.masked_fill(~combined_cond, float("inf"))
        B_masked_max = Bv.masked_fill(~combined_cond, float("-inf"))
        B_min = B_masked_min.min(dim=1, keepdim=True).values
        B_max = B_masked_max.max(dim=1, keepdim=True).values
        Bn = (Bv - B_min) / (B_max - B_min).clamp_min(1e-6)

        S = omega.unsqueeze(1) * Cn + (1 - omega).unsqueeze(1) * Bn
        chosen = _argmin_one_per_row(combined_cond, S, device=device)

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

    propagation_delay_aver = calculate_propagation_latency(user_allocated, latency)

    return (
        allocated_users_num,
        allocated_user_ratio,
        active_servers_ratio,
        # capacity_used_ratio,
        propagation_delay_aver,
        # load_delay_aver,
        # synergy_delay_aver,
    )
