import torch
import torch.nn as nn
from torch.distributions import Categorical
from Analyze.Utils import calculate_propagation_latency


class Encoder(nn.Module):
    def __init__(self, d_model, dropout, user_feature_dim, server_feature_dim):
        super().__init__()
        self.d_model = d_model
        self.user_proj = nn.Sequential(
            nn.Linear(user_feature_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.server_proj = nn.Sequential(
            nn.Linear(server_feature_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def forward(self, users, servers):
        B, U, _ = users.shape
        S = servers.shape[1]
        user_h = self.user_proj(users)
        server_h = self.server_proj(servers)
        return user_h, server_h


class ResourceAllocatorDecoder(nn.Module):
    def __init__(self, d_model, server_state_dim):
        super().__init__()
        self.d_model = d_model
        self.state_proj = nn.Linear(server_state_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        score_dim = max(16, min(64, d_model // 2))
        self.user_score = nn.Linear(d_model, score_dim, bias=False)
        self.server_score = nn.Linear(d_model, score_dim, bias=False)
        self.score_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_enc, server_enc, connect, users, servers, policy, device):
        B, U, _ = users.shape
        S = servers.shape[1]
        needs = users[..., 2:6]
        init_cap = servers[..., 3:7].clone()
        cap = init_cap
        active = torch.zeros(B, S, dtype=torch.bool, device=device)
        allocated = torch.full((B, U), -1, dtype=torch.long, device=device)
        unassigned = torch.ones(B, U, dtype=torch.bool, device=device)
        logp_accum = torch.zeros(B, device=device)
        can_fulfill = (cap.unsqueeze(1) >= needs.unsqueeze(2)).all(dim=-1)
        eligible = connect & can_fulfill & unassigned.unsqueeze(-1)
        while eligible.any():
            dynamic_state = torch.cat([cap, active.unsqueeze(-1).float()], dim=-1)
            state_emb = self.state_proj(dynamic_state)
            server_current_emb = self.norm(server_enc + state_emb)
            u_score = self.user_score(user_enc)
            s_score = self.server_score(server_current_emb)
            us_score = torch.einsum('bur,bsr->bus', u_score, s_score)
            logits = us_score + self.score_bias
            logits_su = logits.transpose(1, 2)
            logits_su = logits_su.masked_fill(~eligible.transpose(1, 2), float("-1e9"))
            has_any = eligible.transpose(1, 2).any(dim=-1)
            BS = B * S
            flat_logits = logits_su.reshape(BS, U)
            flat_has_any = has_any.reshape(BS)
            flat_picks = torch.full((BS,), -1, device=device, dtype=torch.long)
            flat_pick_log = torch.zeros(BS, device=device)
            valid_idx = flat_has_any.nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 0:
                if policy == "sample":
                    dist = Categorical(logits=flat_logits[valid_idx])
                    sampled = dist.sample()
                    flat_picks[valid_idx] = sampled
                    flat_pick_log[valid_idx] = dist.log_prob(sampled)
                elif policy == "greedy":
                    flat_picks[valid_idx] = flat_logits[valid_idx].argmax(dim=-1)
                    flat_pick_log[valid_idx] = 0.0
            picks = flat_picks.view(B, S)
            pick_log = flat_pick_log.view(B, S)
            picks = torch.where(has_any, picks, torch.full_like(picks, -1))
            picked_u_idx = picks.clamp(min=0)
            user_selected_count = torch.zeros(B, U, device=device, dtype=torch.long)
            user_selected_count.scatter_add_(1, picked_u_idx, has_any.long())
            accepted = has_any.clone()
            chosen_b, chosen_s = torch.where(has_any)
            if chosen_b.numel() > 0:
                chosen_u = picks[chosen_b, chosen_s]
                is_conflict_pick = user_selected_count[chosen_b, chosen_u] > 1
                if is_conflict_pick.any():
                    cb = chosen_b[is_conflict_pick]
                    cs = chosen_s[is_conflict_pick]
                    cu = chosen_u[is_conflict_pick]
                    pair_id = cb * U + cu
                    order = torch.argsort(pair_id)
                    pair_id = pair_id[order]
                    cb = cb[order]
                    cs = cs[order]
                    cu = cu[order]
                    cscore = logits[cb, cu, cs]
                    _, inverse = torch.unique_consecutive(pair_id, return_inverse=True)
                    num_groups = int(inverse.max().item()) + 1
                    group_max = torch.full((num_groups,), -float('inf'), device=device, dtype=cscore.dtype)
                    group_max.scatter_reduce_(0, inverse, cscore, reduce='amax', include_self=True)
                    is_group_max = cscore == group_max[inverse]
                    large_idx = torch.full_like(inverse, fill_value=inverse.numel())
                    cand_idx = torch.where(is_group_max, torch.arange(inverse.numel(), device=device), large_idx)
                    first_winner_pos = torch.full((num_groups,), fill_value=inverse.numel(),
                                                  device=device, dtype=torch.long)
                    first_winner_pos.scatter_reduce_(0, inverse, cand_idx, reduce='amin', include_self=True)
                    winner_mask = torch.zeros_like(accepted)
                    winner_mask[cb[first_winner_pos], cs[first_winner_pos]] = True
                    conflict_mask = torch.zeros_like(accepted)
                    conflict_mask[cb, cs] = True
                    accepted = accepted & (~conflict_mask | winner_mask)
            if not accepted.any():
                break
            b_idx_acc, s_idx_acc = torch.where(accepted)
            u_idx_acc = picks[b_idx_acc, s_idx_acc]
            allocated[b_idx_acc, u_idx_acc] = s_idx_acc
            active[b_idx_acc, s_idx_acc] = True
            unassigned[b_idx_acc, u_idx_acc] = False
            cap[b_idx_acc, s_idx_acc] -= needs[b_idx_acc, u_idx_acc]
            logp_accum += (pick_log * accepted.float()).sum(dim=1)
            touched_b = torch.unique(b_idx_acc)
            can_fulfill[touched_b] = (cap[touched_b].unsqueeze(1) >= needs[touched_b].unsqueeze(2)).all(dim=-1)
            eligible[touched_b] = connect[touched_b] & can_fulfill[touched_b] & unassigned[touched_b].unsqueeze(-1)
        return logp_accum, allocated, cap, active


class MACAllocator_Ablate_NOT(nn.Module):

    def __init__(
            self,
            d_model,
            num_heads,
            dropout,
            edge_dim,
            user_feature_dim,
            server_feature_dim,
            spatial_raw_dim,
            server_state_dim,
            device,
            MAX_PROPAGATION_LATENCY,
            policy
    ):
        super().__init__()
        self.device = torch.device(device)
        self.policy = policy
        self.MAX_PROPAGATION_LATENCY = MAX_PROPAGATION_LATENCY
        self.encoder = Encoder(
            d_model,
            dropout,
            user_feature_dim,
            server_feature_dim
        )
        self.decoder = ResourceAllocatorDecoder(
            d_model,
            server_state_dim
        )

    def forward(self, servers, users, connect, p_distance):
        B, U, _ = users.shape
        S = servers.shape[1]
        user_enc, server_enc = self.encoder(users=users, servers=servers)
        logp_accum, allocated, cap, active = self.decoder(
            user_enc=user_enc,
            server_enc=server_enc,
            connect=connect,
            users=users,
            servers=servers,
            policy=self.policy,
            device=self.device
        )
        alloc_num = (allocated != -1).sum(dim=1).float()
        alloc_ratio = alloc_num / float(U)
        active_ratio = active.sum(dim=1).float() / float(S)
        p_lat = calculate_propagation_latency(allocated, p_distance)
        p_lat_normalized = p_lat / self.MAX_PROPAGATION_LATENCY
        reward = alloc_ratio - 0.01 * p_lat_normalized
        return -reward, logp_accum, alloc_num, alloc_ratio, active_ratio, p_lat