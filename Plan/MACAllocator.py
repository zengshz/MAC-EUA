import torch
import torch.nn as nn
from torch.distributions import Categorical
from Analyze.Utils import calculate_propagation_latency

class Encoder(nn.Module):

    def __init__(self, d_model, dropout, num_heads, edge_dim, user_feature_dim, server_feature_dim, spatial_raw_dim):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.edge_dim = edge_dim
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
        self.edge_proj = nn.Sequential(
            nn.Linear(spatial_raw_dim, edge_dim),
            nn.ReLU(),
            nn.LayerNorm(edge_dim),
            nn.Dropout(dropout)
        )
        self.user_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.server_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.user_dummy_kv = nn.Parameter(torch.zeros(1, 1, d_model))
        self.server_dummy_kv = nn.Parameter(torch.zeros(1, 1, d_model))
        self.user_norm = nn.LayerNorm(d_model)
        self.server_norm = nn.LayerNorm(d_model)
        self.user_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        self.server_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        if edge_dim != d_model:
            self.edge_to_d = nn.Linear(edge_dim, d_model)
        else:
            self.edge_to_d = nn.Identity()

    def forward(self, users, servers, connect):
        B, U, _ = users.shape
        S = servers.shape[1]
        user_h = self.user_proj(users)
        server_h = self.server_proj(servers)
        u_pos = users[..., :2].unsqueeze(2)
        s_pos = servers[..., :2].unsqueeze(1)
        s_radius = servers[..., 2].unsqueeze(1).unsqueeze(-1).clamp_min(1e-3)
        delta = u_pos - s_pos
        dist = torch.norm(delta, dim=-1, keepdim=True)
        dist_norm = dist / s_radius
        spatial_raw = torch.cat([delta, dist_norm], dim=-1)
        edge_emb = self.edge_proj(spatial_raw)
        connect_mask = ~connect.bool()
        user_dummy_kv = self.user_dummy_kv.expand(B, 1, self.d_model)
        server_kv_for_user = torch.cat([server_h, user_dummy_kv], dim=1)
        connect_mask_user_ext = torch.nn.functional.pad(connect_mask, (0, 1), value=False)
        attn_mask_user = connect_mask_user_ext.repeat_interleave(self.num_heads, dim=0)
        user_attn_out, _ = self.user_attn(
            query=user_h,
            key=server_kv_for_user,
            value=server_kv_for_user,
            attn_mask=attn_mask_user,
            need_weights=False
        )
        user_attn_out = torch.nan_to_num(user_attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        user_h = self.user_norm(user_h + user_attn_out)
        user_ff_out = torch.nan_to_num(self.user_ff(user_h), nan=0.0, posinf=0.0, neginf=0.0)
        user_h = user_h + user_ff_out
        connect_mask_s = connect_mask.permute(0, 2, 1)
        server_dummy_kv = self.server_dummy_kv.expand(B, 1, self.d_model)
        user_kv_for_server = torch.cat([user_h, server_dummy_kv], dim=1)
        connect_mask_server_ext = torch.nn.functional.pad(connect_mask_s, (0, 1), value=False)
        attn_mask_server = connect_mask_server_ext.repeat_interleave(self.num_heads, dim=0)
        server_attn_out, _ = self.server_attn(
            query=server_h,
            key=user_kv_for_server,
            value=user_kv_for_server,
            attn_mask=attn_mask_server,
            need_weights=False
        )
        server_attn_out = torch.nan_to_num(server_attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        server_h = self.server_norm(server_h + server_attn_out)  # residual + norm
        server_ff_out = torch.nan_to_num(self.server_ff(server_h), nan=0.0, posinf=0.0, neginf=0.0)
        server_h = server_h + server_ff_out
        spatial_enc = self.edge_to_d(edge_emb)
        return user_h, server_h, spatial_enc


class ResourceAllocatorDecoder(nn.Module):

    def __init__(self, d_model, num_heads, dropout, server_state_dim):
        super().__init__()
        self.d_model = d_model
        self.state_proj = nn.Linear(server_state_dim, d_model)
        self.server_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.server_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
        score_dim = max(16, min(64, d_model // 2))
        self.user_score = nn.Linear(d_model, score_dim, bias=False)
        self.server_score = nn.Linear(d_model, score_dim, bias=False)
        self.edge_score = nn.Linear(d_model, 1, bias=False)
        self.score_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_enc, server_enc, spatial_enc, connect, users, servers, policy, device):
        B, U, _ = users.shape
        S = servers.shape[1]
        needs = users[..., 2:6]
        init_cap = servers[..., 3:7].clone()
        cap = init_cap
        active = torch.zeros(B, S, dtype=torch.bool, device=device)
        allocated = torch.full((B, U), -1, dtype=torch.long, device=device)
        unassigned = torch.ones(B, U, dtype=torch.bool, device=device)
        logp_accum = torch.zeros(B, device=device)
        can_fulfill = (cap.unsqueeze(1) >= needs.unsqueeze(2)).all(dim=-1)  # [B,U,S]
        eligible = connect & can_fulfill & unassigned.unsqueeze(-1)  # [B,U,S]
        while eligible.any():
            dynamic_state = torch.cat([cap, active.unsqueeze(-1).float()], dim=-1)
            state_emb = self.state_proj(dynamic_state)
            server_input = self.norm(server_enc + state_emb)
            attn_out, _ = self.server_attention(
                query=server_input,
                key=server_input,
                value=server_input,
                need_weights=False
            )
            server_intermediate = self.norm(server_input + attn_out)
            server_current_emb = self.norm(server_intermediate + self.server_ff(server_intermediate))
            u_score = self.user_score(user_enc)
            s_score = self.server_score(server_current_emb)
            us_score = torch.einsum('bur,bsr->bus', u_score, s_score)
            edge_term = self.edge_score(spatial_enc).squeeze(-1)
            logits = us_score + edge_term + self.score_bias
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


class MACAllocator(nn.Module):

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
            num_heads,
            edge_dim,
            user_feature_dim,
            server_feature_dim,
            spatial_raw_dim
        )
        self.decoder = ResourceAllocatorDecoder(
            d_model,
            num_heads,
            dropout,
            server_state_dim
        )

    def forward(self, servers, users, connect, p_distance):
        B, U, _ = users.shape
        S = servers.shape[1]
        user_enc, server_enc, spatial_enc = self.encoder(users=users, servers=servers, connect=connect)
        logp_accum, allocated, cap, active = self.decoder(
            user_enc=user_enc,
            server_enc=server_enc,
            spatial_enc=spatial_enc,
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