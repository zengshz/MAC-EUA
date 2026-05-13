import torch
import torch.nn as nn
from torch.distributions import Categorical
from Analyze.Utils import calculate_propagation_latency


# --------------------------
# encoder (用户 <-> 服务器)
# --------------------------
class Encoder(nn.Module):
    """

    Inputs:
        users:  [B, U, 6]   (X,Y, cpu,ram,storage,bandwidth)
        servers:[B, S, 7]   (X,Y, RADIUS, cpu,ram,storage,bandwidth)
        connect:[B, U, S]   bool adjacency mask (user u can connect to server s)
    Outputs:
        user_enc: [B, U, D]
        server_enc: [B, S, D]
        spatial_enc: [B, U, S, D]  (projected edge features)

    """

    def __init__(self, d_model, dropout, num_heads, edge_dim, user_feature_dim, server_feature_dim, spatial_raw_dim):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.edge_dim = edge_dim

        # initial projections for nodes
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

        # project raw edge spatial (Δx,Δy,dist_norm) -> edge_emb
        self.edge_proj = nn.Sequential(
            nn.Linear(spatial_raw_dim, edge_dim),
            nn.ReLU(),
            nn.LayerNorm(edge_dim),
            nn.Dropout(dropout)
        )

        # 一次 cross-attention: user queries server
        self.user_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 一次 cross-attention: server queries user
        self.server_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 为cross-attention引入dummy候选，确保每个query至少有一个可选key
        # 这样即使connect全False，也不会出现全mask softmax
        self.user_dummy_kv = nn.Parameter(torch.zeros(1, 1, d_model))
        self.server_dummy_kv = nn.Parameter(torch.zeros(1, 1, d_model))

        # Layer normalization
        self.user_norm = nn.LayerNorm(d_model)
        self.server_norm = nn.LayerNorm(d_model)

        # 轻量 feedforward
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

        # edge to d_model projection (预先定义，避免每次forward创建)
        if edge_dim != d_model:
            self.edge_to_d = nn.Linear(edge_dim, d_model)
        else:
            self.edge_to_d = nn.Identity()

    def forward(self, users, servers, connect):
        """
        users: [B,U,6]
        servers: [B,S,7]
        connect: [B,U,S] bool
        """
        B, U, _ = users.shape
        S = servers.shape[1]

        # 1. Initial node embeddings (User MLP + Server MLP)
        user_h = self.user_proj(users)  # [B,U,D]
        server_h = self.server_proj(servers)  # [B,S,D]

        # 2. Compute spatial edge features (Δx,Δy,dist_norm)
        # 使用广播避免显式expand，减少内核调度与中间张量开销
        u_pos = users[..., :2].unsqueeze(2)  # [B,U,1,2]
        s_pos = servers[..., :2].unsqueeze(1)  # [B,1,S,2]
        s_radius = servers[..., 2].unsqueeze(1).unsqueeze(-1).clamp_min(1e-3)  # [B,1,S,1]

        delta = u_pos - s_pos  # [B,U,S,2]
        dist = torch.norm(delta, dim=-1, keepdim=True)  # [B,U,S,1]
        dist_norm = dist / s_radius  # [B,U,S,1]
        spatial_raw = torch.cat([delta, dist_norm], dim=-1)  # [B,U,S,3]

        # project edge features to edge_emb
        edge_emb = self.edge_proj(spatial_raw)  # [B,U,S,edge_dim]

        # 3. 一次 cross-attention: user queries server
        # user作为query，server作为key/value
        connect_mask = ~connect.bool()  # [B,U,S]，True表示不可连接（需要mask）

        # 结构性优化：追加dummy server，保证每个user query至少有一个可选key
        # key/value: [B,S+1,D]，最后一列是dummy
        user_dummy_kv = self.user_dummy_kv.expand(B, 1, self.d_model)
        server_kv_for_user = torch.cat([server_h, user_dummy_kv], dim=1)

        # 对dummy列永不mask，避免全mask行（pad比cat+zeros更轻）
        connect_mask_user_ext = torch.nn.functional.pad(connect_mask, (0, 1), value=False)  # [B,U,S+1]

        # 直接沿batch维复制head，避免4D expand+reshape路径
        attn_mask_user = connect_mask_user_ext.repeat_interleave(self.num_heads, dim=0)  # [B*H,U,S+1]

        user_attn_out, _ = self.user_attn(
            query=user_h,  # [B,U,D]
            key=server_kv_for_user,  # [B,S+1,D]
            value=server_kv_for_user,  # [B,S+1,D]
            attn_mask=attn_mask_user,  # [B*num_heads,U,S+1]
            need_weights=False
        )
        user_attn_out = torch.nan_to_num(user_attn_out, nan=0.0, posinf=0.0, neginf=0.0)

        user_h = self.user_norm(user_h + user_attn_out)  # residual + norm
        user_ff_out = torch.nan_to_num(self.user_ff(user_h), nan=0.0, posinf=0.0, neginf=0.0)
        user_h = user_h + user_ff_out

        # 4. 一次 cross-attention: server queries user
        # server作为query，user作为key/value
        connect_mask_s = connect_mask.permute(0, 2, 1)  # [B,S,U]，True表示不可连接

        # 结构性优化：追加dummy user，保证每个server query至少有一个可选key
        server_dummy_kv = self.server_dummy_kv.expand(B, 1, self.d_model)
        user_kv_for_server = torch.cat([user_h, server_dummy_kv], dim=1)  # [B,U+1,D]

        connect_mask_server_ext = torch.nn.functional.pad(connect_mask_s, (0, 1), value=False)  # [B,S,U+1]

        # 直接沿batch维复制head，避免4D expand+reshape路径
        attn_mask_server = connect_mask_server_ext.repeat_interleave(self.num_heads, dim=0)  # [B*H,S,U+1]

        server_attn_out, _ = self.server_attn(
            query=server_h,  # [B,S,D]
            key=user_kv_for_server,  # [B,U+1,D]
            value=user_kv_for_server,  # [B,U+1,D]
            attn_mask=attn_mask_server,  # [B*num_heads,S,U+1]
            need_weights=False
        )
        server_attn_out = torch.nan_to_num(server_attn_out, nan=0.0, posinf=0.0, neginf=0.0)

        server_h = self.server_norm(server_h + server_attn_out)  # residual + norm
        server_ff_out = torch.nan_to_num(self.server_ff(server_h), nan=0.0, posinf=0.0, neginf=0.0)
        server_h = server_h + server_ff_out

        # 5. Project edge_emb to d_model for decoder
        spatial_enc = self.edge_to_d(edge_emb)  # [B,U,S,D]

        return user_h, server_h, spatial_enc


# --------------------------
# 模块2：解码器（处理多轮分配决策）
# --------------------------
class ResourceAllocatorDecoder(nn.Module):


    def __init__(self, d_model, server_state_dim):
        super().__init__()
        self.d_model = d_model

        self.state_proj = nn.Linear(server_state_dim, d_model)

        self.norm = nn.LayerNorm(d_model)

        # 分解式打分头：避免构造 fused [B,U,S,D] 大张量
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

        # 可行性缓存：后续按受影响batch增量更新
        can_fulfill = (cap.unsqueeze(1) >= needs.unsqueeze(2)).all(dim=-1)  # [B,U,S]
        eligible = connect & can_fulfill & unassigned.unsqueeze(-1)  # [B,U,S]

        while eligible.any():

            dynamic_state = torch.cat([cap, active.unsqueeze(-1).float()], dim=-1)
            state_emb = self.state_proj(dynamic_state)
            # server embedding 融合动态状态
            server_current_emb = self.norm(server_enc + state_emb)

            # =====================================================
            # 2. 计算logits（分解式，避免 fused [B,U,S,D]）
            # =====================================================
            u_score = self.user_score(user_enc)  # [B,U,R]
            s_score = self.server_score(server_current_emb)  # [B,S,R]
            us_score = torch.einsum('bur,bsr->bus', u_score, s_score)  # [B,U,S]
            edge_term = self.edge_score(spatial_enc).squeeze(-1)  # [B,U,S]
            logits = us_score + edge_term + self.score_bias  # [B,U,S]

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

            # 轻量冲突解决：仅在被选中的索引上处理，避免构造冲突×S大矩阵
            picked_u_idx = picks.clamp(min=0)

            user_selected_count = torch.zeros(B, U, device=device, dtype=torch.long)
            user_selected_count.scatter_add_(1, picked_u_idx, has_any.long())

            accepted = has_any.clone()  # [B, S]

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

            # 增量更新 eligible：仅更新受影响的 batch 切片
            touched_b = torch.unique(b_idx_acc)
            can_fulfill[touched_b] = (cap[touched_b].unsqueeze(1) >= needs[touched_b].unsqueeze(2)).all(dim=-1)
            eligible[touched_b] = connect[touched_b] & can_fulfill[touched_b] & unassigned[touched_b].unsqueeze(-1)

        return logp_accum, allocated, cap, active


# ----------------------------
# 主模块：轻量多智能体资源分配器  多智能体资源分配器
# ----------------------------
class MACAllocator_Ablate_Encoder(nn.Module):
    """
        输入张量规格：
            servers: [B, S, 7]  (X, Y, RADIUS, cpu, ram, storage, bandwidth)
            users:   [B, U, 6]  (X, Y, cpu, ram, storage, bandwidth)
            connect: [B, U, S]  布尔，用户-服务器可连接矩阵
            p_distance: [B, U, S]  浮点，用户到服务器传播距离

        前向返回：
            -loss:        [B]   建议用 loss = -reward 求均值训练
            logp_sum:     [B]   本 batch 各用户对数概率之和（策略梯度用）
            alloc_num:    [B]   成功接入的用户数量
            alloc_ratio:  [B]   成功接入比例
            active_ratio: [B]   被激活服务器占比
            p_lat:        [B]   平均传播时延
            sy_lat:       [B]   平均协作时延
        """

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

        # 编码器 + 解码器
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
            server_state_dim
        )

    def forward(self, servers, users, connect, p_distance):
        """
           输入：
           servers: [B, S, 7] (服务器特征) X(坐标),Y(坐标),RADIUS,cpu ram storage bandwidth
           users: [B, U, 6] (用户特征) X(坐标),Y(坐标),cpu ram storage bandwidth
           connect: [B, U, S] (连接矩阵) 边缘服务器的覆盖用户信息，行为用户，列为服务器，总长度为用户数*服务器数，数据类型设置为布尔值
           p_distance: [B, U, S] (传播距离矩阵） 行为用户，列为服务器，存有用户到该服务器的传播距离
        """

        B, U, _ = users.shape
        S = servers.shape[1]

        # 1. 编码器：处理静态输入，输出上下文嵌入
        user_enc, server_enc, spatial_enc = self.encoder(users=users, servers=servers, connect=connect)

        # 2. 解码器：多轮分配决策，输出核心结果
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

        # 3. 计算奖励与统计指标（与原模型逻辑完全一致）
        # 分配率
        alloc_num = (allocated != -1).sum(dim=1).float()  # [B]
        alloc_ratio = alloc_num / float(U)  # [B]
        # 服务器激活率
        active_ratio = active.sum(dim=1).float() / float(S)  # [B]

        # 传播时延
        p_lat = calculate_propagation_latency(allocated, p_distance)  # [B]（原始时延）
        p_lat_normalized = p_lat / self.MAX_PROPAGATION_LATENCY  # [B]（归一化时延）
        reward = alloc_ratio - p_lat_normalized


        return -reward, logp_accum, alloc_num, alloc_ratio, active_ratio, p_lat


