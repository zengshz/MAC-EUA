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
        u_pos = users[..., :2].unsqueeze(2).expand(B, U, S, 2)  # [B,U,S,2]
        s_pos = servers[..., :2].unsqueeze(1).expand(B, U, S, 2)
        s_radius = servers[..., 2].unsqueeze(1).unsqueeze(-1).expand(B, U, S, 1).clamp_min(1e-3)

        delta = u_pos - s_pos  # [B,U,S,2]
        dist = torch.norm(delta, dim=-1, keepdim=True)  # [B,U,S,1]
        dist_norm = dist / s_radius  # [B,U,S,1]
        spatial_raw = torch.cat([delta, dist_norm], dim=-1)  # [B,U,S,3]

        # project edge features to edge_emb
        edge_emb = self.edge_proj(spatial_raw)  # [B,U,S,edge_dim]

        # 3. 一次 cross-attention: user queries server
        # user作为query，server作为key/value
        connect_mask = ~connect.bool()  # [B,U,S]，True表示不可连接（需要mask）

        # 创建attention mask: [B*num_heads, U, S]
        # 对于batch_first=True的MultiheadAttention，attn_mask应该是bool类型
        # PyTorch会自动将True位置设为-inf
        attn_mask_user = connect_mask.unsqueeze(1).expand(B, self.num_heads, U, S)  # [B, num_heads, U, S]
        attn_mask_user = attn_mask_user.reshape(B * self.num_heads, U, S)  # [B*num_heads, U, S]

        user_attn_out, _ = self.user_attn(
            query=user_h,  # [B,U,D]
            key=server_h,  # [B,S,D]
            value=server_h,  # [B,S,D]
            attn_mask=attn_mask_user  # [B*num_heads, U, S]
        )
        user_h = self.user_norm(user_h + user_attn_out)  # residual + norm
        user_h = user_h + self.user_ff(user_h)  # feedforward

        # 4. 一次 cross-attention: server queries user
        # server作为query，user作为key/value
        connect_mask_s = connect_mask.permute(0, 2, 1)  # [B,S,U]，True表示不可连接

        # 创建attention mask: [B*num_heads, S, U]
        attn_mask_server = connect_mask_s.unsqueeze(1).expand(B, self.num_heads, S, U)  # [B, num_heads, S, U]
        attn_mask_server = attn_mask_server.reshape(B * self.num_heads, S, U)  # [B*num_heads, S, U]

        server_attn_out, _ = self.server_attn(
            query=server_h,  # [B,S,D]
            key=user_h,  # [B,U,D]
            value=user_h,  # [B,U,D]
            attn_mask=attn_mask_server  # [B*num_heads, S, U]
        )
        server_h = self.server_norm(server_h + server_attn_out)  # residual + norm
        server_h = server_h + self.server_ff(server_h)  # feedforward

        # 5. Project edge_emb to d_model for decoder
        spatial_enc = self.edge_to_d(edge_emb)  # [B,U,S,D]

        return user_h, server_h, spatial_enc


# --------------------------
# 模块2：解码器（处理多轮分配决策）
# --------------------------
class ResourceAllocatorDecoder(nn.Module):
    """
    解码器：基于编码器输出，结合动态状态（资源/激活）进行多轮分配决策
    输入：
        user_enc: [B, U, D]  编码器输出的用户嵌入
        server_enc: [B, S, D]  编码器输出的服务器初始嵌入
        spatial_enc: [B, U, S, D]  编码器输出的空间嵌入
        connect: [B, U, S]  连接矩阵
        users: [B, U, 6]  原始用户特征（取资源需求）
        servers: [B, S, 7]  原始服务器特征（取初始资源）
        policy: str  决策策略（sample/greedy）
        device: torch.device  计算设备
    输出：
        logp_accum: [B]  对数概率累积
        allocated: [B, U]  分配结果（-1未分配）
        cap: [B, S, 4]  剩余资源
        active: [B, S]  服务器激活状态
    """

    def __init__(self, d_model, num_heads, dropout, server_state_dim):
        super().__init__()
        self.d_model = d_model

        self.state_proj = nn.Linear(server_state_dim, d_model)

        # 轻量级多智能体注意力聚合器
        self.server_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 轻量级前馈网络
        self.server_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(d_model)

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

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

        while True:
            # ---------------------
            # 1. 动态 server 状态嵌入
            # ---------------------
            dynamic_state = torch.cat([cap, active.unsqueeze(-1).float()], dim=-1)
            state_emb = self.state_proj(dynamic_state)
            # server embedding 融合动态状态
            server_input = self.norm(server_enc + state_emb)

            # 多智能体注意力聚合 - 并行处理所有服务器间的交互
            # 自注意力：每个服务器都能看到所有其他服务器的状态
            attn_out, _ = self.server_attention(
                query=server_input,  # [B, S, d_model]
                key=server_input,    # [B, S, d_model]
                value=server_input   # [B, S, d_model]
            )

            # 残差连接 + 轻量前馈
            server_intermediate = self.norm(server_input + attn_out)
            server_current_emb = self.norm(server_intermediate + self.server_ff(server_intermediate))

            # =====================================================
            # 2. 计算logits
            # =====================================================
            s_emb_expand = server_current_emb.unsqueeze(1).expand(B, U, S, self.d_model)
            u_emb_expand = user_enc.unsqueeze(2).expand(B, U, S, self.d_model)

            # 融合特征计算
            fused = torch.tanh(s_emb_expand + u_emb_expand + spatial_enc)
            logits = self.score_head(fused).squeeze(-1)  # [B,U,S]
            # =====================================================
            # 可行性约束
            # =====================================================
            can_fulfill = (cap.unsqueeze(1) >= needs.unsqueeze(2)).all(dim=-1)
            eligible = connect & can_fulfill & unassigned.unsqueeze(-1)

            if not eligible.any():
                break

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

            # 简化的冲突解决：服务器优先，选择冲突时最好的服务器赢得用户
            picked_u_idx = picks.clamp(min=0)

            # 1. 统计每个用户被选择的次数和对应的服务器
            user_selected_count = torch.zeros(B, U, device=device, dtype=torch.long)
            user_selected_count.scatter_add_(1, picked_u_idx, has_any.long())

            # 2. 对于没有冲突的用户（只被一个服务器选择），直接接受
            accepted = has_any.clone()  # [B, S] 默认接受所有选择

            # 3. 并行处理冲突用户：大规模场景优化 (U=6000, S=200)
            conflict_users = (user_selected_count > 1)  # [B, U]

            if conflict_users.any():
                # 收集所有冲突信息，用于并行处理
                conflict_b, conflict_u = torch.where(conflict_users)  # [num_conflicts]

                if len(conflict_b) > 0:
                    # 为所有冲突用户收集竞争服务器信息
                    # conflict_logits: [num_conflicts, S] - 每个冲突用户对所有服务器的logits
                    conflict_logits = logits[conflict_b, conflict_u, :]  # [num_conflicts, S]

                    # competing_mask: [num_conflicts, S] - 每个冲突用户被哪些服务器选择
                    conflict_picked_u = picked_u_idx[conflict_b, :]  # [num_conflicts, S]
                    conflict_has_any = has_any[conflict_b, :]  # [num_conflicts, S]
                    competing_mask = (conflict_picked_u == conflict_u.unsqueeze(1)) & conflict_has_any

                    # 计算每个冲突用户的竞争服务器数量
                    competing_counts = competing_mask.sum(dim=1)  # [num_conflicts]

                    # 只处理有多个竞争者的冲突
                    multi_compete_mask = competing_counts > 1  # [num_conflicts]

                    if multi_compete_mask.any():
                        # 筛选出需要解决的冲突
                        active_indices = multi_compete_mask.nonzero(as_tuple=False).squeeze(-1)
                        active_logits = conflict_logits[active_indices]  # [num_active, S]
                        active_competing = competing_mask[active_indices]  # [num_active, S]
                        active_conflict_b = conflict_b[active_indices]  # [num_active]

                        # 并行为每个活跃冲突选择最好的服务器
                        # 将非竞争服务器的logits设为负无穷
                        masked_active_logits = torch.where(active_competing, active_logits,
                                                           torch.full_like(active_logits, -float('inf')))

                        # 找到每个冲突中最好的服务器索引
                        best_server_local_idx = masked_active_logits.argmax(dim=1)  # [num_active]

                        # 将局部服务器索引转换为全局accepted索引
                        # 创建一个全零的掩码，然后只设置赢家为True
                        winner_mask = torch.zeros_like(accepted)  # [B, S]

                        # 使用高级索引同时设置所有赢家
                        winner_mask[active_conflict_b, best_server_local_idx] = True

                        # 并行创建竞争服务器掩码
                        all_competing_mask_float = torch.zeros(B, S, dtype=torch.float, device=device)  # [B, S]

                        # 并行设置所有竞争服务器：使用add累加实现逻辑或
                        # 为每个活跃冲突创建完整的掩码，然后累加
                        expanded_b = active_conflict_b.unsqueeze(1).expand(-1, S)  # [num_active, S]
                        expanded_competing = active_competing.float()  # [num_active, S]

                        # scatter_add_ to accumulate competing servers (逻辑或效果)
                        all_competing_mask_float.scatter_add_(0, expanded_b, expanded_competing)

                        # 转换回bool类型：任何位置 > 0 都表示有竞争
                        all_competing_mask = (all_competing_mask_float > 0)

                        # 最终结果：保留赢家，取消其他竞争者
                        # accepted 保持不变，只取消非赢家的竞争者
                        accepted = accepted & (~all_competing_mask | winner_mask)

            if not accepted.any():
                break

            b_idx_acc, s_idx_acc = torch.where(accepted)
            u_idx_acc = picks[b_idx_acc, s_idx_acc]
            allocated[b_idx_acc, u_idx_acc] = s_idx_acc
            active[b_idx_acc, s_idx_acc] = True
            unassigned[b_idx_acc, u_idx_acc] = False
            cap[b_idx_acc, s_idx_acc] -= needs[b_idx_acc, u_idx_acc]
            logp_accum += (pick_log * accepted.float()).sum(dim=1)

        return logp_accum, allocated, cap, active


# ----------------------------
# 主模块：轻量多智能体资源分配器  多智能体资源分配器
# ----------------------------
class MACAllocator(nn.Module):
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
            num_heads,
            dropout,
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
        # 资源利用率
        masked_cap = cap.masked_fill(~active.unsqueeze(-1).expand(B, S, 4), 0)  # [B, S, 4]（未激活服务器资源置0）
        masked_servers = servers[:, :, 3:7].masked_fill(~active.unsqueeze(-1).expand(B, S, 4), 0)  # [B, S, 4]
        total_remain = torch.sum(masked_cap, dim=1)  # [B, 4]（总剩余资源）
        total_original = torch.sum(masked_servers, dim=1)  # [B, 4]（总初始资源）
        per_dim_remain_ratio = torch.nan_to_num(total_remain / total_original, nan=0.0)  # [B, 4]（各资源剩余率）
        mean_remain_ratio = torch.mean(per_dim_remain_ratio, dim=1)  # [B]（平均剩余率）
        capacity_used_ratio = 1 - mean_remain_ratio  # [B]（资源利用率）
        # 传播时延
        p_lat = calculate_propagation_latency(allocated, p_distance)  # [B]（原始时延）
        p_lat_normalized = p_lat / self.MAX_PROPAGATION_LATENCY  # [B]（归一化时延）
        reward = (
                alloc_ratio
                capacity_used_ratio
                p_lat_normalized
        )

        # print(users[:, :, 2:6].sum())
        # print((users[:, :, 2:6] * (allocated != -1).unsqueeze(-1)).sum())
        # print(cap.sum())
        # print(servers[:, :, 3:7].sum())
        # print(alloc_num)

        return -reward, logp_accum, alloc_num, alloc_ratio, active_ratio, capacity_used_ratio, p_lat_normalized
