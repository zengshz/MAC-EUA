import math
import torch
import torch.nn as nn
from Analyze.Utils import calculate_propagation_latency
from Plan.baseline.graph_encoder import GraphAttentionEncoder



class UserEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=8, n_layers=6,
                 normalization='batch', feed_forward_hidden=512, embedding_type='transformer'):
        super(UserEncoder, self).__init__()
        self.embedding_type = embedding_type
        if embedding_type == 'transformer':
            self.embedding = GraphAttentionEncoder(n_heads, hidden_dim, n_layers,
                                                   input_dim, normalization, feed_forward_hidden)
        elif embedding_type == 'linear':
            self.embedding = nn.Linear(input_dim, hidden_dim)
        elif embedding_type == 'lstm':
            self.embedding = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        if self.embedding_type == 'lstm':
            embedded, _ = self.embedding(inputs)
            return embedded
        else:
            return self.embedding(inputs)


class ServerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=8, n_layers=6,
                 normalization='batch', feed_forward_hidden=512, embedding_type='linear'):
        super(ServerEncoder, self).__init__()
        self.embedding_type = embedding_type
        if embedding_type == 'transformer':
            self.embedding = GraphAttentionEncoder(n_heads, hidden_dim, n_layers,
                                                   input_dim, normalization, feed_forward_hidden)
        elif embedding_type == 'linear':
            self.embedding = nn.Linear(input_dim, hidden_dim)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.embedding(inputs)


class Glimpse(nn.Module):
    # input :
    # query:    batch_size * 1 * query_input_dim
    # ref:      batch_size * seq_len * ref_hidden_dim
    def __init__(self, hidden_dim):
        super(Glimpse, self).__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self._norm_fact = 1 / math.sqrt(hidden_dim)

    def forward(self, query, ref):
        Q = self.q(query)  # Q: batch_size * 1 * hidden_dim
        K = self.k(ref)  # K: batch_size * seq_len * hidden_dim
        V = self.v(ref)  # V: batch_size * seq_len * hidden_dim

        attn_score = torch.bmm(Q, K.permute(0, 2, 1))
        attn_score = attn_score * self._norm_fact
        attn = torch.softmax(attn_score, dim=-1)    # Q * K.T() # batch_size * 1 * seq_len

        output = torch.bmm(attn, V)  # Q * K.T() * V # batch_size * 1 * hidden_dim
        # 混合了所有服务器的相似度的一个表示服务器的变量
        return output


class Attention(nn.Module):
    def __init__(self, hidden_dim, exploration_c=10, user_scale_alpha=0.05):
        super(Attention, self).__init__()
        self.hidden_size = hidden_dim
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vt = nn.Linear(hidden_dim, 1, bias=False)
        self.exploration_c = exploration_c
        self.user_scale_alpha = user_scale_alpha

    def forward(self, decoder_state, encoder_outputs, mask):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1, hidden_size)
        decoder_transform = self.W2(decoder_state * self.user_scale_alpha)

        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        # softmax with only valid inputs, excluding zero padded parts
        # log_softmax for a better numerical stability
        score = u_i.masked_fill(~mask, value=torch.log(torch.tensor(1e-45))) * self.exploration_c
        prob = torch.softmax(score, dim=-1)
        return prob


class AttentionNet(nn.Module):
    def __init__(self, user_input_dim, server_input_dim, hidden_dim, device, capacity_reward_rate, exploration_c=10,
                 policy='sample', user_embedding_type='transformer', server_embedding_type='linear',
                 transformer_n_heads=8, transformer_n_layers=3, transformer_feed_forward_hidden=512,
                 user_scale_alpha=0.05, beam_num=1, ):
        super(AttentionNet, self).__init__()
        # decoder hidden size
        self.hidden_dim = hidden_dim
        self.device = device

        self.user_encoder = UserEncoder(user_input_dim, hidden_dim, n_heads=transformer_n_heads,
                                        n_layers=transformer_n_layers,
                                        feed_forward_hidden=transformer_feed_forward_hidden,
                                        embedding_type=user_embedding_type).to(device)
        self.server_encoder = ServerEncoder(server_input_dim + 1, hidden_dim, n_heads=transformer_n_heads,
                                            n_layers=transformer_n_layers,
                                            feed_forward_hidden=transformer_feed_forward_hidden,
                                            embedding_type=server_embedding_type).to(device)

        # glimpse输入（用户，上次选择的服务器），维度为2*dim， 跟所有的服务器作相似度并输出融合后的服务器
        self.glimpse = Glimpse(hidden_dim).to(device)
        self.pointer = Attention(hidden_dim, exploration_c, user_scale_alpha).to(device)
        self.capacity_reward_rate = capacity_reward_rate
        self.policy = policy
        self.beam_num = beam_num

    def choose_server_id(self, mask, user, static_server_seq, tmp_server_capacity, server_active):
        """
        每一步根据用户和所有服务器，输出要选择的服务器
        """
        server_seq = torch.cat((static_server_seq, tmp_server_capacity, server_active), dim=-1)
        server_encoder_outputs = self.server_encoder(server_seq)
        server_glimpse = self.glimpse(user, server_encoder_outputs)
        server_glimpse = server_glimpse

        # get a pointer distribution over the encoder outputs using attention
        # (batch_size, server_len)
        probs = self.pointer(server_glimpse, server_encoder_outputs, mask)
        # (batch_size, server_len)

        if self.policy == 'sample':
            # (batch_size, 1)
            idx = torch.multinomial(probs, num_samples=self.beam_num)
            prob = torch.gather(probs, dim=1, index=idx)
        elif self.policy == 'greedy':
            prob, idx = torch.topk(probs, k=self.beam_num, dim=-1)
        else:
            raise NotImplementedError

        if self.beam_num == 1:
            prob = prob.squeeze(1)
            idx = idx.squeeze(1)

        return prob, idx

    @staticmethod
    def update_server_capacity(server_id, tmp_server_capacity, user_workload):
        batch_size = server_id.size(0)
        # 取出一个batch里所有第j个用户选择的服务器
        index_tensor = server_id.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, 4)  # 4个资源维度
        j_th_server_capacity = torch.gather(tmp_server_capacity, dim=1, index=index_tensor).squeeze(1)
        # (batch_size)的True，False矩阵
        can_be_allocated = can_allocate(user_workload, j_th_server_capacity)
        # 如果不能分配容量就不减
        mask = can_be_allocated.unsqueeze(-1).expand(batch_size, 4)
        # 切片时指定batch和server_id对应关系
        batch_range = torch.arange(batch_size)
        # 服务器减去相应容量
        tmp_server_capacity[batch_range, server_id] -= user_workload * mask
        # 记录服务器分配情况，即server_id和mask的内积
        server_id = torch.masked_fill(server_id, mask=~can_be_allocated, value=-1)
        return tmp_server_capacity, server_id

    @staticmethod
    def calc_rewards(user_allocate_list, user_len, server_allocate_mat, server_len,
                     original_servers_capacity, batch_size, tmp_server_capacity):
        # 目前user_allocate_list是(batch_size, user_len)
        # 计算每个分配的用户数，即不是-1的个数，(batch_size)
        user_allocate_num = torch.sum(user_allocate_list != -1, dim=1)
        user_allocated_props = user_allocate_num.float() / user_len

        server_used_num = torch.sum(server_allocate_mat[:, :-1], dim=1)
        server_used_props = server_used_num.float() / server_len

        # 已使用的服务器的资源利用率
        server_allocated_flag = server_allocate_mat[:, :-1].unsqueeze(-1).expand(batch_size, server_len, 4)
        # (batch_size, server_len, 4)
        used_original_server = original_servers_capacity.masked_fill(~server_allocated_flag.bool(), value=0)
        # (batch_size, server_len, 4)
        servers_remain_capacity = tmp_server_capacity.masked_fill(~server_allocated_flag.bool(), value=0)
        # 对于每个维度的资源求和，得到的结果应该是(batch_size, 4)，被压缩的维度是1
        sum_all_capacity = torch.sum(used_original_server, dim=1)
        sum_remain_capacity = torch.sum(servers_remain_capacity, dim=1)
        # 对于每个维度的资源求资源利用率
        every_capacity_remain_props = torch.div(sum_remain_capacity, sum_all_capacity)
        mean_capacity_remain_props = torch.mean(every_capacity_remain_props, dim=1)
        capacity_used_props = 1 - mean_capacity_remain_props
        return user_allocate_num.float(), user_allocated_props, server_used_props, capacity_used_props

    def forward(self, user_input_seq, server_input_seq, masks, latency):
        batch_size = user_input_seq.size(0)
        user_len = user_input_seq.size(1)
        server_len = server_input_seq.size(1)

        # 真实分配情况
        user_allocate_list = -torch.ones(batch_size, user_len, dtype=torch.long, device=self.device)
        # 服务器分配矩阵，加一是为了给index为-1的来赋值
        server_allocate_mat = torch.zeros(batch_size, server_len + 1, dtype=torch.long, device=self.device)

        # 服务器信息由三部分组成
        static_server_seq = server_input_seq[:, :, :3]
        tmp_server_capacity = server_input_seq[:, :, 3:].clone()

        user_encoder_outputs = self.user_encoder(user_input_seq)

        action_probs = []
        action_idx = []

        for i in range(user_len):
            mask = masks[:, i]
            user_code = user_encoder_outputs[:, i, :].unsqueeze(1)
            prob, idx = self.choose_server_id(mask, user_code, static_server_seq, tmp_server_capacity,
                                              server_allocate_mat[:, :-1].unsqueeze(-1))

            action_probs.append(prob)
            action_idx.append(idx)

            tmp_server_capacity, idx = self.update_server_capacity(idx, tmp_server_capacity, user_input_seq[:, i, 2:])

            # 真实分配情况
            user_allocate_list[:, i] = idx
            # 给分配了的服务器在服务器分配矩阵中赋值为True
            batch_range = torch.arange(batch_size)
            server_allocate_mat[batch_range, idx] = 1

        action_probs = torch.stack(action_probs)
        action_idx = torch.stack(action_idx, dim=-1)

        user_allocate_num, user_allocated_props, server_used_props, capacity_used_props = \
            self.calc_rewards(user_allocate_list, user_len, server_allocate_mat, server_len,
                              server_input_seq[:, :, 3:].clone(), batch_size, tmp_server_capacity)

        prop_lat = calculate_propagation_latency(user_allocate_list, latency)

        return -self.get_reward(user_allocated_props, capacity_used_props), action_probs, \
            action_idx, user_allocate_num, user_allocated_props, server_used_props, capacity_used_props, user_allocate_list, prop_lat

    def get_reward(self, user_allocated_props, capacity_used_props):
        return (1 - self.capacity_reward_rate) * user_allocated_props + self.capacity_reward_rate * capacity_used_props


def can_allocate(workload: torch.Tensor, capacity: torch.Tensor):
    """
    计算能不能分配并返回分配情况
    :param workload: (batch, 4)
    :param capacity: (batch, 4)
    :return:
    """
    # (batch, 4)
    bools = capacity >= workload
    # (batch)，bool值

    return bools.all(dim=1)
