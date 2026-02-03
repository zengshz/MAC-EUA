import torch
import numpy as np
import matplotlib.pyplot as plt



def calculate_propagation_latency(user_allocated: torch.LongTensor,
                                   distances: torch.Tensor) -> torch.Tensor:
    """
    基于分配结果和传播距离矩阵计算平均传播距离（米）。

    参数：
        user_allocated - [B, U]，用户分配到的服务器索引，-1 表示未分配
        distances      - [B, U, S]，传播距离矩阵，行是用户，列是服务器（单位：米）

    返回：
        avg_distance   - [B]，每个 batch 的平均传播距离（米）
    """
    B, U = user_allocated.shape

    # 1) 掩码：哪些用户是真正分配到服务器的
    mask = (user_allocated != -1)                        # [B, U]

    # 2) 防越界索引
    idx = user_allocated.clone()
    idx[~mask] = 0                                       # 将未分配的索引置为0

    # 3) 收集对应的传播距离 [B, U]
    indices = idx.unsqueeze(-1)                          # [B, U, 1]
    prop_dist = torch.gather(distances, 2, indices).squeeze(-1)

    # 4) 将 nan 或 inf 替换成 0，确保后续运算安全
    prop_dist = prop_dist.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    # 5) 把未分配的位置强制为 0
    prop_dist = prop_dist.masked_fill(~mask, 0.0)

    # 6) 计算总距离与平均距离
    total_dist   = prop_dist.sum(dim=1)                  # [B]
    valid_counts = mask.sum(dim=1).float()               # [B]

    # 平均传播距离，防止除零
    avg_distance = torch.zeros(B, device=distances.device)
    nonzero      = valid_counts > 0
    avg_distance[nonzero] = total_dist[nonzero] / valid_counts[nonzero]
    return avg_distance


def check_gradients(model, plot_grad_dist=True):
    """
    检查模型参数的梯度状态并可视化梯度分布

    参数:
    model: nn.Module
        要检查的PyTorch模型
    plot_grad_dist: bool (默认为True)
        是否绘制梯度分布直方图
    """
    # 检查梯度是否存在
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"⚠️ [梯度丢失] {name}: 未接收梯度")
        else:
            grad_norm = param.grad.norm().item()
            print(f"✅ [梯度正常] {name}: 梯度范数={grad_norm:.4e}")

    # 可视化梯度分布
    if plot_grad_dist:
        all_grads = []
        for param in model.parameters():
            if param.grad is not None:
                # 将梯度数据转为numpy数组并展平
                all_grads.append(param.grad.detach().view(-1).cpu().numpy())

        if len(all_grads) > 0:
            all_grads = np.concatenate(all_grads)
            plt.figure(figsize=(10, 6))
            plt.hist(all_grads, bins=100, alpha=0.7)
            plt.yscale('log')
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency (log scale)")
            plt.title("Gradient Distribution")
            plt.grid(True, which="both", ls="--")
            # **保存为图片**
            plt.savefig("gradient_distribution.png")  # 避免 plt.show() 的问题
            print("📊 梯度分布图已保存为 gradient_distribution.png")
        else:
            print("⚠️ 所有参数均无梯度，无法绘制分布图")
