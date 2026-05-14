import torch
import numpy as np
import matplotlib.pyplot as plt



def calculate_propagation_latency(user_allocated: torch.LongTensor,
                                   distances: torch.Tensor) -> torch.Tensor:
    B, U = user_allocated.shape
    mask = (user_allocated != -1)
    idx = user_allocated.clone()
    idx[~mask] = 0
    indices = idx.unsqueeze(-1)
    prop_dist = torch.gather(distances, 2, indices).squeeze(-1)
    prop_dist = prop_dist.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    prop_dist = prop_dist.masked_fill(~mask, 0.0)
    total_dist = prop_dist.sum(dim=1)
    valid_counts = mask.sum(dim=1).float()
    avg_distance = torch.zeros(B, device=distances.device)
    nonzero = valid_counts > 0
    avg_distance[nonzero] = total_dist[nonzero] / valid_counts[nonzero]
    return avg_distance


def check_gradients(model, plot_grad_dist=True):
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
            plt.savefig("gradient_distribution.png")
            print("📊 梯度分布图已保存为 gradient_distribution.png")
        else:
            print("⚠️ 所有参数均无梯度，无法绘制分布图")
