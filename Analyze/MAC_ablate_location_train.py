import time
import yaml
import os
import gc
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from Plan.MACAllocator_Ablate_Location import MACAllocator_Ablate_Location
from Monitoring.user_gen import gen_dataset
from Analyze.Utils import check_gradients

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

train_data_size = config.get('train_data_size')
miu = config.get('telecom_miu')
sigma = config.get('telecom_sigma')
radius_low = config.get('telecom_radius_low')
radius_high = config.get('telecom_radius_high')
batch_size = config.get('batch_large_size')
user_num = config.get('telecom_train_user_num')
server_percent = config.get('telecom_train_server_percent')
dataset_save_path = config.get('telecom_train_dataset_save_path')
server_path = config.get('server_telecom_path')
d_model = config.get('d_model')
dropout = config.get('dropout')
num_heads = config.get('num_heads')
edge_dim = config.get('edge_dim')
user_feature_dim = config.get('user_feature_dim')
server_feature_dim = config.get('server_feature_dim')
spatial_raw_dim = config.get('spatial_raw_dim')
server_state_dim = config.get('server_state_dim')
lr = config.get('lr')
patience = config.get('patience')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_set = 'telecom'


def MAC_ablate_location_train():
    torch.autograd.set_detect_anomaly(False)

    data_type = {'train': [], 'valid': []}
    dataset = gen_dataset(
        user_num, train_data_size, server_path, dataset_save_path, server_percent,
        radius_low, radius_high, miu, sigma, device, data_type, data_set
    )
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)

    model = MACAllocator_Ablate_Location(
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        edge_dim=edge_dim,
        user_feature_dim=user_feature_dim,
        server_feature_dim=server_feature_dim,
        spatial_raw_dim=spatial_raw_dim,
        server_state_dim=server_state_dim,
        device=device,
        MAX_PROPAGATION_LATENCY=radius_high,
        policy='sample').to(device)
    ema_model = MACAllocator_Ablate_Location(
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        edge_dim=edge_dim,
        user_feature_dim=user_feature_dim,
        server_feature_dim=server_feature_dim,
        spatial_raw_dim=spatial_raw_dim,
        server_state_dim=server_state_dim,
        device=device,
        MAX_PROPAGATION_LATENCY=radius_high,
        policy='sample').to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False
    model.to(device)
    ema_model.to(device)

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"📊 可训练参数量: {count_parameters(model):,}")  # 267,521  990,977  210,465

    model_dir = f"./model/MACAllocator_Ablate_Location/{time.strftime('%m%d%H%M')}_server_{server_percent}_user_{user_num}_miu_{miu}_sigma_{sigma}"
    os.makedirs(model_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    epoch = 0
    best_val_loss = float('inf')
    stagnation = 0
    start_time = time.time()
    ema_baseline_interval = int(config.get('ema_baseline_interval', 2))
    global_step = 0
    baseline_cache = None

    while True:
        model.train()
        model.policy = 'sample'
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] TRAIN")
        for srv, usr, conn, lat in pbar:
            srv, usr, conn, lat = map(lambda x: x.to(device), [srv, usr, conn, lat])
            optimizer.zero_grad()

            loss, log_prob, alloc_num, user_ratio, active_ratio, prop_lat = model(srv, usr, conn, lat)
            if (
                    (global_step % ema_baseline_interval == 0)
                    or (baseline_cache is None)
                    or (baseline_cache.shape != loss.shape)
            ):
                with torch.no_grad():
                    V, *_ = ema_model(srv, usr, conn, lat)
                baseline_cache = V.detach()
            else:
                V = baseline_cache

            advantage = loss - V
            adv_mean = advantage.mean()
            adv_std = advantage.std(unbiased=False) + 1e-6  # 防止除零
            advantage_normalized = (advantage - adv_mean) / adv_std
            reinforce_loss = (advantage_normalized.detach() * log_prob).mean()
            reinforce_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # check_gradients(model)
            global_step += 1

            decay = 0.99
            alpha = 1 - decay
            with torch.no_grad():
                for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                    p_ema.mul_(decay).add_(p_model, alpha=alpha)

            loss_val = loss.mean().item()
            bl_val = V.mean().item()
            alloc_val = float(alloc_num.mean().item())
            ur = float(user_ratio.mean().item()) * 100.0
            svr = float(active_ratio.mean().item()) * 100.0
            lat = float(prop_lat.mean().item()) * 100.0
            lr_val = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                "Actor": f"{loss_val:.4f}",
                "BL": f"{bl_val:.4f}",
                "AllocNum": f"{alloc_val:.1f}",
                "User%": f"{ur:.2f}%",
                "Svr%": f"{svr:.2f}%",
                "PropLat": f"{lat:.2f}",
                "LR": f"{lr_val:.6f}"
            })
            # -------- 显式释放 --------
            del loss, log_prob, alloc_num, user_ratio
            del active_ratio, prop_lat
            del V, advantage, reinforce_loss

        model.eval()
        model.policy = 'greedy'
        val_losses, val_user_ratios, val_svr_ratios, val_prop_lats = [], [], [], []

        with torch.no_grad():
            for srv, usr, conn, lat in valid_loader:
                srv, usr, conn, lat = map(lambda x: x.to(device), [srv, usr, conn, lat])
                val_loss, *_, val_user_ratio, val_svr_ratio, val_prop_lat = model(srv, usr, conn, lat)
                val_losses.append(val_loss.mean().item())
                val_user_ratios.append(val_user_ratio.mean().item())
                val_svr_ratios.append(val_svr_ratio.mean().item())
                val_prop_lats.append(val_prop_lat.mean().item())
        val_loss_mean = np.mean(val_losses)
        val_user_ratio_mean = np.mean(val_user_ratios)
        val_svr_ratio_mean = np.mean(val_svr_ratios)
        val_prop_lat_mean = np.mean(val_prop_lats)

        print(
            f"\n[VALID] Epoch {epoch} | BestLoss: {best_val_loss:.4f} | ValLoss: {val_loss_mean:.4f} | User%: {val_user_ratio_mean:.4%}"
            f" | Server%: {val_svr_ratio_mean:.2%} | Lat/m: {val_prop_lat_mean:.2f}")

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            stagnation = 0
            save_path = os.path.join(model_dir,
                                     f"{time.strftime('%m%d%H%M')}_{epoch}_alloc_{val_user_ratio_mean:.4f}_lat_{val_prop_lat_mean:.2f}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 模型已保存: {save_path}")
        else:
            stagnation += 1
            print(f"⚠️ 验证指标停滞 {stagnation} 轮")
            if stagnation >= patience:
                print(f"⏹️ 训练早停，{patience} 轮无提升")
                break
        torch.cuda.empty_cache()
        gc.collect()
        epoch += 1

    total_time = (time.time() - start_time) / 3600
    print(f"✅ 总训练时间: {total_time:.2f} 小时")

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(torch.__version__)  # 需要 >= 2.0.0
        # print(torch.cuda.is_available())  # 应为True
        # print(torch.cuda.get_device_capability())  # 计算能力需 >= (8,0)
        MAC_ablate_location_train()
    else:
        print("cuda.unavailable!")
