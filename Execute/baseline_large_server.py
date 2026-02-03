from torch.utils.data import DataLoader
from Plan.baseline.random_allocation import random_allocation
from Plan.baseline.greedy_capacity_allocation import greedy_capacity_allocation
from Plan.baseline.mcf_capacity_allocation import mcf_capacity_allocation
from Plan.baseline.mcf_distance_allocation import mcf_distance_allocation
from Plan.baseline.dro_allocation import dro_allocation
from Plan.baseline.DSAM_allocation import AttentionNet
from Plan.MACAllocator import MACAllocator
from Monitoring.user_gen import gen_dataset
import yaml
import os
import torch
import pandas as pd
import time
from tqdm import tqdm

# 读取 config.yaml 文件
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

test_data_size = config.get('test_data_size')
miu = config.get('telecom_miu')
sigma = config.get('telecom_sigma')
radius_low = config.get('telecom_radius_low')
radius_high = config.get('telecom_radius_high')
server_path = config.get('server_telecom_path')
dataset_save_path = config.get('telecom_test_save_path')
d_model = config.get('d_model')
dropout = config.get('dropout')
num_heads = config.get('num_heads')
edge_dim = config.get('edge_dim')
user_feature_dim = config.get('user_feature_dim')
server_feature_dim = config.get('server_feature_dim')
spatial_raw_dim = config.get('spatial_raw_dim')
server_state_dim = config.get('server_state_dim')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_set = 'telecom'
user_num = 3000

def measure_gpu_time(func, *args, num_runs=3):
    """
    使用CUDA事件精确测量GPU操作时间，多次运行取中位数以提高稳定性

    Args:
        func: 要测量的函数
        *args: 函数参数
        num_runs: 运行次数，默认3次

    Returns:
        float: 中位数执行时间（毫秒）
    """
    times = []

    for _ in range(num_runs):
        # 创建CUDA事件
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 记录开始时间
        start_event.record()

        # 执行函数
        result = func(*args)

        # 记录结束时间
        end_event.record()

        # 等待测量完成
        end_event.synchronize()

        # 获取执行时间
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)

    # 返回中位数以减少异常值影响
    times.sort()
    return times[len(times) // 2], result


def baseline_large_server():
    servers_percent_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # 确保结果目录存在
    result_dir = "../Knowledge/result"
    os.makedirs(result_dir, exist_ok=True)

    all_results = []  # 用于存放所有实验数据

    # 2. 加载所有模型（只加载一次，复用）
    # 2.1 加载DSAM模型（基线）
    dsam_model = AttentionNet(
        6, 7, hidden_dim=256, device=device,
        exploration_c=10,
        capacity_reward_rate=0.5,
        user_embedding_type='transformer',
        server_embedding_type='linear',
        transformer_n_heads=8,
        transformer_n_layers=3,
        transformer_feed_forward_hidden=512,
        user_scale_alpha=0.0625
    )
    # 加载权重时映射到CPU，避免单卡加载内存溢出
    dsam_state_dict = torch.load(
        '../Analyze/model/dsam/11181603_server_25_user_1500_mean_80_std_20/11220746_71.70_100.00_67.56.mdl',
        map_location='cpu')
    dsam_model.load_state_dict(dsam_state_dict)
    dsam_model.to(device)  # 移到目标设备
    dsam_model.eval()
    dsam_model.policy = 'greedy'

    mac_model = MACAllocator(d_model=d_model,
                             num_heads=num_heads,
                             dropout=dropout,
                             edge_dim=edge_dim,
                             user_feature_dim=user_feature_dim,
                             server_feature_dim=server_feature_dim,
                             spatial_raw_dim=spatial_raw_dim,
                             server_state_dim=server_state_dim,
                             device=device,
                             MAX_PROPAGATION_LATENCY=radius_high,
                             policy='greedy')
    mac_state_dict = torch.load(
        '../Analyze/model/MACAllocator/01100944_server_25_user_1500_miu_80_sigma_20/01111612_175_alloc_0.7219_cap_0.6723_lat_242.92_best.pth',
        map_location='cpu')
    mac_model.load_state_dict(mac_state_dict)
    mac_model.to(device)  # 移到目标设备
    mac_model.eval()
    mac_model.policy = 'greedy'

    for index, server_percent in enumerate(servers_percent_list):
        print(f"\n===== 处理服务器百分比数%: {server_percent}（{index + 1}/{len(servers_percent_list)}） =====")

        data_type = {'test': []}
        dataset = gen_dataset(
            user_num, test_data_size, server_path, dataset_save_path, server_percent,
            radius_low, radius_high, miu, sigma, torch.device("cpu"), data_type, data_set
        )
        test_data = DataLoader(dataset=dataset['test'], batch_size=1, shuffle=False)

        # 预热阶段：多次运行以稳定GPU状态和JIT编译
        print("正在预热GPU和JIT编译...")
        warmup_iterations = 5

        # 获取一个batch用于预热
        warmup_data = next(iter(test_data))
        warmup_servers, warmup_users, warmup_connect, warmup_latency = warmup_data
        warmup_servers = warmup_servers.to(device)
        warmup_users = warmup_users.to(device)
        warmup_connect = warmup_connect.to(device)
        warmup_latency = warmup_latency.to(device)

        # 预热所有算法
        for _ in range(warmup_iterations):
            random_allocation(warmup_servers, warmup_users, warmup_connect, warmup_latency)
            greedy_capacity_allocation(warmup_servers, warmup_users, warmup_connect, warmup_latency)
            dro_allocation(warmup_servers, warmup_users, warmup_connect, warmup_latency)
            mcf_capacity_allocation(warmup_servers, warmup_users, warmup_connect, warmup_latency)
            mcf_distance_allocation(warmup_servers, warmup_users, warmup_connect, warmup_latency)
            with torch.no_grad():
                mac_model(warmup_servers, warmup_users, warmup_connect, warmup_latency)
                dsam_model(warmup_users, warmup_servers, warmup_connect, warmup_latency)

        torch.cuda.empty_cache()
        print("预热完成，开始正式测试...")

        # 重新创建DataLoader（预热消耗了一个batch）
        test_data = DataLoader(dataset=dataset['test'], batch_size=1, shuffle=False)

        # 3. 遍历测试数据，同步运行所有算法
        batch_results = []

        for group_idx, (servers, users, connect, latency) in tqdm(
                enumerate(test_data), total=len(test_data), desc="运行基线算法"):
            # 数据分发到对应设备（按算法负载分配，平衡内存）
            # device0: 轻量算法（随机、贪婪容量、贪婪距离、DSAM）
            servers0 = servers.to(device)
            users0 = users.to(device)
            connect0 = connect.to(device)
            latency0 = latency.to(device)

            # 存储当前batch的所有结果（按算法顺序）
            current_results = []

            # 3.1 随机分配算法
            elapsed_ms_random, (_, random_allocated_user_ratio, _, random_capacity_used_ratio, \
                random_propagation_delay_aver) = measure_gpu_time(
                random_allocation, servers0, users0, connect0, latency0, num_runs=3)
            current_results.extend([
                random_allocated_user_ratio.mean().cpu(),  # 用户分配率
                random_capacity_used_ratio.mean().cpu(),  # 容量利用率
                random_propagation_delay_aver.mean().cpu(),  # 平均延迟
                elapsed_ms_random
            ])

            # 3.2 贪婪容量分配算法
            elapsed_ms_greedy_capacity, (_, greedy_capacity_allocated_user_ratio, _, greedy_capacity_capacity_used_ratio, \
                greedy_capacity_propagation_delay_aver) = measure_gpu_time(
                greedy_capacity_allocation, servers0, users0, connect0, latency0, num_runs=3)
            current_results.extend([
                greedy_capacity_allocated_user_ratio.mean().cpu(),
                greedy_capacity_capacity_used_ratio.mean().cpu(),
                greedy_capacity_propagation_delay_aver.mean().cpu(),
                elapsed_ms_greedy_capacity
            ])

            # 3.5 DRO分配算法
            elapsed_ms_dro, (_, dro_allocated_user_ratio, _, dro_capacity_used_ratio, \
                dro_propagation_delay_aver) = measure_gpu_time(
                dro_allocation, servers0, users0, connect0, latency0, num_runs=3)
            current_results.extend([
                dro_allocated_user_ratio.mean().cpu(),
                dro_capacity_used_ratio.mean().cpu(),
                dro_propagation_delay_aver.mean().cpu(),
                elapsed_ms_dro
            ])

            # 3.6 MCF容量分配算法
            elapsed_ms_mcf_capacity, (_, mcf_capacity_allocated_user_ratio, _, mcf_capacity_capacity_used_ratio, \
                mcf_capacity_propagation_delay_aver) = measure_gpu_time(
                mcf_capacity_allocation, servers0, users0, connect0, latency0, num_runs=3)
            current_results.extend([
                mcf_capacity_allocated_user_ratio.mean().cpu(),
                mcf_capacity_capacity_used_ratio.mean().cpu(),
                mcf_capacity_propagation_delay_aver.mean().cpu(),
                elapsed_ms_mcf_capacity
            ])

            # 3.7 MCF距离分配算法
            elapsed_ms_mcf_distance, (_, mcf_distance_allocated_user_ratio, _, mcf_distance_capacity_used_ratio, \
                mcf_distance_propagation_delay_aver) = measure_gpu_time(
                mcf_distance_allocation, servers0, users0, connect0, latency0, num_runs=3)
            current_results.extend([
                mcf_distance_allocated_user_ratio.mean().cpu(),
                mcf_distance_capacity_used_ratio.mean().cpu(),
                mcf_distance_propagation_delay_aver.mean().cpu(),
                elapsed_ms_mcf_distance
            ])

            # -------------------------- 运行MAC算法 --------------------------
            def mac_inference():
                with torch.no_grad():  # 关闭梯度计算
                    return mac_model(servers0, users0, connect0, latency0)

            elapsed_ms_mac, (_, _, _, MAC_allocated_user_ratio, _, MAC_capacity_used_ratio,
                              MAC_propagation_delay_aver) = measure_gpu_time(
                mac_inference, num_runs=3)
            current_results.extend([
                MAC_allocated_user_ratio.mean().cpu(),
                MAC_capacity_used_ratio.mean().cpu(),
                MAC_propagation_delay_aver.mean().cpu(),
                elapsed_ms_mac
            ])

            # 3.4 DSAM算法
            def dsam_inference():
                with torch.no_grad():  # 关闭梯度计算，节省内存
                    return dsam_model(users0, servers0, connect0, latency0)

            elapsed_ms_dsam, (_, _, _, _, dsam_allocated_user_ratio, _, dsam_capacity_used_ratio, _, dsam_propagation_delay_aver) = measure_gpu_time(
                dsam_inference, num_runs=3)
            current_results.extend([
                dsam_allocated_user_ratio.mean().cpu(),
                dsam_capacity_used_ratio.mean().cpu(),
                dsam_propagation_delay_aver.mean().cpu(),
                elapsed_ms_dsam
            ])

            # 将当前batch结果转换为张量并存储
            batch_results.append(torch.stack([
                # 若x是张量，用clone+detach；否则直接创建张量
                x.clone().detach().to(dtype=torch.float32) if isinstance(x, torch.Tensor)
                else torch.tensor(x, dtype=torch.float32)
                for x in current_results
            ]))

            # 4. 计算当前用户数的所有batch均值（四舍五入保留4位小数）
        all_batch_tensor = torch.stack(batch_results)
        mean_results = torch.round(all_batch_tensor.mean(dim=0) * 1e4) / 1e4

        # 5. 构建结果DataFrame（列名与算法顺序严格对应）
        columns = [
            # 随机分配算法
            'random_user_ratio', 'random_capacity_used_ratio', 'random_propagation_delay_aver', 'random_time',
            # 贪婪容量分配算法
            'greedy_capacity_user_ratio', 'greedy_capacity_capacity_used_ratio',
            'greedy_capacity_propagation_delay_aver', 'greedy_capacity_time',
            # DRO分配算法
            'dro_user_ratio', 'dro_capacity_used_ratio', 'dro_propagation_delay_aver', 'dro_time',
            # MCF容量分配算法
            'mcf_capacity_user_ratio', 'mcf_capacity_capacity_used_ratio', 'mcf_capacity_propagation_delay_aver',
            'mcf_capacity_time',
            # MCF距离分配算法
            'mcf_distance_user_ratio', 'mcf_distance_capacity_used_ratio', 'mcf_distance_propagation_delay_aver',
            'mcf_distance_time',
            'MAC_user_ratio', 'MAC_capacity_used_ratio', 'MAC_propagation_delay_aver', 'MAC_time',
            # DSAM算法
            'dsam_user_ratio', 'dsam_capacity_used_ratio', 'dsam_propagation_delay_aver', 'dsam_time',

        ]

        # 创建当前用户数的结果行
        result_df = pd.DataFrame([mean_results.numpy()], columns=columns)
        result_df.insert(0, 'users', user_num)  # 添加用户数列
        result_df.insert(1, 'servers', server_percent)  # 添加服务器数量列
        all_results.append(result_df)

        # 清理当前用户数的模型和张量，释放显存
        del all_batch_tensor
        torch.cuda.empty_cache()

        # 6. 合并所有结果并保存
    final_df = pd.concat(all_results, ignore_index=True)
    save_time = time.strftime('%m%d%H%M', time.localtime())
    save_path = os.path.join(
        result_dir,
        f"baseline_large_{user_num}users_{save_time}.csv"
    )
    final_df.to_csv(save_path, index=False)
    print(f"\n所有算法对比结果已保存至：{save_path}")


# -------------------------- 执行入口 --------------------------
if __name__ == '__main__':
    if torch.cuda.is_available():
        baseline_large_server()
    else:
        print("警告：未检测到CUDA设备，将使用CPU运行（可能很慢）")
        baseline_large_server()
