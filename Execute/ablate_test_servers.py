from torch.utils.data import DataLoader
from Monitoring.user_gen import gen_dataset
import yaml
import os
import torch
import pandas as pd
import time
from tqdm import tqdm
from Plan.MACAllocator import MACAllocator
from Plan.MACAllocator_Ablate_NOT import MACAllocator_Ablate_NOT
from Plan.MACAllocator_Ablate_Location import MACAllocator_Ablate_Location
from Plan.MACAllocator_Ablate_Encoder import MACAllocator_Ablate_Encoder

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


def ablate_test_servers():
    servers_percent_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    result_dir = "../Knowledge/result"
    os.makedirs(result_dir, exist_ok=True)

    all_results = []  # 用于存放所有实验数据

    for index, server_percent in enumerate(servers_percent_list):
        print(f"\n===== 处理服务器百分比数%: {server_percent}（{index + 1}/{len(servers_percent_list)}） =====")

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

        not_model = MACAllocator_Ablate_NOT(d_model=d_model,
                                            dropout=dropout,
                                            user_feature_dim=user_feature_dim,
                                            server_feature_dim=server_feature_dim,
                                            server_state_dim=server_state_dim,
                                            device=device,
                                            MAX_PROPAGATION_LATENCY=radius_high,
                                            policy='greedy')
        not_state_dict = torch.load(
            '../Analyze/model/MACAllocator_Ablate_NOT/01221008_server_25_user_1500_miu_80_sigma_20/01231130_200_alloc_0.7127_cap_0.6462_lat_300.63_best.pth',
            map_location='cpu')
        not_model.load_state_dict(not_state_dict)
        not_model.to(device)  # 移到目标设备
        not_model.eval()
        not_model.policy = 'greedy'

        location_model = MACAllocator_Ablate_Location(d_model=d_model,
                                                      dropout=dropout,
                                                      edge_dim=edge_dim,
                                                      user_feature_dim=user_feature_dim,
                                                      server_feature_dim=server_feature_dim,
                                                      spatial_raw_dim=spatial_raw_dim,
                                                      server_state_dim=server_state_dim,
                                                      device=device,
                                                      MAX_PROPAGATION_LATENCY=radius_high,
                                                      policy='greedy')
        location_state_dict = torch.load(
            '../Analyze/model/MACAllocator_Ablate_Location/01221015_server_25_user_1500_miu_80_sigma_20/01230756_162_alloc_0.7100_cap_0.6388_lat_216.51_best.pth',
            map_location='cpu')
        location_model.load_state_dict(location_state_dict)
        location_model.to(device)  # 移到目标设备
        location_model.eval()
        location_model.policy = 'greedy'

        encoder_model = MACAllocator_Ablate_Encoder(d_model=d_model,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    edge_dim=edge_dim,
                                                    user_feature_dim=user_feature_dim,
                                                    server_feature_dim=server_feature_dim,
                                                    spatial_raw_dim=spatial_raw_dim,
                                                    server_state_dim=server_state_dim,
                                                    device=device,
                                                    MAX_PROPAGATION_LATENCY=radius_high,
                                                    policy='sample')
        encoder_state_dict = torch.load(
            '../Analyze/model/MACAllocator_Ablate_Encoder/01221457_server_25_user_1500_miu_80_sigma_20/01231233_175_alloc_0.7406_cap_0.7191_lat_225.29_best.pth',
            map_location='cpu')
        encoder_model.load_state_dict(encoder_state_dict)
        encoder_model.to(device)  # 移到目标设备
        encoder_model.eval()
        encoder_model.policy = 'greedy'

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

        # 预热所有模型
        for _ in range(warmup_iterations):
            with torch.no_grad():
                not_model(warmup_servers, warmup_users, warmup_connect, warmup_latency)
                location_model(warmup_servers, warmup_users, warmup_connect, warmup_latency)
                encoder_model(warmup_servers, warmup_users, warmup_connect, warmup_latency)
                mac_model(warmup_servers, warmup_users, warmup_connect, warmup_latency)

        torch.cuda.empty_cache()
        print("预热完成，开始正式测试...")

        # 重新创建DataLoader（预热消耗了一个batch）
        test_data = DataLoader(dataset=dataset['test'], batch_size=1, shuffle=False)

        # 3. 遍历测试数据，同步运行所有算法
        batch_results = []

        for group_idx, (servers, users, connect, latency) in tqdm(
                enumerate(test_data), total=len(test_data), desc="运行算法"):
            # 数据分发到对应设备（按算法负载分配，平衡内存）
            # device0: 轻量算法（随机、贪婪容量、贪婪距离、DSAM）
            servers0 = servers.to(device)
            users0 = users.to(device)
            connect0 = connect.to(device)
            latency0 = latency.to(device)

            # 存储当前batch的所有结果（按算法顺序）
            current_results = []

            def not_inference():
                with torch.no_grad():
                    return not_model(servers0, users0, connect0, latency0)

            elapsed_ms_not, (_, _, _, not_allocated_user_ratio, _, not_capacity_used_ratio,
                             not_propagation_delay_aver) = measure_gpu_time(
                not_inference, num_runs=3)
            current_results.extend([
                not_allocated_user_ratio.mean().cpu(),
                not_capacity_used_ratio.mean().cpu(),
                not_propagation_delay_aver.mean().cpu(),
                elapsed_ms_not
            ])

            def location_inference():
                with torch.no_grad():
                    return location_model(servers0, users0, connect0, latency0)

            elapsed_ms_location, (_, _, _, location_allocated_user_ratio, _, location_capacity_used_ratio,
                                  location_propagation_delay_aver) = measure_gpu_time(
                location_inference, num_runs=3)
            current_results.extend([
                location_allocated_user_ratio.mean().cpu(),
                location_capacity_used_ratio.mean().cpu(),
                location_propagation_delay_aver.mean().cpu(),
                elapsed_ms_location
            ])

            def encoder_inference():
                with torch.no_grad():
                    return encoder_model(servers0, users0, connect0, latency0)

            elapsed_ms_encoder, (_, _, _, encoder_allocated_user_ratio, _, encoder_capacity_used_ratio,
                                 encoder_propagation_delay_aver) = measure_gpu_time(
                encoder_inference, num_runs=3)
            current_results.extend([
                encoder_allocated_user_ratio.mean().cpu(),
                encoder_capacity_used_ratio.mean().cpu(),
                encoder_propagation_delay_aver.mean().cpu(),
                elapsed_ms_encoder
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
            'not_user_ratio', 'not_capacity_used_ratio', 'not_propagation_delay_aver', 'not_time',
            'location_user_ratio', 'location_capacity_used_ratio', 'location_propagation_delay_aver', 'location_time',
            'encoder_user_ratio', 'encoder_capacity_used_ratio', 'encoder_propagation_delay_aver', 'encoder_time',
            'MAC_user_ratio', 'MAC_capacity_used_ratio', 'MAC_propagation_delay_aver', 'MAC_time',
        ]

        # 创建当前用户数的结果行
        result_df = pd.DataFrame([mean_results.numpy()], columns=columns)
        result_df.insert(0, 'users', user_num)  # 添加用户数列
        result_df.insert(1, 'servers', server_percent)  # 添加服务器数量列
        all_results.append(result_df)

        # 清理当前用户数的模型和张量，释放显存
        del mac_model, all_batch_tensor
        torch.cuda.empty_cache()

    # 6. 合并所有结果并保存
    final_df = pd.concat(all_results, ignore_index=True)
    save_time = time.strftime('%m%d%H%M', time.localtime())
    save_path = os.path.join(
        result_dir,
        f"ablate_test_{user_num}users_{save_time}.csv"
    )
    final_df.to_csv(save_path, index=False)
    print(f"\n所有算法对比结果已保存至：{save_path}")


# -------------------------- 执行入口 --------------------------
if __name__ == '__main__':
    if torch.cuda.is_available():
        ablate_test_servers()
    else:
        print("警告：未检测到CUDA设备，将使用CPU运行（可能很慢）")
        ablate_test_servers()
