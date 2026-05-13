# 边缘服务器状态
# ram cpu storage bandwidth longitude latitude radius


import numpy as np
import pandas as pd
import os
import math


def miller_to_xy(lons, lats):
    """
    向量化米勒投影转换（支持批量输入）
    :param lons: 经度数组，形状 [N]
    :param lats: 纬度数组，形状 [N]
    :return: x, y 平面坐标数组，形状均为 [N]
    """
    L = 6381372 * math.pi * 2  # 地球周长
    W = L
    H = L / 2
    mill = 2.3

    # 将经纬度数组转换为弧度
    x_rad = np.radians(lons)
    y_rad = np.radians(lats)

    # 米勒投影核心公式（向量化计算）
    y_trans = 1.25 * np.log(np.tan(0.25 * math.pi + 0.4 * y_rad))

    # 转换为平面坐标
    x = (W / 2) + (W / (2 * math.pi)) * x_rad
    y = (H / 2) - (H / (2 * mill)) * y_trans

    return x, y


def coordinate_transformation_pipeline(coords):
    # 初始平移
    coords -= np.min(coords, axis=0)
    # 单位：米转换为100米
    coords /= 100
    return coords



def gen_eua_servers_dataset(server_path, percent, radius_low, radius_high, miu, sigma, save_path):
    save_path = os.path.join(save_path, f'{percent}_miu_{miu}_sigma_{sigma}_low_{radius_low}_high_{radius_high}',
                             f'servers_pct_{percent}.csv')
    # 检查服务器数据是否存在
    if os.path.exists(save_path):
        edge_servers = pd.read_csv(save_path)
    else:
        print("生成新服务器数据集...")
        #CBD area, Melbourne, AU 2.04 km2   125个
        # edge_servers_list = pd.read_csv(server_path)

        edge_servers_list = pd.read_csv(
            server_path,
            usecols=['LONGITUDE', 'LATITUDE']  # 仅加载经纬度列
        )

        # 米勒投影转换
        x_coords, y_coords = miller_to_xy(
            edge_servers_list['LONGITUDE'].values,
            edge_servers_list['LATITUDE'].values
        )
        coords = np.column_stack((x_coords, y_coords))

        # 几何变换
        transformed_coords = coordinate_transformation_pipeline(coords)

        # 根据percent百分比截取边缘服务器的数量
        total_servers = len(transformed_coords)
        sample_size = max(1, int(total_servers * percent / 100))
        filtered_idx = np.random.choice(total_servers, size=sample_size, replace=False)
        final_coords = transformed_coords[filtered_idx]

        # 仅对齐到原点（不添加安全边界）
        final_coords -= np.min(final_coords, axis=0)

        # 构建数据集
        edge_servers = pd.DataFrame({
            'X': final_coords[:, 0],
            'Y': final_coords[:, 1],
            'RADIUS': np.random.uniform(radius_low, radius_high, len(final_coords)),
        })

        num_servers = len(final_coords)
        # 真实服务器（16 核 CPU/32GB 内存 / 400GB 存储 / 50Mbps 带宽）→ 统一单位转换后，物理容量为 (32,32,40,50)（单位：0.5 核 / 1GB/10GB/1Mbps）。
        # 用正态分布 N (μ,σ²) 生成服务器资源，μ=35（贴近物理容量 (32,32,40,50) 的典型值，体现 “真实配置基准”），σ=10（让资源波动覆盖物理容量范围，体现服务器异质性）。
        resource_arr = np.random.normal(miu, sigma, size=(num_servers, 4))
        resource_arr = np.where(resource_arr < 0, 1, resource_arr)  # 负值→1，非负值保留原数值
        resource_df = pd.DataFrame(
            resource_arr,
            columns=['Resource_CPU', 'Resource_Memory', 'Resource_Storage', 'Resource_Bandwidth']  # 可根据实际需求修改列名
        )
        # 合并资源信息
        edge_servers = pd.concat([edge_servers, resource_df], axis=1)

        # 保存数据
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        edge_servers.to_csv(save_path, index=False)
        print('边缘服务器数据已保存到：', save_path)
    # 不包含原有索引列
    edge_servers = edge_servers.reset_index(drop=True)
    return edge_servers


def gen_telecom_servers_dataset(server_path, percent, radius_low, radius_high, miu, sigma, save_path):
    save_path = os.path.join(save_path, f'{percent}_miu_{miu}_sigma_{sigma}_low_{radius_low}_high_{radius_high}',
                             f'servers_pct_{percent}.csv')
    # 检查服务器数据是否存在
    if os.path.exists(save_path):
        edge_servers = pd.read_csv(save_path)
    else:
        print("生成新服务器数据集...")
        # 从 Excel 文件中读取原始数据
        df_raw = pd.read_excel(server_path)
        location_col = 'location(latitude/lontitude)'
        # 将 location 列拆分为 LATITUDE 和 LONGITUDE location 格式形如 "31.237872/121.470259"
        df_raw = (
            df_raw[location_col]
            .astype(str)
            .str.split('/', expand=True)
            .rename(columns={0: 'LATITUDE', 1: 'LONGITUDE'})
            [['LATITUDE', 'LONGITUDE']]  # 只保留需要的列
        )

        # 将经纬度列转换为浮点数（若有无效数据需要额外处理）
        df_raw['LATITUDE'] = pd.to_numeric(df_raw['LATITUDE'], errors='coerce')
        df_raw['LONGITUDE'] = pd.to_numeric(df_raw['LONGITUDE'], errors='coerce')
        # 2. 对经纬度坐标去重（只保留一个条目）
        df_unique = df_raw.drop_duplicates(subset=['LATITUDE', 'LONGITUDE'])

        # 筛选经纬度，Central Urban area, Shanghai, CN 32.67 km2  208个
        min_latitude = 31.196371
        max_latitude = 31.233580
        min_longitude = 121.423307
        max_longitude = 121.494572

        # 筛选位于陆家嘴区域内的服务器坐标
        df_unique = df_unique[
            (df_unique['LATITUDE'] >= min_latitude) &
            (df_unique['LATITUDE'] <= max_latitude) &
            (df_unique['LONGITUDE'] >= min_longitude) &
            (df_unique['LONGITUDE'] <= max_longitude)
            ]

        # 米勒投影转换
        x_coords, y_coords = miller_to_xy(
            df_unique['LONGITUDE'].values,
            df_unique['LATITUDE'].values
        )
        coords = np.column_stack((x_coords, y_coords))

        # 几何变换
        transformed_coords = coordinate_transformation_pipeline(coords)

        # 根据percent百分比截取边缘服务器的数量
        total_servers = len(transformed_coords)
        sample_size = max(1, int(total_servers * percent / 100))
        filtered_idx = np.random.choice(total_servers, size=sample_size, replace=False)
        final_coords = transformed_coords[filtered_idx]

        # 仅对齐到原点（不添加安全边界）
        final_coords -= np.min(final_coords, axis=0)

        # 构建数据集
        edge_servers = pd.DataFrame({
            'X': final_coords[:, 0],
            'Y': final_coords[:, 1],
            'RADIUS': np.random.uniform(radius_low, radius_high, len(final_coords)),
        })

        num_servers = len(final_coords)
        # 真实服务器（32 核 CPU/64GB 内存 / 1000GB 存储 / 100Mbps 带宽）→ 统一单位转换后，物理容量为 (64, 64, 100, 100)（单位：0.5 核 / 1GB/10GB/1Mbps）。
        # 用正态分布 N (μ,σ²) 生成服务器资源，μ=80（贴近物理容量(64, 64, 100, 100)的典型值，体现 “真实配置基准”），σ=20（让资源波动覆盖物理容量范围，体现服务器异质性）。
        resource_arr = np.random.normal(miu, sigma, size=(num_servers, 4))
        resource_arr = np.where(resource_arr < 0, 1, resource_arr)  # 负值→1，非负值保留原数值
        resource_df = pd.DataFrame(
            resource_arr,
            columns=['Resource_CPU', 'Resource_Memory', 'Resource_Storage', 'Resource_Bandwidth']  # 可根据实际需求修改列名
        )
        # 合并资源信息
        edge_servers = pd.concat([edge_servers, resource_df], axis=1)

        # 保存数据
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        edge_servers.to_csv(save_path, index=False)
        print('边缘服务器数据已保存到：', save_path)
        # 不包含原有索引列
    edge_servers = edge_servers.reset_index(drop=True)
    return edge_servers



