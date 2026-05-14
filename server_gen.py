
import numpy as np
import pandas as pd
import os
import math

def miller_to_xy(lons, lats):
    L = 6381372 * math.pi * 2
    W = L
    H = L / 2
    mill = 2.3
    x_rad = np.radians(lons)
    y_rad = np.radians(lats)
    y_trans = 1.25 * np.log(np.tan(0.25 * math.pi + 0.4 * y_rad))
    x = (W / 2) + (W / (2 * math.pi)) * x_rad
    y = (H / 2) - (H / (2 * mill)) * y_trans
    return x, y

def coordinate_transformation_pipeline(coords):
    coords -= np.min(coords, axis=0)
    coords /= 100
    return coords

def gen_eua_servers_dataset(server_path, percent, radius_low, radius_high, miu, sigma, save_path):
    save_path = os.path.join(save_path, f'{percent}_miu_{miu}_sigma_{sigma}_low_{radius_low}_high_{radius_high}',
                             f'servers_pct_{percent}.csv')
    if os.path.exists(save_path):
        edge_servers = pd.read_csv(save_path)
    else:
        print("生成新服务器数据集...")
        edge_servers_list = pd.read_csv(
            server_path,
            usecols=['LONGITUDE', 'LATITUDE']
        )
        x_coords, y_coords = miller_to_xy(
            edge_servers_list['LONGITUDE'].values,
            edge_servers_list['LATITUDE'].values
        )
        coords = np.column_stack((x_coords, y_coords))
        transformed_coords = coordinate_transformation_pipeline(coords)
        total_servers = len(transformed_coords)
        sample_size = max(1, int(total_servers * percent / 100))
        filtered_idx = np.random.choice(total_servers, size=sample_size, replace=False)
        final_coords = transformed_coords[filtered_idx]
        final_coords -= np.min(final_coords, axis=0)
        edge_servers = pd.DataFrame({
            'X': final_coords[:, 0],
            'Y': final_coords[:, 1],
            'RADIUS': np.random.uniform(radius_low, radius_high, len(final_coords)),
        })
        num_servers = len(final_coords)
        resource_arr = np.random.normal(miu, sigma, size=(num_servers, 4))
        resource_arr = np.where(resource_arr < 0, 1, resource_arr)
        resource_df = pd.DataFrame(
            resource_arr,
            columns=['Resource_CPU', 'Resource_Memory', 'Resource_Storage', 'Resource_Bandwidth']
        )
        edge_servers = pd.concat([edge_servers, resource_df], axis=1)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        edge_servers.to_csv(save_path, index=False)
        print('边缘服务器数据已保存到：', save_path)
    edge_servers = edge_servers.reset_index(drop=True)
    return edge_servers


def gen_telecom_servers_dataset(server_path, percent, radius_low, radius_high, miu, sigma, save_path):
    save_path = os.path.join(save_path, f'{percent}_miu_{miu}_sigma_{sigma}_low_{radius_low}_high_{radius_high}',
                             f'servers_pct_{percent}.csv')
    if os.path.exists(save_path):
        edge_servers = pd.read_csv(save_path)
    else:
        print("生成新服务器数据集...")
        df_raw = pd.read_excel(server_path)
        location_col = 'location(latitude/lontitude)'
        df_raw = (
            df_raw[location_col]
            .astype(str)
            .str.split('/', expand=True)
            .rename(columns={0: 'LATITUDE', 1: 'LONGITUDE'})
            [['LATITUDE', 'LONGITUDE']]
        )
        df_raw['LATITUDE'] = pd.to_numeric(df_raw['LATITUDE'], errors='coerce')
        df_raw['LONGITUDE'] = pd.to_numeric(df_raw['LONGITUDE'], errors='coerce')
        df_unique = df_raw.drop_duplicates(subset=['LATITUDE', 'LONGITUDE'])
        min_latitude = 31.196371
        max_latitude = 31.233580
        min_longitude = 121.423307
        max_longitude = 121.494572
        df_unique = df_unique[
            (df_unique['LATITUDE'] >= min_latitude) &
            (df_unique['LATITUDE'] <= max_latitude) &
            (df_unique['LONGITUDE'] >= min_longitude) &
            (df_unique['LONGITUDE'] <= max_longitude)
            ]
        x_coords, y_coords = miller_to_xy(
            df_unique['LONGITUDE'].values,
            df_unique['LATITUDE'].values
        )
        coords = np.column_stack((x_coords, y_coords))
        transformed_coords = coordinate_transformation_pipeline(coords)
        total_servers = len(transformed_coords)
        sample_size = max(1, int(total_servers * percent / 100))
        filtered_idx = np.random.choice(total_servers, size=sample_size, replace=False)
        final_coords = transformed_coords[filtered_idx]
        final_coords -= np.min(final_coords, axis=0)
        edge_servers = pd.DataFrame({
            'X': final_coords[:, 0],
            'Y': final_coords[:, 1],
            'RADIUS': np.random.uniform(radius_low, radius_high, len(final_coords)),
        })

        num_servers = len(final_coords)
        resource_arr = np.random.normal(miu, sigma, size=(num_servers, 4))
        resource_arr = np.where(resource_arr < 0, 1, resource_arr)
        resource_df = pd.DataFrame(
            resource_arr,
            columns=['Resource_CPU', 'Resource_Memory', 'Resource_Storage', 'Resource_Bandwidth']
        )
        edge_servers = pd.concat([edge_servers, resource_df], axis=1)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        edge_servers.to_csv(save_path, index=False)
        print('边缘服务器数据已保存到：', save_path)
    edge_servers = edge_servers.reset_index(drop=True)
    return edge_servers



