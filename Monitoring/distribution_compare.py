#!/usr/bin/env python3
"""
Generate two figures (each with 6 subplots) for paper:
- Figure 1: EUA (6 subplots)
- Figure 2: Telecom (6 subplots)
Each figure uses its own coordinate extent (no shared axes) and is saved as PNG/PDF.
EUA params: users [100,500,1000] with servers_percent=50, and users=500 with servers_percent [10,50,100]
Telecom params: (1000,50),(3000,50),(6000,50) and (3000,10),(3000,50),(3000,100)
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import seaborn as sns

# reuse dataset generator from repo
from Monitoring.user_gen import gen_dataset

# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="Plot 12 subplots (EUA + Telecom) for paper")
    p.add_argument("--out", type=str, default="./distribution_12subplots", help="output prefix (no ext)")
    p.add_argument("--dpi", type=int, default=300, help="output dpi")
    return p.parse_args()

# --- fixed settings (no config.yaml) ---
# TEST_DATA_SIZE: mapping of dataset splits to size (gen_dataset expects indexing by split name)
TEST_DATA_SIZE = {"TEST": 1}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# EUA-specific generation params
EUA_RADIUS_LOW = 1
EUA_RADIUS_HIGH = 2
EUA_MIU = 35
EUA_SIGMA = 10

# Telecom-specific generation params
TELECOM_RADIUS_LOW = 3
TELECOM_RADIUS_HIGH = 7
TELECOM_MIU = 80
TELECOM_SIGMA = 20

# server path constants (aligned with existing script)
SERVER_EUA_PATH = "../Monitoring/eua-dataset-master/edge-servers/site-optus-melbCBD.csv"
SERVER_TELECOM_PATH = "../Monitoring/dataset-telecom/data_6.1~6.30_.xlsx"

# parameter lists per user request
EUA_PARAMS = [
    (100, 50), (500, 50), (1000, 50),  # USERS vary, SERVERS_PERCENT fixed 50%
    (500, 10), (500, 50), (500, 100)   # USERS fixed 500, SERVERS_PERCENT vary
]
TELECOM_PARAMS = [
    (1000, 50), (3000, 50), (6000, 50),  # USERS vary
    (3000, 10), (3000, 50), (3000, 100)  # SERVERS vary
]

def collect_all_coords(params_list, dataset_name, server_path, save_path="./data_tmp"):
    """
    For each (user_num, servers_percent) in params_list, call gen_dataset and
    extract coordinates for plotting. Return list of dicts containing arrays.
    """
    collected = []
    for user_num, servers_percent in params_list:
        # choose dataset-specific generation parameters
        if dataset_name.lower().startswith("eua"):
            radius_low, radius_high = EUA_RADIUS_LOW, EUA_RADIUS_HIGH
            miu, sigma = EUA_MIU, EUA_SIGMA
        else:
            radius_low, radius_high = TELECOM_RADIUS_LOW, TELECOM_RADIUS_HIGH
            miu, sigma = TELECOM_MIU, TELECOM_SIGMA

        dataset = gen_dataset(
            user_num, TEST_DATA_SIZE, server_path, save_path, servers_percent,
            radius_low, radius_high, miu, sigma, device, {'TEST': []}, dataset_name
        )
        loader = DataLoader(dataset=dataset['TEST'], batch_size=1, shuffle=False)
        servers, users, connect, latency = next(iter(loader))
        servers = servers.squeeze(0).cpu().numpy()
        users = users.squeeze(0).cpu().numpy()

        users_x = users[:, 0] * 100
        users_y = users[:, 1] * 100
        servers_x = servers[:, 0] * 100
        servers_y = servers[:, 1] * 100
        servers_r = servers[:, 2] * 100

        collected.append({
            "dataset": dataset_name,
            "user_num": int(user_num),
            "servers_percent": int(servers_percent),
            "users_x": users_x, "users_y": users_y,
            "servers_x": servers_x, "servers_y": servers_y,
            "servers_r": servers_r,
            "n_users": users_x.size, "n_servers": servers_x.size
        })
    return collected

def compute_global_extent(all_entries, pad_ratio=0.05):
    xs = []
    ys = []
    for e in all_entries:
        xs.extend(e["users_x"].tolist())
        xs.extend(e["servers_x"].tolist())
        ys.extend(e["users_y"].tolist())
        ys.extend(e["servers_y"].tolist())
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    xpad = (xmax - xmin) * pad_ratio
    ypad = (ymax - ymin) * pad_ratio
    return (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)

def make_figure_single(entries, out_prefix, title_prefix, dpi=300):
    """
    Create one figure (2x3) from six entries. Each figure uses its own extent.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10
    })

    fig, axes = plt.subplots(2, 3, figsize=(14, 5.5), dpi=dpi)
    axes = axes.flatten()

    # compute extent for this set only
    extent = compute_global_extent(entries, pad_ratio=0.04)
    xmin, xmax, ymin, ymax = extent

    # colors/markers
    user_color = "#1f77b4"
    server_color = "#d62728"
    circle_face = "#fde0dd"

    for idx, entry in enumerate(entries):
        ax = axes[idx]
        for sx, sy, sr in zip(entry["servers_x"], entry["servers_y"], entry["servers_r"]):
            c = patches.Circle((sx, sy), sr, facecolor=circle_face, edgecolor=server_color,
                               linestyle='--', linewidth=0.8, alpha=0.25, zorder=1)
            ax.add_patch(c)

        ax.scatter(entry["users_x"], entry["users_y"], c=user_color, s=16,
                   alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3, label="Users")
        ax.scatter(entry["servers_x"], entry["servers_y"], c=server_color, s=50,
                   marker="^", edgecolors="black", linewidths=0.6, zorder=4, label="Servers")

        dataset_label = entry["dataset"].upper()
        ax.set_title(f"{dataset_label} — Users={entry['user_num']}, Servers={entry['servers_percent']}%", pad=6)
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

        col = idx % 3
        row = idx // 3
        if col == 0:
            ax.set_ylabel("Y (meters)")
        else:
            ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel("X (meters)")
        else:
            ax.set_xticklabels([])

    # legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Users', markerfacecolor=user_color, markersize=7, markeredgecolor='white'),
        Line2D([0], [0], marker='^', color='w', label='Servers', markerfacecolor=server_color, markersize=8, markeredgecolor='black'),
        patches.Patch(facecolor=circle_face, edgecolor=server_color, label='Server coverage', alpha=0.25)
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.97))

    # slightly lower the suptitle and tighten vertical spacing between rows
    plt.suptitle(title_prefix, fontsize=13, y=0.985)
    # tighten subplot spacing: reduce top margin and vertical/horizontal gaps
    # reduce hspace to bring row1 and row2 closer together (smaller -> tighter)
    fig.subplots_adjust(top=0.92, hspace=0.03, wspace=0.12)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    png_path = out_prefix + ".png"
    pdf_path = out_prefix + ".pdf"
    # 强制 PNG 输出为高分辨率以便论文使用；如果用户传入更高 dpi 则使用更高值
    png_dpi = max(dpi, 1200)
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    # 保存 PDF 为矢量格式（不指定 dpi），以获得最高打印/排版质量
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path} (dpi={png_dpi})  and  {pdf_path} (vector PDF)")

def main():
    args = parse_args()

    # collect coords for EUA and Telecom
    print("Collecting EUA data (this may take some seconds)...")
    eua_entries = collect_all_coords(EUA_PARAMS, "eua", SERVER_EUA_PATH, save_path="./data/eua_tmp")
    print("Collecting Telecom data (this may take some seconds)...")
    telecom_entries = collect_all_coords(TELECOM_PARAMS, "telecom", SERVER_TELECOM_PATH, save_path="./data/telecom_tmp")
    # create two separate figures (EUA and Telecom)
    make_figure_single(eua_entries, args.out + "_EUA", "EUA Dataset", dpi=args.dpi)
    make_figure_single(telecom_entries, args.out + "_Telecom", "Telecom Dataset", dpi=args.dpi)

if __name__ == "__main__":
    main()


