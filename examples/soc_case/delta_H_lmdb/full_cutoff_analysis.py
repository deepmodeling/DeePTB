import os
import sys
import lmdb
import pickle
import json
import glob
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 确保无头模式运行
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from ase.data import chemical_symbols

# ==========================================
# 用户配置区域
# ==========================================
# [Part 4 配置] 用户自定义测试的 r_max 字典
USER_INPUT_R_MAX = {
    "As": 11,
    "Ga": 12
}
"""
[Global Loss Report]
Matrix          | Count Loss (%)  | Value Loss (%) 
--------------------------------------------------
hamiltonian     | 3.0309          | 0.000852       
overlap         | 2.9882          | 0.000303       
density_matrix  | 3.0309          | 0.197047    
"""

# [Part 1 配置] 数据源 (支持通配符，如 data.*.lmdb)
LMDB_PATH = r'data.*.lmdb'
OUTPUT_DIR = 'analysis_results'
STATS_PKL_NAME = 'stats_data.pkl'

# False = 统计所有 block (包括 PBC shift)
ZEROS_ONLY = False

# [Part 3 配置] 自动优化的目标 Count Loss
AUTO_OPT_TARGETS = [30.0, 10.0, 5.0, 1.0, 0.1]

MATRICES = ['hamiltonian', 'overlap', 'density_matrix']


# ==========================================
# 工具类与函数
# ==========================================

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_atom_symbol(z):
    if 0 < z < len(chemical_symbols):
        return chemical_symbols[z]
    return f"Z{z}"


def parse_key_fast(key_str):
    """
    解析键值，返回 i, j 以及 shift 向量
    Key format: i_j_x_y_z
    """
    parts = key_str.split('_')
    i, j = int(parts[0]), int(parts[1])
    # 解析 R vector indices
    shift = np.array([int(parts[2]), int(parts[3]), int(parts[4])], dtype=int)
    return i, j, shift


def get_pair_label(z1, z2):
    return f"{z2}-{z1}" if z1 > z2 else f"{z1}-{z2}"


# ==========================================
# Step 1: 数据提取 (Process LMDB)
# ==========================================

def run_step_1_extraction():
    print("\n" + "=" * 60)
    print("STEP 1: Processing LMDB & Extracting Stats")
    print("=" * 60)

    # 检查 pickle 是否已存在
    save_path = os.path.join(OUTPUT_DIR, STATS_PKL_NAME)
    if os.path.exists(save_path):
        print(f"Found existing pickle file: {save_path}")
        print("Loading data from disk (skipping LMDB processing)...")
        try:
            with open(save_path, 'rb') as f:
                final_data = pickle.load(f)
            print("Data loaded successfully.")
            return final_data
        except Exception as e:
            print(f"Error loading pickle: {e}")
            print("Falling back to LMDB processing...")

    # 获取 LMDB 文件列表
    lmdb_files = sorted(glob.glob(LMDB_PATH))
    if not lmdb_files:
        print(f"Error: No files found matching {LMDB_PATH}")
        sys.exit(1)

    print(f"Found {len(lmdb_files)} LMDB file(s):")
    for f in lmdb_files:
        print(f" - {f}")

    data_store = defaultdict(lambda: defaultdict(lambda: {'dist': [], 'val': []}))
    total_count = 0

    # 遍历每个 LMDB 文件
    for lmdb_file in lmdb_files:
        print(f"Opening: {lmdb_file}")
        try:
            env = lmdb.open(lmdb_file, readonly=True, lock=False)
        except Exception as e:
            print(f"Failed to open {lmdb_file}: {e}")
            continue

        with env.begin() as txn:
            cursor = txn.cursor()
            file_count = 0
            for key, value in cursor:
                try:
                    entry = pickle.loads(value)
                except:
                    continue

                if 'atomic_numbers' not in entry or 'pos' not in entry:
                    continue
                atoms = entry['atomic_numbers']
                pos = entry['pos']  # Shape: (N_atoms, 3)

                # 获取晶胞信息 (3, 3)
                cell = np.array(entry['cell']) if 'cell' in entry else None

                # 预计算单胞内距离矩阵 (用于加速 0_0_0 情况)
                dist_matrix = cdist(pos, pos)

                for mat_name in MATRICES:
                    if mat_name not in entry:
                        continue

                    # 遍历稀疏矩阵的 blocks
                    for block_key, block_data in entry[mat_name].items():
                        # 1. 解析 Key
                        i, j, shift_idx = parse_key_fast(block_key)
                        is_zero_shift = np.all(shift_idx == 0)

                        # 2. 过滤逻辑
                        if ZEROS_ONLY and not is_zero_shift:
                            continue

                        # 3. 距离计算逻辑 (包含 PBC shift)
                        if is_zero_shift:
                            # 胞内直接查表
                            dist = dist_matrix[i, j]
                        else:
                            if cell is None:
                                # 如果没有 cell 信息却出现了 shift，跳过
                                continue

                            # 计算原子 j 在邻胞中的绝对坐标
                            # pos_j_shifted = pos[j] + (n_x*L_x + n_y*L_y + n_z*L_z)
                            shift_cart = shift_idx @ cell
                            dist = np.linalg.norm(pos[i] - (pos[j] + shift_cart))

                        # 4. 存储数据
                        pair = get_pair_label(atoms[i], atoms[j])
                        val = np.mean(np.abs(block_data))

                        target = data_store[pair][mat_name]
                        target['dist'].append(dist)
                        target['val'].append(val)

                file_count += 1
                total_count += 1
                if total_count % 100 == 0:
                    print(f"Processed {total_count} entries (current file: {file_count})...", end='\r')

        env.close()

    print(f"\nProcessing complete. Total processed: {total_count}. Saving pickle...")

    # 转换为普通 dict 并保存
    final_data = {k: dict(v) for k, v in data_store.items()}

    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(save_path, 'wb') as f:
        pickle.dump(final_data, f)

    return final_data


# ==========================================
# Step 2: 绘图 (Visualization)
# ==========================================

def run_step_2_plotting(data_store):
    print("\n" + "=" * 60)
    print("STEP 2: Generating Distribution Plots")
    print("=" * 60)

    sns.set_theme(style="whitegrid")

    for pair_label, mats_data in data_store.items():
        z1, z2 = map(int, pair_label.split('-'))
        pair_name = f"{get_atom_symbol(z1)}-{get_atom_symbol(z2)}"

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        fig.suptitle(f'Distance vs Matrix Strength ({pair_name})', fontsize=16)

        has_data = False
        for idx, mat_name in enumerate(MATRICES):
            ax = axes[idx]
            data = mats_data.get(mat_name)

            if not data or len(data['dist']) == 0:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_title(mat_name)
                continue

            has_data = True
            dists = np.array(data['dist'])
            vals = np.array(data['val'])
            p80 = np.percentile(dists, 80)
            p95 = np.percentile(dists, 95)

            df = pd.DataFrame({'D': dists, 'V': vals})

            # Scatter
            sns.scatterplot(data=df, x='D', y='V', ax=ax, s=15, color=".3", alpha=0.15, linewidth=0)
            # KDE
            if len(df) > 10 and df['D'].std() > 1e-4:
                try:
                    sns.kdeplot(data=df, x='D', y='V', ax=ax, fill=True, cmap="mako", alpha=0.5, thresh=0.05)
                except:
                    pass

            # Lines
            ax.axvline(p80, color='orange', ls='--', label=f'80%: {p80:.2f}')
            ax.axvline(p95, color='red', ls=':', label=f'95%: {p95:.2f}')

            ax.set_title(mat_name.capitalize())
            ax.set_xlabel('Distance (Å)')
            if idx == 0:
                ax.set_ylabel('Mean Abs')
                ax.legend()

        if has_data:
            plt.savefig(os.path.join(OUTPUT_DIR, f"{pair_name}_stats.png"), dpi=150)
        plt.close(fig)
    print(f"Plots saved to {OUTPUT_DIR}")


# ==========================================
# Step 3: 自动优化 (Optimization)
# ==========================================

class CountCutoffSolver:
    def __init__(self, data_store):
        self.elements = set()
        self.pair_data = {}
        for pair_label, mats in data_store.items():
            z1, z2 = map(int, pair_label.split('-'))
            self.elements.add(z1)
            self.elements.add(z2)
            self.pair_data[pair_label] = {}
            for m in MATRICES:
                if m in mats and len(mats[m]['dist']) > 0:
                    dists = np.array(mats[m]['dist'])
                    vals = np.array(mats[m]['val'])
                    idx = np.argsort(dists)
                    self.pair_data[pair_label][m] = {
                        'dists': dists[idx], 'vals': vals[idx],
                        't_val': np.sum(vals), 'cnt': len(dists)
                    }
                else:
                    self.pair_data[pair_label][m] = None
        self.sorted_elements = sorted(list(self.elements))
        self.z_to_idx = {z: i for i, z in enumerate(self.sorted_elements)}
        self.num_vars = len(self.elements)

    def solve(self, mat_type, loss_pct):
        c = np.ones(self.num_vars)
        A_ub, b_ub = [], []
        req_dists = {}

        for pair, data_entry in self.pair_data.items():
            data = data_entry.get(mat_type)
            if not data or data['cnt'] == 0:
                req_dists[pair] = 0.1
                continue

            # Determine required distance
            if loss_pct <= 1e-9:
                req = data['dists'][-1]
            else:
                idx = int(data['cnt'] * (1 - loss_pct / 100.0))
                req = data['dists'][min(idx, data['cnt'] - 1)]

            req_dists[pair] = req
            if req <= 0.1:
                continue

            z1, z2 = map(int, pair.split('-'))
            row = np.zeros(self.num_vars)
            row[self.z_to_idx[z1]] -= 1
            row[self.z_to_idx[z2]] -= 1
            A_ub.append(row)
            b_ub.append(-2.0 * req)

        if not A_ub:
            return None, None

        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=(0, None), method='highs')
        if not res.success:
            return None, None

        r_dict = {z: res.x[self.z_to_idx[z]] for z in self.sorted_elements}
        return r_dict, req_dists


def run_step_3_optimization(data_store):
    print("\n" + "=" * 60)
    print("STEP 3: Auto-Optimization (Linear Programming)")
    print("=" * 60)

    solver = CountCutoffSolver(data_store)

    for mat in MATRICES:
        print(f"\n>>> MATRIX: {mat.upper()}")
        for target_loss in AUTO_OPT_TARGETS:
            label = "100% SAFE" if target_loss == 0 else f"{100 - target_loss}% Retention"
            print(f"    Target: {label} ({target_loss}% Count Loss)")

            r_dict, _ = solver.solve(mat, target_loss)
            if not r_dict:
                print("    Optimization failed.")
                continue

            # Format output
            out_dict = {get_atom_symbol(z): round(r, 4) for z, r in r_dict.items()}
            print(f"    Recommended Dictionary: {json.dumps(out_dict)}")


# ==========================================
# Step 4: 用户验证 (User Eval)
# ==========================================

def run_step_4_user_eval(data_store):
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating User Custom Input")
    print("=" * 60)
    print("User Input:")
    print(json.dumps(USER_INPUT_R_MAX, indent=4))

    # 1. 全局统计
    print(f"\n[Global Loss Report]")
    print(f"{'Matrix':<15} | {'Count Loss (%)':<15} | {'Value Loss (%)':<15}")
    print("-" * 50)

    missing = set()

    # 获取 Hamiltonian 95% 覆盖作为参考
    solver = CountCutoffSolver(data_store)
    _, h_req_95 = solver.solve('density_matrix', 5.0)

    pair_comparison = []

    for mat in MATRICES:
        l_cnt, t_cnt, l_val, t_val = 0, 0, 0.0, 0.0

        for pair, mats in data_store.items():
            z1, z2 = map(int, pair.split('-'))
            s1, s2 = get_atom_symbol(z1), get_atom_symbol(z2)
            if s1 not in USER_INPUT_R_MAX or s2 not in USER_INPUT_R_MAX:
                missing.add(s1 if s1 not in USER_INPUT_R_MAX else s2)
                continue

            uc = 0.5 * (USER_INPUT_R_MAX[s1] + USER_INPUT_R_MAX[s2])

            # Collect comparison data (only once per pair)
            if mat == 'hamiltonian':
                ref = h_req_95.get(pair, 0.0)
                if ref > 0:
                    pair_comparison.append({
                        'pair': f"{s1}-{s2}", 'user': uc, 'ref': ref, 'gap': uc - ref
                    })

            if mat in mats and len(mats[mat]['dist']) > 0:
                d = np.array(mats[mat]['dist'])
                v = np.array(mats[mat]['val'])
                mask = d > uc
                l_cnt += np.sum(mask)
                t_cnt += len(d)
                l_val += np.sum(v[mask])
                t_val += np.sum(v)

        if t_cnt > 0:
            print(f"{mat:<15} | {l_cnt / t_cnt * 100:<15.4f} | {l_val / t_val * 100:<15.6f}")
        else:
            print(f"{mat:<15} | N/A")

    if missing:
        print(f"Warning: Missing elements {missing}")

    # 2. Pair 详细对比
    print(f"\n[Detailed Pair Check vs Hamiltonian 95% Safe Dist]")
    print(f"Gap = User - Ref. Positive=Safe, Negative=Risky.")
    print("-" * 60)
    print(f"{'Pair':<12} | {'User(Å)':<10} | {'Ref95(Å)':<10} | {'Gap(Å)':<10} | {'Status'}")
    print("-" * 60)

    pair_comparison.sort(key=lambda x: x['gap'])
    for item in pair_comparison:
        gap = item['gap']
        stat = "OK"
        if gap < -0.5:
            stat = "AGGRESSIVE"
        elif gap < 0:
            stat = "Tight"
        elif gap > 2.0:
            stat = "WASTEFUL"
        elif gap > 0:
            stat = "Safe"
        print(f"{item['pair']:<12} | {item['user']:<10.3f} | {item['ref']:<10.3f} | {gap:<10.3f} | {stat}")


# ==========================================
# 主 入 口
# ==========================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    sys.stdout = Logger(os.path.join(OUTPUT_DIR, 'full_report.txt'))

    # Step 1: Extract (or load)
    data = run_step_1_extraction()

    # Step 2: Plot
    # run_step_2_plotting(data)

    # Step 3: Auto Optimize
    run_step_3_optimization(data)

    # Step 4: User Check
    run_step_4_user_eval(data)

    print("\n" + "=" * 60)
    print(f"All done. Results in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
