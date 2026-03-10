import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import logging
import re  # Added for the manual patch logic
from pathlib import Path
from ase.io import read
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erf
from dprep.post_analysis_tools import _extract_band_data
from dptb.data import build_dataset
from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band
from dptb.utils.tools import j_loader
from tqdm import tqdm
import warnings
import torch
from dptb.nnops.loss import HamilLossAnalysis
from dptb.data.dataloader import DataLoader
from dptb.data import AtomicData
from dprep.dptb_dpdispatcher import DPTBDpdispatcher, parse_orbital_files

warnings.filterwarnings('ignore')


def setup_logging(log_file):
    """
    Set up logging configuration to output to both file and console
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def read_kpt_nscf(filename):
    """
    Read kpath and klabels information from KPT.nscf file
    """
    kpath = []
    klabels = []

    with open(filename, 'r') as file:
        lines = file.readlines()

    # Skip the first 3 lines (K_POINTS, count, Line)
    for line in lines[3:]:
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        parts = line.split()
        if len(parts) >= 4:
            # Extract first 4 columns as kpath
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            weight = int(parts[3])
            kpath.append([x, y, z, weight])

            # Extract labels (content after # symbol if exists)
            if '#' in line:
                label = line.split('#')[1].strip()
                klabels.append(label)
            else:
                klabels.append('')  # Add empty string if no label

    return kpath, klabels


def delta_band(band_energy1, band_energy2, n_elec, wk, smearing, smearing_sigma, return_all=False, efermi_1=None,
               efermi_2=None):
    """
    Calculate the "distance" between two band structures.
    """

    # occupation function
    def f_occ(x, x0):
        if smearing == 'gaussian':
            return 0.5 * (1.0 - erf((x - x0) / smearing_sigma)) \
                if smearing_sigma > 0 else 0.5 * (1 - np.sign(x - x0))
        elif smearing == 'fermi-dirac':
            return 1.0 / (1.0 + np.exp((x - x0) / smearing_sigma)) \
                if smearing_sigma > 0 else 0.5 * (1 - np.sign(x - x0))
        else:
            raise ValueError('Unknown smearing method: %s' % smearing)

    def efermi(wk, be, n_elec):
        _nmax = np.sum(wk * f_occ(be, np.max(be)))
        _delta = (_nmax - n_elec) / n_elec
        if np.abs(_delta) < 1e-4 or np.abs(_delta * n_elec) <= 0.01:
            print(f"WARNING: all bands are occupied, error of this estimation: {_delta:.4%}")
            return np.max(be)
        else:
            if _delta < 0:
                raise ValueError(f"""WARNING: maximum possible number of electrons in band structure not-enough:
{n_elec:.4f} vs. {_nmax:.4f} (nelec vs. nmax). This is always because of too small basis size and all
bands are occupied, otherwise please check your data.""")
            return brentq(lambda x: np.sum(wk * f_occ(be, x)) - n_elec, np.min(be), np.max(be))

    min_n_band = min(band_energy1.shape[0], band_energy2.shape[0])
    band_energy1 = band_energy1[:min_n_band]
    band_energy2 = band_energy2[:min_n_band]

    # convert to arrays for convenience
    be1 = np.expand_dims(band_energy1.T, 0)
    be2 = np.expand_dims(band_energy2.T, 0)

    # convert spinless weight to the one with spin
    nspin = len(be1)
    wk = [1] * band_energy1.shape[1]
    wk = np.array(wk).reshape(1, len(wk), 1) * (2 / nspin)
    wk = 2 * wk / np.sum(wk)  # normalize the weight

    n_elec1, n_elec2 = n_elec if isinstance(n_elec, tuple) else (n_elec, n_elec)

    assert smearing_sigma >= 0 and n_elec1 > 0 and n_elec2 > 0

    if not efermi_1:
        efermi_1 = efermi(wk, be1, n_elec1)
    if not efermi_2:
        efermi_2 = efermi(wk, be2, n_elec2)

    # geometrically averaged occupation (no efermi_shift)
    f_avg = np.sqrt(f_occ(be1, efermi_1) * f_occ(be2, efermi_2))
    res = minimize_scalar(lambda omega: np.sum(wk * f_avg * (be1 - be2 + omega) ** 2),
                          (-10, 10), method='brent')
    omega = res.x
    eta = np.sqrt(res.fun / np.sum(wk * f_avg))
    eta_max = np.max(np.abs(be1 - be2 + omega))

    return (eta, eta_max) if not return_all else (eta, eta_max, efermi_1, efermi_2, omega)


def plot_band_comparison_optimized(
        data_dict1: dict,
        data_dict2: dict,
        kpt_lines_ref: dict,
        n_occupied_bands: int,
        formula: str,
        output_path: str,
        ham_err: float,
        labels: list = ['DPTB', 'ABACUS'],
        colors: list = ['blue', 'red'],
        linestyles: list = ['-', '--'],
        figsize: tuple = (8, 6),
        calculate_error: bool = True,
        smearing: str = 'gaussian',
        smearing_sigma: float = 0.01,
        plot_flag: bool = True,
        is_soc: bool = False
):
    """
    Optimized band comparison plotting function
    """
    # Check data integrity
    required_keys = ['e_vbm_max', 'bands']
    for i, data in enumerate([data_dict1, data_dict2], 1):
        if not all(key in data for key in required_keys):
            raise ValueError(f"data_dict{i} missing required keys: {required_keys}")

    # Use the first dataset as reference for k-point processing
    bands_ref = data_dict1['bands']

    # Simulate PostBand.rearrange_plotdata functionality
    from abacustest.lib_model.model_012_band import PostBand
    band_idx, symbol_index, symbols = PostBand.rearrange_plotdata(bands_ref, kpt_lines_ref)

    # Calculate band error (if needed)
    error_info = ""
    eta, eta_max = None, None
    if calculate_error:
        try:
            # Prepare data for delta_band calculation
            bands1 = data_dict1['bands']  # shape: [iband][ik]
            bands2 = data_dict2['bands']  # shape: [iband][ik]

            # Create k-point weights (uniform weights)
            wk = [1.0] * bands1.shape[1]

            # [Full SOC 更新] 调整 delta_band 的输入电子数
            total_electrons = n_occupied_bands * 1 if is_soc else n_occupied_bands * 2

            eta, eta_max = delta_band(
                bands1, bands2, total_electrons, wk,
                smearing, smearing_sigma, return_all=False
            )
            error_info = f"\nBand Error eta: {eta:.4f} eV"
        except Exception as e:
            print(f"Warning: Error calculation failed: {e}")
            error_info = ""
    if not plot_flag:
        return eta, eta_max

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get data
    datasets = [data_dict1, data_dict2]

    # Calculate Y-axis range
    cbm_max_list = []

    for data in datasets:
        bands = data['bands']
        e_vbm_max = data['e_vbm_max']

        # Align bands relative to VBM
        bands_shifted = bands - e_vbm_max

        # Calculate CBM (conduction band minimum)
        if bands_shifted.shape[0] > n_occupied_bands:
            cbm_min = float(np.min(bands_shifted[n_occupied_bands:]))
            cbm_max_list.append(cbm_min + 2)  # CBM + 2eV as upper bound
        else:
            cbm_max_list.append(5)  # Default backup

    # Set Y-axis range: lower bound VBM-5, upper bound determined by CBM
    ylim_upper = max(cbm_max_list)

    # Plot both datasets
    plotted_labels = set()

    for i, (data, label, color, ls) in enumerate(zip(datasets, labels, colors, linestyles)):
        bands = data['bands']
        e_vbm_max = data['e_vbm_max']

        # Align bands relative to VBM
        bands_shifted = bands - e_vbm_max

        # Plot bands
        for band_num, band_data in enumerate(bands_shifted):
            label_to_use = label if (label not in plotted_labels and band_num == 0) else ""
            if label_to_use:
                plotted_labels.add(label)

            # Plot each k-point segment
            try:
                for j, idx in enumerate(band_idx):
                    if idx[2] < len(band_data) and idx[3] <= len(band_data):
                        ax.plot(range(idx[0], idx[1]), band_data[idx[2]:idx[3]],
                                linestyle=ls, color=color, linewidth=1.0, alpha=0.8,
                                label=label_to_use if j == 0 else "")
            except (IndexError, TypeError) as e:
                print(f"Warning: Error plotting band {band_num}: {e}")
                continue

    # Set figure properties
    if band_idx:
        ax.set_xlim(0, band_idx[-1][1])

    ax.set_ylim(-10, max(10, ylim_upper))

    # Set k-point labels
    if symbols is not None and symbol_index is not None:
        ax.set_xticks(symbol_index)
        ax.set_xticklabels(symbols)
        # Add vertical divider lines
        for index in symbol_index[1:-1]:
            ax.axvline(x=index, color='gray', linestyle=':', linewidth=0.6)

    # Add VBM reference line
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    # Set labels and title
    ax.set_xlabel("K points")
    ax.set_ylabel("Energy (E - E$_{VBM_{MAX}}$, eV)")

    title_note = "\n(Dash line: E$_{VBM_{MAX}}$)"
    title_text = f"Band Comparison: {formula}" + title_note + error_info
    ax.set_title(title_text)
    ax.legend()

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return eta, eta_max


def find_db_seq_id_folders(base_dir, prefix='db_seq_id_', id_list=None):
    """
    Use os.walk to find all folders starting with db_seq_id_
    """
    db_folders = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if id_list:
                for a_q_id in id_list:
                    a_q_name = prefix + str(a_q_id)
                    if dir_name == a_q_name:
                        db_folders.append(os.path.join(root, dir_name))
            else:
                if dir_name.startswith(prefix):
                    db_folders.append(os.path.join(root, dir_name))
    return db_folders


def extract_id_from_path(path):
    """
    Extract ID from path
    """
    folder_name = os.path.basename(path)
    if folder_name.startswith('db_seq_id_'):
        return folder_name.replace('db_seq_id_', '')
    return None


def process_single_task(task_dir, model, output_dir, plot_data_dir, logger, id_err_map, plot_flag):
    """
    Process single task for band calculation and plotting
    """
    task_id = extract_id_from_path(task_dir)
    if not task_id:
        logger.error(f"Failed to extract task ID from path: {task_dir}")
        raise RuntimeError
        return None
    task_id = int(task_id)

    # [Full SOC 更新] 检测模型是否为 SOC 模式
    is_soc = getattr(model.idp, "has_soc", False)

    # Check if calculated data already exists
    plot_data_file = os.path.join(plot_data_dir, f"{task_id}_plot_data.pkl")
    ham_err = 0
    if os.path.exists(plot_data_file):
        # Load existing data
        with open(plot_data_file, 'rb') as f:
            saved_data = pickle.load(f)

        data_infer = saved_data['data_infer']
        data_label = saved_data['data_label']
        kpt_lines = saved_data['kpt_lines']
        n_occupied_bands = saved_data['n_occupied_bands']
        formula = saved_data['formula']

        logger.info(f"Loaded existing data: {task_id}")
    else:
        # Recalculate
        logger.info(f"Processing task: {task_id}")

        # Build Band calculator
        bcal = Band(
            model=model,
            use_gui=False,
            results_path=None,
            device=model.device,
        )

        # Apply specific cutoffs
        bcal.cutoffs['r_max'] = {
            "As": 10,
            "Ga": 11
        }

        # Read structure file
        abacus_stru_path = os.path.join(task_dir, 'OUT.ABACUS', 'STRU.cif')
        if not os.path.exists(abacus_stru_path):
            logger.error(f"Structure file does not exist: {abacus_stru_path}")
            return None

        abacus_atoms = read(abacus_stru_path, format='cif')

        # Read k-point path
        kpt_nscf_path = os.path.join(task_dir, 'KPT.nscf')
        if not os.path.exists(kpt_nscf_path):
            logger.error(f"KPT.nscf file does not exist: {kpt_nscf_path}")
            return None

        kpath, klabels = read_kpt_nscf(kpt_nscf_path)

        # [Full SOC 更新] 定位 H0 文件
        h0_path = os.path.join(task_dir, 'hamiltonians_0.h5')

        # [Debug 新增] 定位 Full H (Label) 文件
        # 假设你的标签文件名为 hamiltonians.h5
        full_h_path = os.path.join(task_dir, 'hamiltonians_full.h5')

        # 定位 Overlap 文件
        overlap_path = os.path.join(task_dir, 'overlaps.h5')

        # === Debug 开关 ===
        USE_FULL_H_OVERRIDE = False  # 设置为 True 来测试工作流，设置为 False 来测试模型

        kpath_kwargs = {
            'kline_type': 'abacus',
            'kpath': kpath,
            'klabels': klabels,
            "override_overlap": overlap_path,
            # "override_full_h_uureal": full_h_path,
            # "override_full_h_wo_uureal": full_h_path,
            "add_h0": h0_path
        }

        # logger.warning(f"Using FULL H Override from: {full_h_path}")
        # raise RuntimeError

        if USE_FULL_H_OVERRIDE:
            if os.path.exists(full_h_path):
                kpath_kwargs["override_full_h"] = full_h_path
                # 当 override_full_h 存在时，内部逻辑会自动忽略 add_h0，但你可以显式注释掉以防万一
                # kpath_kwargs.pop("add_h0", None)
                logger.warning(f"Using FULL H Override from: {full_h_path}")
            else:
                logger.error(f"Cannot find full H file for override: {full_h_path}")
                return None

        # Calculate DPTB bands
        eigenstatus = bcal.get_bands(
            data=abacus_atoms,
            kpath_kwargs=kpath_kwargs,
        )

        # Extract ABACUS band data
        bands_ev, kpt_lines, efermi, n_occupied_bands, e_vbm_max = _extract_band_data(
            job_dir_path=Path(task_dir),
        )

        # Process DPTB data
        dptb_band = eigenstatus['eigenvalues'].T
        dptb_vbm_max = float(max(dptb_band[n_occupied_bands - 1]))

        # Prepare data
        data_infer = {
            'e_vbm_max': dptb_vbm_max,
            'bands': dptb_band
        }

        data_label = {
            'e_vbm_max': e_vbm_max,
            'bands': bands_ev,
        }

        # Get chemical formula
        formula = abacus_atoms.get_chemical_formula()

        # Save calculated data
        saved_data = {
            'data_infer': data_infer,
            'data_label': data_label,
            'kpt_lines': kpt_lines,
            'n_occupied_bands': n_occupied_bands,
            'formula': formula
        }

        with open(plot_data_file, 'wb') as f:
            pickle.dump(saved_data, f)

        logger.info(f"Calculated data saved: {plot_data_file}")

    # Plot band comparison
    output_path = os.path.join(output_dir, f"ID_{task_id}_{formula}.png")

    eta, eta_max = plot_band_comparison_optimized(
        data_dict1=data_infer,
        data_dict2=data_label,
        kpt_lines_ref=kpt_lines,
        n_occupied_bands=n_occupied_bands,
        formula=formula,
        output_path=output_path,
        labels=['DPTB', 'ABACUS'],
        colors=['blue', 'red'],
        linestyles=['-', '--'],
        calculate_error=True,
        smearing='gaussian',
        smearing_sigma=0.01,
        ham_err=0,
        plot_flag=plot_flag,
        is_soc=is_soc
    )

    logger.info(f"Band plot saved: {output_path}")

    return {
        'task_id': task_id,
        'formula': formula,
        'Ham mae': 0,
        'eta': eta,
        'eta_max': eta_max,
        'output_path': output_path
    }


def plot_eta_histogram(eta_values, output_path):
    """
    Plot statistical histogram of eta errors
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: eta distribution histogram
    ax1.hist(eta_values, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('RMS Error (eV)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of RMS Errors')
    ax1.grid(True, alpha=0.3)

    # Add statistical information
    mean_eta = np.mean(eta_values)
    median_eta = np.median(eta_values)
    std_eta = np.std(eta_values)

    ax1.axvline(mean_eta, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_eta:.4f}')
    ax1.axvline(median_eta, color='green', linestyle='--', linewidth=2, label=f'Median: {median_eta:.4f}')
    ax1.legend()

    # Right plot: cumulative distribution
    sorted_eta = np.sort(eta_values)
    cumulative = np.arange(1, len(sorted_eta) + 1) / len(sorted_eta)

    ax2.plot(sorted_eta, cumulative, 'b-', linewidth=2)
    ax2.set_xlabel('RMS Error (eV)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution of RMS Errors')
    ax2.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f"""Statistics:
    Count: {len(eta_values)}
    Mean: {mean_eta:.4f} eV
    Median: {median_eta:.4f} eV
    Std: {std_eta:.4f} eV
    Min: {min(eta_values):.4f} eV
    Max: {max(eta_values):.4f} eV"""

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Statistical histogram saved: {output_path}")


def main():
    # Parameter settings
    base_dir = './delta_H_lmdb'
    pth_path = r'./0310_soc_train/checkpoint/nnenv.best.pth'
    q_id_list = list(range(1))
    n_plot_items = 1
    # Output directory settings
    output_base_dir = os.getcwd()
    band_plots_dir = os.path.join(output_base_dir, 'band_plots')
    plot_data_dir = os.path.join(output_base_dir, 'plot_data')
    log_file = os.path.join(output_base_dir, 'processing.log')

    # Create output directories
    os.makedirs(band_plots_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)
    # Setup logging
    logger = setup_logging(log_file)
    logger.info("Starting band analysis processing")

    # Build model
    logger.info("Loading model...")

    model = build_model(checkpoint=pth_path)
    logger.info("Model loaded successfully")

    # =========================================================================
    # [PATCH START] NextHAM mask_uureal injection
    # 由于训练时没有保存 mask_uureal，这里手动检测并注入，以支持推理
    # =========================================================================
    if getattr(model.idp, "has_soc", False):
        if not hasattr(model.idp, "mask_uureal"):
            logger.warning("Detected SOC model without 'mask_uureal'. Applying manual patch for NextHAM inference...")

            # 1. 强制开启 NextHAM mask 标志
            model.idp.nextham_uureal_mask = True

            # 2. 尝试调用类自带的方法（如果环境里的库代码已经更新了）
            try:
                model.idp._apply_nextham_uureal_mask()
                logger.info("Successfully applied mask_uureal using model.idp._apply_nextham_uureal_mask()")
            except AttributeError:
                # 3. 如果已加载的对象类定义较旧，手动执行生成逻辑
                logger.warning("Method _apply_nextham_uureal_mask not found on model.idp. Injecting logic manually.")

                # 定义角动量映射 (Manual fallback)
                anglrMId_local = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}

                # 确保 orbpair_maps 存在
                if not hasattr(model.idp, "orbpair_maps"):
                    model.idp.get_orbpair_maps()

                device = model.device
                rme_size = model.idp.reduced_matrix_element

                uu_real_mask_1d = torch.zeros(rme_size, dtype=torch.bool, device=device)

                for k, sli in model.idp.orbpair_maps.items():
                    io, jo = k.split("-")
                    # 使用 regex 提取轨道类型 (e.g. "1s" -> "s")
                    il = anglrMId_local[re.findall(r"[a-z]", io)[0]]
                    jl = anglrMId_local[re.findall(r"[a-z]", jo)[0]]

                    base_dim = (2 * il + 1) * (2 * jl + 1)
                    # NextHAM 逻辑：保留每个 slice 的前 base_dim 个元素 (uu.real)
                    uu_real_mask_1d[sli.start: sli.start + base_dim] = True

                # 注入 mask
                model.idp.mask_uureal = uu_real_mask_1d

                # 更新与 ERME/NRME mask 的交集 (如果存在)
                if hasattr(model.idp, "mask_to_nrme"):
                    uu_real_mask_nrme = uu_real_mask_1d.unsqueeze(0)
                    model.idp.mask_to_nrme = model.idp.mask_to_nrme & uu_real_mask_nrme

                if hasattr(model.idp, "mask_to_erme"):
                    uu_real_mask_erme = uu_real_mask_1d.unsqueeze(0)
                    model.idp.mask_to_erme = model.idp.mask_to_erme & uu_real_mask_erme

                logger.info(
                    f"Manual patch complete. mask_uureal set ({int(uu_real_mask_1d.sum().item())} active elements).")
    # =========================================================================
    # [PATCH END]
    # =========================================================================

    # Find all db_seq_id_ folders
    logger.info("Searching for task folders...")

    db_folders = find_db_seq_id_folders(base_dir, id_list=q_id_list)
    # db_folders = [r'/home/mingkang_nt/dev_abacus/soc_test/band_GaAs_soc_lcao']
    logger.info(f"Found {len(db_folders)} task folders")

    # Batch processing
    results = []
    failed_tasks = []

    for idx, task_dir in enumerate(tqdm(db_folders, desc="Processing tasks")):
        if idx < n_plot_items:
            plot_flag = True
        else:
            plot_flag = False

        result = process_single_task(task_dir, model, band_plots_dir, plot_data_dir, logger, None, plot_flag)

        if result:
            results.append(result)
        else:
            failed_tasks.append(task_dir)

    # Collect statistics
    logger.info(f"Processing completed: {len(results)} successful, {len(failed_tasks)} failed")

    if results:
        # Extract eta values
        eta_values = [r['eta'] for r in results if r['eta'] is not None]

        if eta_values:
            # Plot statistical histogram
            histogram_path = os.path.join(output_base_dir, 'eta_statistics.png')
            plot_eta_histogram(eta_values, histogram_path)

            # Save detailed results - sort by eta in ascending order
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('eta', ascending=True)  # Sort by eta from small to large

            # Save as both CSV and Excel files
            # results_csv_path = os.path.join(output_base_dir, 'band_analysis_results.csv')
            results_excel_path = os.path.join(output_base_dir, 'band_analysis_results.xlsx')

            # results_df.to_csv(results_csv_path, index=False)
            results_df.to_excel(results_excel_path, index=False)

            # logger.info(f"Detailed results saved: {results_csv_path}")
            logger.info(f"Detailed results saved: {results_excel_path}")

            # Print statistics
            logger.info("=== Statistics ===")
            logger.info(f"Total tasks: {len(db_folders)}")
            logger.info(f"Successfully processed: {len(results)}")
            logger.info(f"Failed tasks: {len(failed_tasks)}")
            logger.info(f"Valid eta values: {len(eta_values)}")
            logger.info(f"Average RMS error: {np.mean(eta_values):.4f} eV")
            logger.info(f"Median RMS error: {np.median(eta_values):.4f} eV")
            logger.info(f"Standard deviation: {np.std(eta_values):.4f} eV")
            logger.info(f"Minimum RMS error: {min(eta_values):.4f} eV")
            logger.info(f"Maximum RMS error: {max(eta_values):.4f} eV")

    if failed_tasks:
        logger.info("Failed tasks:")
        for task in failed_tasks:
            logger.info(f"  - {task}")

    logger.info("Band analysis processing completed")


if __name__ == "__main__":
    main()