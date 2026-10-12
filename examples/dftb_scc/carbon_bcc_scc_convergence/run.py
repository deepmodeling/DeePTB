#!/usr/bin/env python3
"""Standalone Carbon BCC self-consistent total-energy k-point scan.

This example deliberately does not call the multi-structure DFTB+ pytest
benchmark. It runs one fixed Carbon BCC structure and tests convergence of the
self-consistent total energy with respect to k-point density. SCC charge
convergence is recorded as an inner-loop diagnostic, not the primary result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from ase.io import read


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dptb.kpoints.geometry import calculate_reciprocal_vectors
from dptb.kpoints.mesh import build_kmeshgrid
from dptb.nn.dftb.dftb_scc import SKSCC
from dptb.nn.dftb.scc_params import SCCParams
from dptb.nn.dftb.sk_param import SKParam
from dptb.nn.dftbsk import DFTBSK


DEFAULT_STRUCTURE = EXAMPLE_DIR / "POSCAR"
DEFAULT_SK_PATH = REPO_ROOT / "dptb" / "tests" / "data" / "dftb"
BASIS = {"C": ["2s", "2p"]}
VALENCE_ELECTRONS = {"C": 4}
R_MAX = {"C": 4.2}
SIGMA_REP = {"C": 0.561}


@dataclass(frozen=True)
class ScanCase:
    name: str
    meshgrid: Optional[list[int]]
    meshspacing: Optional[list[float]]
    mixer: str
    mix_rate: float
    smearing_method: str
    temperature_K: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        choices=("kmesh", "spacing", "mixer", "smearing"),
        default="kmesh",
        help="Numerical dimension to scan (default: kmesh).",
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=DEFAULT_STRUCTURE,
        help="Carbon BCC structure (default: the POSCAR beside this script).",
    )
    parser.add_argument(
        "--sk-path",
        type=Path,
        default=DEFAULT_SK_PATH,
        help="Directory containing C-C.skf.",
    )
    parser.add_argument(
        "--meshes",
        type=int,
        nargs="+",
        default=(8, 12, 16, 20, 24, 28, 32),
        help="Isotropic N x N x N grids for --scan kmesh.",
    )
    parser.add_argument(
        "--spacings",
        type=float,
        nargs="+",
        default=(0.20, 0.10),
        help=(
            "Isotropic reciprocal-space spacings for --scan spacing. "
            "Request the expensive legacy 0.05 case explicitly."
        ),
    )
    parser.add_argument(
        "--fixed-mesh",
        type=int,
        default=8,
        help="Isotropic grid used by mixer and smearing scans.",
    )
    parser.add_argument("--mixer", default="pulay", help="Mixer used outside a mixer scan.")
    parser.add_argument(
        "--mixers",
        nargs="+",
        default=("simple", "pulay", "broyden", "diis"),
        help="Mixers used by --scan mixer.",
    )
    parser.add_argument("--mix-rate", type=float, default=0.30)
    parser.add_argument(
        "--mix-rates",
        type=float,
        nargs="+",
        default=(0.10, 0.30),
        help="Mixing rates used by --scan mixer.",
    )
    parser.add_argument("--smearing-method", default="FD")
    parser.add_argument(
        "--smearing-methods",
        nargs="+",
        default=("FD", "G"),
        help="Occupation methods used by --scan smearing.",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Electronic temperature in K.")
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=(0.1, 300.0),
        help="Temperatures used by --scan smearing.",
    )
    parser.add_argument("--tol", type=float, default=1e-10, help="Maximum charge-residual tolerance.")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument(
        "--energy-tol-mev-per-atom",
        type=float,
        default=1.0,
        help=(
            "Total-energy k-point convergence tolerance in meV/atom. A mesh "
            "passes only if it agrees with every denser mesh in the same scan."
        ),
    )
    parser.add_argument(
        "--rotational-symmetry",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--time-inversion-symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gamma-centered",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve meshes and raw k-point counts without running SCC.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; parent directories are created.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(mesh < 1 for mesh in (*args.meshes, args.fixed_mesh)):
        raise ValueError("All mesh dimensions must be positive integers.")
    if any(spacing <= 0 for spacing in args.spacings):
        raise ValueError("All mesh spacings must be positive.")
    if any(rate <= 0 or rate > 1 for rate in (*args.mix_rates, args.mix_rate)):
        raise ValueError("All mixing rates must be in (0, 1].")
    if args.temperature <= 0 or any(temp <= 0 for temp in args.temperatures):
        raise ValueError("All electronic temperatures must be positive.")
    if args.tol <= 0:
        raise ValueError("--tol must be positive.")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be at least 1.")
    if args.energy_tol_mev_per_atom <= 0:
        raise ValueError("--energy-tol-mev-per-atom must be positive.")
    if not args.structure.is_file():
        raise FileNotFoundError(f"Structure not found: {args.structure}")
    if not (args.sk_path / "C-C.skf").is_file():
        raise FileNotFoundError(f"C-C.skf not found under: {args.sk_path}")


def build_cases(args: argparse.Namespace) -> list[ScanCase]:
    fixed_grid = [args.fixed_mesh] * 3
    common = {
        "mixer": args.mixer,
        "mix_rate": args.mix_rate,
        "smearing_method": args.smearing_method,
        "temperature_K": args.temperature,
    }

    if args.scan == "kmesh":
        return [
            ScanCase(
                name=f"grid_{mesh}x{mesh}x{mesh}",
                meshgrid=[mesh] * 3,
                meshspacing=None,
                **common,
            )
            for mesh in args.meshes
        ]
    if args.scan == "spacing":
        return [
            ScanCase(
                name=f"spacing_{spacing:g}",
                meshgrid=None,
                meshspacing=[spacing] * 3,
                **common,
            )
            for spacing in args.spacings
        ]
    if args.scan == "mixer":
        return [
            ScanCase(
                name=f"{mixer}_rate_{rate:g}",
                meshgrid=fixed_grid,
                meshspacing=None,
                mixer=mixer,
                mix_rate=rate,
                smearing_method=args.smearing_method,
                temperature_K=args.temperature,
            )
            for mixer in args.mixers
            for rate in args.mix_rates
        ]
    return [
        ScanCase(
            name=f"{method}_T_{temperature:g}K",
            meshgrid=fixed_grid,
            meshspacing=None,
            mixer=args.mixer,
            mix_rate=args.mix_rate,
            smearing_method=method,
            temperature_K=temperature,
        )
        for method in args.smearing_methods
        for temperature in args.temperatures
    ]


def resolve_mesh(structure_path: Path, case: ScanCase) -> list[int]:
    if case.meshgrid is not None:
        return case.meshgrid
    structure = read(structure_path)
    reciprocal = calculate_reciprocal_vectors(*np.asarray(structure.cell))
    return build_kmeshgrid(*reciprocal, case.meshspacing)


def build_solver(sk_path: Path) -> SKSCC:
    model = DFTBSK(
        basis=BASIS,
        skdata=str(sk_path),
        overlap=True,
        smooth_ski=True,
        dtype=torch.float64,
        r_max=R_MAX,
    )
    sk_params = SKParam(
        basis=BASIS,
        skdata=str(sk_path),
        cal_rcuts=True,
        dtype=torch.float64,
    )
    return SKSCC(
        model=model,
        params=SCCParams.from_skparam(sk_params),
        overlap=True,
        scc_dtype=torch.float64,
    )


def scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def run_case(
    solver: SKSCC,
    structure_path: Path,
    case: ScanCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    resolved_mesh = resolve_mesh(structure_path, case)
    started = time.perf_counter()
    total_energy = solver.get_total_energy(
        data=str(structure_path),
        nel_atom=VALENCE_ELECTRONS,
        sigma_rep=SIGMA_REP,
        kmeshgrid=case.meshgrid,
        kmeshspacing=case.meshspacing,
        kgamma_center=args.gamma_centered,
        krotational_symmetry=args.rotational_symmetry,
        ktime_inversion_symmetry=args.time_inversion_symmetry,
        Temp=case.temperature_K,
        AtomicData_options={"r_max": R_MAX},
        smearing_method=case.smearing_method,
        mixer=case.mixer,
        tol=args.tol,
        mix_rate=case.mix_rate,
        max_iter=args.max_iter,
    )
    elapsed = time.perf_counter() - started

    mulliken_charge = np.asarray(solver.mulliken.mul_charge, dtype=float)
    delta_charge = np.asarray(solver.mulliken.delta_charge, dtype=float)
    target_electrons = sum(solver.mulliken.per_atom_charge)
    atom_count = len(mulliken_charge)
    return {
        **asdict(case),
        "resolved_mesh": resolved_mesh,
        "raw_kpoints": int(np.prod(resolved_mesh)),
        "reduced_kpoints": int(len(solver.mulliken.klist)),
        "kpoint_weight_sum": float(np.sum(solver.mulliken.wk)),
        "scc_converged": bool(solver.is_converged),
        "scc_iterations": int(solver.n_iterations),
        "scc_final_charge_residual_e": solver.final_charge_residual,
        "scc_charge_residual_history_e": list(solver.charge_residual_history),
        "electron_count": float(np.sum(mulliken_charge)),
        "electron_count_error": float(np.sum(mulliken_charge) - target_electrons),
        "mulliken_charge_e": mulliken_charge.tolist(),
        "delta_charge_e": delta_charge.tolist(),
        "max_abs_delta_charge_e": float(np.max(np.abs(delta_charge))),
        "fermi_level_eV": scalar(solver.E_fermi),
        "total_energy_eV": float(total_energy),
        "total_energy_per_atom_eV": float(total_energy) / atom_count,
        "electronic_energy_eV": scalar(solver.elec_totE),
        "repulsive_energy_eV": scalar(solver.total_rep_energy),
        "h0_band_energy_eV": scalar(solver.elec_H0_bandE),
        "scc_shift_energy_eV": scalar(solver.scc_shift_energy),
        "wall_time_s": elapsed,
    }


def dry_run_case(structure_path: Path, case: ScanCase) -> dict[str, Any]:
    resolved_mesh = resolve_mesh(structure_path, case)
    return {
        **asdict(case),
        "resolved_mesh": resolved_mesh,
        "raw_kpoints": int(np.prod(resolved_mesh)),
    }


def add_kpoint_energy_convergence(
    results: list[dict[str, Any]],
    tolerance_meV_per_atom: float,
) -> None:
    completed = [row for row in results if row.get("total_energy_per_atom_eV") is not None]
    if not completed:
        return
    completed.sort(key=lambda row: row["raw_kpoints"])
    reference = completed[-1]
    reference_charge = np.asarray(reference["mulliken_charge_e"], dtype=float)
    for index, row in enumerate(completed):
        row["finest_mesh_case"] = reference["name"]
        row["total_energy_delta_to_finest_meV_per_atom"] = 1000.0 * (
            row["total_energy_per_atom_eV"] - reference["total_energy_per_atom_eV"]
        )
        if index == 0:
            row["successive_total_energy_delta_meV_per_atom"] = None
        else:
            previous = completed[index - 1]
            row["successive_total_energy_delta_meV_per_atom"] = 1000.0 * (
                row["total_energy_per_atom_eV"] - previous["total_energy_per_atom_eV"]
            )
        denser = completed[index + 1 :]
        if denser:
            max_delta = 1000.0 * max(
                abs(row["total_energy_per_atom_eV"] - denser_row["total_energy_per_atom_eV"])
                for denser_row in denser
            )
            row["max_abs_total_energy_delta_to_denser_meV_per_atom"] = max_delta
            row["kpoint_energy_converged_within_scan"] = max_delta <= tolerance_meV_per_atom
        else:
            row["max_abs_total_energy_delta_to_denser_meV_per_atom"] = None
            row["kpoint_energy_converged_within_scan"] = None
        row["fermi_delta_to_finest_eV"] = row["fermi_level_eV"] - reference["fermi_level_eV"]
        charge = np.asarray(row["mulliken_charge_e"], dtype=float)
        row["max_charge_delta_to_finest_e"] = float(np.max(np.abs(charge - reference_charge)))


def format_value(value: Any, precision: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def print_results(results: list[dict[str, Any]], dry_run: bool) -> None:
    if dry_run:
        columns = ("name", "resolved_mesh", "raw_kpoints")
    else:
        columns = [
            "name",
            "resolved_mesh",
            "raw_kpoints",
            "reduced_kpoints",
            "total_energy_per_atom_eV",
            "scc_converged",
            "scc_iterations",
            "scc_final_charge_residual_e",
            "electron_count_error",
            "wall_time_s",
        ]
        if any("successive_total_energy_delta_meV_per_atom" in row for row in results):
            insert_at = columns.index("scc_converged")
            columns[insert_at:insert_at] = [
                "successive_total_energy_delta_meV_per_atom",
                "max_abs_total_energy_delta_to_denser_meV_per_atom",
                "kpoint_energy_converged_within_scan",
            ]
    widths = {
        column: max(len(column), *(len(format_value(row.get(column))) for row in results))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in results:
        print("  ".join(format_value(row.get(column)).ljust(widths[column]) for column in columns))


def write_output(path: Path, args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    payload = {
        "structure": str(args.structure.resolve()),
        "sk_path": str(args.sk_path.resolve()),
        "scan": args.scan,
        "tol": args.tol,
        "max_iter": args.max_iter,
        "energy_tolerance_meV_per_atom": args.energy_tol_mev_per_atom,
        "sigma_rep": SIGMA_REP,
        "gamma_centered": args.gamma_centered,
        "rotational_symmetry": args.rotational_symmetry,
        "time_inversion_symmetry": args.time_inversion_symmetry,
        "dry_run": args.dry_run,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    cases = build_cases(args)
    if args.dry_run:
        results = [dry_run_case(args.structure, case) for case in cases]
    else:
        solver = build_solver(args.sk_path)
        results = [run_case(solver, args.structure, case, args) for case in cases]
        if args.scan in {"kmesh", "spacing"}:
            add_kpoint_energy_convergence(results, args.energy_tol_mev_per_atom)
    print_results(results, args.dry_run)
    if args.output is not None:
        write_output(args.output, args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
