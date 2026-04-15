import argparse
import copy
import json
import os
import os.path as osp
import pickle
import shutil
import time
from typing import Dict, Iterable, List, Tuple

import lmdb
import numpy as np
import torch
from torch_scatter import scatter_mean

import dptb.data.AtomicDataDict as AtomicDataDict
import dptb.data._keys as _keys
from dptb.data.AtomicData import AtomicData
from dptb.data.AtomicDataDict import with_batch, with_edge_vectors
from dptb.data.build import build_dataset
from dptb.data.interfaces.ham_to_feature import block_to_feature
from dptb.nn.build import build_model
from dptb.nn.deeptb import NNENV
from dptb.nnops.loss import Loss
from dptb.utils.argcheck import get_cutoffs_from_model_options


DEFAULT_H0_KEY = "hamiltonian_0"


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_lmdb_dirs(root: str) -> List[str]:
    lmdb_dirs = []
    for dirpath, dirnames, _filenames in os.walk(root):
        if dirpath.endswith(".lmdb"):
            lmdb_dirs.append(dirpath)
            dirnames[:] = []
    return sorted(lmdb_dirs)


def _relative_lmdb_dir(root: str, lmdb_dir: str) -> str:
    return osp.relpath(lmdb_dir, root)


def _copy_non_lmdb_tree(src_root: str, dst_root: str) -> None:
    os.makedirs(dst_root, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src_root):
        rel = osp.relpath(dirpath, src_root)
        if rel == ".":
            rel = ""
        dst_dir = osp.join(dst_root, rel)
        os.makedirs(dst_dir, exist_ok=True)
        dirnames[:] = [d for d in dirnames if not d.endswith(".lmdb")]
        for filename in filenames:
            shutil.copy2(osp.join(dirpath, filename), osp.join(dst_dir, filename))


def path_size_bytes(path: str) -> int:
    if osp.isfile(path):
        return osp.getsize(path)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            total += osp.getsize(osp.join(dirpath, filename))
    return total


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}TB"


def _open_read_env(path: str) -> lmdb.Environment:
    return lmdb.open(
        path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=512,
        subdir=True,
    )


def _open_write_env(path: str, map_size: int) -> lmdb.Environment:
    os.makedirs(path, exist_ok=True)
    return lmdb.open(
        path,
        map_size=map_size,
        subdir=True,
        lock=True,
        readahead=False,
        max_dbs=1,
    )


def merge_lmdb_roots(
    delta_root: str,
    h0_root: str,
    output_root: str,
    h0_key: str = DEFAULT_H0_KEY,
    overwrite: bool = False,
) -> Dict[str, object]:
    if osp.exists(output_root):
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists")
        shutil.rmtree(output_root)

    _copy_non_lmdb_tree(delta_root, output_root)

    delta_lmdb_dirs = _iter_lmdb_dirs(delta_root)
    h0_lmdb_map = {
        _relative_lmdb_dir(h0_root, lmdb_dir): lmdb_dir for lmdb_dir in _iter_lmdb_dirs(h0_root)
    }

    total_entries = 0
    for delta_lmdb_dir in delta_lmdb_dirs:
        rel = _relative_lmdb_dir(delta_root, delta_lmdb_dir)
        if rel not in h0_lmdb_map:
            raise FileNotFoundError(f"Missing matching H0 LMDB for {rel}")

        out_lmdb_dir = osp.join(output_root, rel)
        src_size = path_size_bytes(delta_lmdb_dir) + path_size_bytes(h0_lmdb_map[rel])
        map_size = max(1 << 30, int(src_size * 4))

        delta_env = _open_read_env(delta_lmdb_dir)
        h0_env = _open_read_env(h0_lmdb_map[rel])
        out_env = _open_write_env(out_lmdb_dir, map_size=map_size)

        with delta_env.begin() as delta_txn, h0_env.begin() as h0_txn, out_env.begin(write=True) as out_txn:
            cursor = delta_txn.cursor()
            for key, value in cursor:
                h0_value = h0_txn.get(key)
                if h0_value is None:
                    raise KeyError(f"Missing key {key!r} in H0 LMDB {h0_lmdb_map[rel]}")

                delta_record = pickle.loads(value)
                h0_record = pickle.loads(h0_value)
                delta_record[h0_key] = h0_record["hamiltonian"]
                out_txn.put(key, pickle.dumps(delta_record, protocol=pickle.HIGHEST_PROTOCOL))
                total_entries += 1

        delta_env.close()
        h0_env.close()
        out_env.close()

    return {
        "output_root": output_root,
        "size_bytes": path_size_bytes(output_root),
        "entries": total_entries,
        "h0_key": h0_key,
    }


def _dataset_args_from_input(input_json: str, root_override: str, get_h0: bool = False) -> Tuple[dict, dict, dict]:
    jdata = _load_json(input_json)
    common_options = copy.deepcopy(jdata["common_options"])
    train_options = copy.deepcopy(jdata["data_options"]["train"])
    train_options["root"] = root_override
    if get_h0:
        train_options["get_H0"] = True
    cutoff_options = dict(
        zip(
            ["r_max", "er_max", "oer_max"],
            get_cutoffs_from_model_options(jdata["model_options"]),
        )
    )
    return cutoff_options, train_options, common_options


def _build_context_dataset(input_json: str, root_override: str) -> Tuple[object, dict]:
    cutoff_options, train_options, common_options = _dataset_args_from_input(
        input_json=input_json,
        root_override=root_override,
        get_h0=False,
    )
    dataset = build_dataset(**cutoff_options, **train_options, **common_options)
    return dataset, _load_json(input_json)


def _record_to_atomicdata(record: dict, info: dict) -> AtomicData:
    return AtomicData.from_points(
        pos=np.asarray(record[AtomicDataDict.POSITIONS_KEY]).reshape(-1, 3),
        cell=np.asarray(record[AtomicDataDict.CELL_KEY]).reshape(3, 3),
        atomic_numbers=np.asarray(record[AtomicDataDict.ATOMIC_NUMBERS_KEY]),
        pbc=record[AtomicDataDict.PBC_KEY],
        **info,
    )


def materialize_h0_features(
    input_root: str,
    output_root: str,
    input_json: str,
    h0_key: str = DEFAULT_H0_KEY,
    drop_raw_h0: bool = True,
    overwrite: bool = False,
) -> Dict[str, object]:
    if osp.exists(output_root):
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists")
        shutil.rmtree(output_root)

    dataset, _ = _build_context_dataset(input_json=input_json, root_override=input_root)
    _copy_non_lmdb_tree(input_root, output_root)

    total_entries = 0
    for lmdb_dir in _iter_lmdb_dirs(input_root):
        rel = _relative_lmdb_dir(input_root, lmdb_dir)
        out_lmdb_dir = osp.join(output_root, rel)
        map_size = max(1 << 30, int(path_size_bytes(lmdb_dir) * 4))
        in_env = _open_read_env(lmdb_dir)
        out_env = _open_write_env(out_lmdb_dir, map_size=map_size)
        folder_name = osp.basename(lmdb_dir)
        info = copy.deepcopy(dataset.info_files[folder_name])

        with in_env.begin() as in_txn, out_env.begin(write=True) as out_txn:
            cursor = in_txn.cursor()
            for key, value in cursor:
                record = pickle.loads(value)
                if h0_key not in record:
                    raise KeyError(f"{h0_key} not found in record under {lmdb_dir}")

                atomicdata = _record_to_atomicdata(record, info=info)
                block_to_feature(
                    atomicdata,
                    dataset.type_mapper,
                    record[h0_key],
                    False,
                    dataset.orthogonal,
                    node_field=_keys.NODE_H0_KEY,
                    edge_field=_keys.EDGE_H0_KEY,
                )

                record[_keys.NODE_H0_KEY] = atomicdata[_keys.NODE_H0_KEY].cpu().numpy()
                record[_keys.EDGE_H0_KEY] = atomicdata[_keys.EDGE_H0_KEY].cpu().numpy()
                if drop_raw_h0:
                    del record[h0_key]

                out_txn.put(key, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                total_entries += 1

        in_env.close()
        out_env.close()

    return {
        "output_root": output_root,
        "size_bytes": path_size_bytes(output_root),
        "entries": total_entries,
        "dropped_raw_h0": drop_raw_h0,
    }


def materialize_main_features(
    input_root: str,
    output_root: str,
    input_json: str,
    drop_raw_hamiltonian: bool = True,
    drop_raw_overlap: bool = True,
    overwrite: bool = False,
) -> Dict[str, object]:
    if osp.exists(output_root):
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists")
        shutil.rmtree(output_root)

    dataset, _ = _build_context_dataset(input_json=input_json, root_override=input_root)
    _copy_non_lmdb_tree(input_root, output_root)

    total_entries = 0
    for lmdb_dir in _iter_lmdb_dirs(input_root):
        rel = _relative_lmdb_dir(input_root, lmdb_dir)
        out_lmdb_dir = osp.join(output_root, rel)
        map_size = max(1 << 30, int(path_size_bytes(lmdb_dir) * 4))
        in_env = _open_read_env(lmdb_dir)
        out_env = _open_write_env(out_lmdb_dir, map_size=map_size)
        folder_name = osp.basename(lmdb_dir)
        info = copy.deepcopy(dataset.info_files[folder_name])

        with in_env.begin() as in_txn, out_env.begin(write=True) as out_txn:
            cursor = in_txn.cursor()
            for key, value in cursor:
                record = pickle.loads(value)
                atomicdata = _record_to_atomicdata(record, info=info)
                block_to_feature(
                    atomicdata,
                    dataset.type_mapper,
                    record.get("hamiltonian", False),
                    record.get("overlap", False),
                    dataset.orthogonal,
                )

                record[_keys.NODE_FEATURES_KEY] = atomicdata[_keys.NODE_FEATURES_KEY].cpu().numpy()
                record[_keys.EDGE_FEATURES_KEY] = atomicdata[_keys.EDGE_FEATURES_KEY].cpu().numpy()
                if _keys.NODE_OVERLAP_KEY in atomicdata:
                    record[_keys.NODE_OVERLAP_KEY] = atomicdata[_keys.NODE_OVERLAP_KEY].cpu().numpy()
                if _keys.EDGE_OVERLAP_KEY in atomicdata:
                    record[_keys.EDGE_OVERLAP_KEY] = atomicdata[_keys.EDGE_OVERLAP_KEY].cpu().numpy()

                if drop_raw_hamiltonian and "hamiltonian" in record:
                    del record["hamiltonian"]
                if drop_raw_overlap and "overlap" in record:
                    del record["overlap"]

                out_txn.put(key, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                total_entries += 1

        in_env.close()
        out_env.close()

    return {
        "output_root": output_root,
        "size_bytes": path_size_bytes(output_root),
        "entries": total_entries,
        "dropped_raw_hamiltonian": drop_raw_hamiltonian,
        "dropped_raw_overlap": drop_raw_overlap,
    }


def _sync_device(device: str) -> None:
    if isinstance(device, torch.device):
        device = str(device)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def _build_h0_model_from_input(input_json: str, device_override: str = None) -> NNENV:
    jdata = _load_json(input_json)
    model_options = copy.deepcopy(jdata["model_options"])
    common_options = copy.deepcopy(jdata["common_options"])
    model_options["embedding"] = copy.deepcopy(model_options["embedding"])
    model_options["embedding"]["method"] = "lem_moe_v3_h0"
    model_options["embedding"]["use_h0_init"] = True
    model_options["embedding"]["h0_node_mode"] = "direct"
    model_options["embedding"]["fallback_to_hamiltonian"] = False
    if device_override is not None:
        common_options["device"] = device_override
    model = NNENV(**model_options, **common_options)
    model = model.to(common_options["device"])
    model.eval()
    return model


def _build_plain_model_from_input(input_json: str, device_override: str = None):
    jdata = _load_json(input_json)
    common_options = copy.deepcopy(jdata["common_options"])
    if device_override is not None:
        common_options["device"] = device_override
    model = build_model(
        checkpoint=None,
        model_options=copy.deepcopy(jdata["model_options"]),
        common_options=common_options,
    )
    model = model.to(common_options["device"])
    model.eval()
    return model


def _build_plain_loss_from_input(input_json: str, model, device_override: str = None):
    jdata = _load_json(input_json)
    common_options = copy.deepcopy(jdata["common_options"])
    if device_override is not None:
        common_options["device"] = device_override
    loss = Loss(
        **copy.deepcopy(jdata["train_options"]["loss_options"]["train"]),
        **common_options,
        idp=model.hamiltonian.idp,
    )
    if isinstance(loss, torch.nn.Module):
        loss = loss.to(common_options["device"])
        loss.eval()
    return loss


@torch.no_grad()
def _run_init_stage(model: NNENV, sample) -> None:
    data = sample.to(model.device).to_dict()
    data = with_edge_vectors(data, with_lengths=True)
    data = with_batch(data)

    edge_index = data[_keys.EDGE_INDEX_KEY]
    edge_vector = data[_keys.EDGE_VECTORS_KEY]
    edge_sh = model.embedding.sh(edge_vector[:, [1, 2, 0]])
    edge_length = data[_keys.EDGE_LENGTH_KEY]

    data = model.embedding.onehot(data)
    edge_one_hot = model.embedding.edge_one_hot(data)
    node_one_hot = data[_keys.NODE_ATTRS_KEY]
    atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
    bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
    batch = data[_keys.BATCH_KEY]
    global_feat = scatter_mean(node_one_hot, batch, dim=0)
    model.embedding.router(global_feat)
    model.embedding.init_layer(
        data,
        edge_index,
        atom_type,
        bond_type,
        edge_sh,
        edge_length,
        edge_one_hot,
    )


def benchmark_h0_pipeline(
    input_json: str,
    dataset_root: str,
    samples: int = 10,
    repeats: int = 20,
    warmup: int = 2,
    device_override: str = None,
) -> Dict[str, object]:
    cutoff_options, train_options, common_options = _dataset_args_from_input(
        input_json=input_json,
        root_override=dataset_root,
        get_h0=True,
    )
    dataset = build_dataset(**cutoff_options, **train_options, **common_options)
    model = _build_h0_model_from_input(input_json=input_json, device_override=device_override)

    n = min(samples, len(dataset))
    indices = list(range(n))

    for _ in range(warmup):
        for idx in indices:
            sample = dataset[idx]
            _run_init_stage(model, sample)
            _sync_device(model.device)

    dataset_get_times = []
    init_stage_times = []
    end_to_end_times = []

    for _ in range(repeats):
        for idx in indices:
            t0 = time.perf_counter()
            sample = dataset[idx]
            t1 = time.perf_counter()
            _sync_device(model.device)
            t2 = time.perf_counter()
            _run_init_stage(model, sample)
            _sync_device(model.device)
            t3 = time.perf_counter()

            dataset_get_times.append((t1 - t0) * 1000.0)
            init_stage_times.append((t3 - t2) * 1000.0)
            end_to_end_times.append((t3 - t0) * 1000.0)

    def _stats(values: List[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
        }

    return {
        "dataset_root": dataset_root,
        "dataset_size_bytes": path_size_bytes(dataset_root),
        "num_samples": n,
        "repeats": repeats,
        "dataset_get": _stats(dataset_get_times),
        "init_stage": _stats(init_stage_times),
        "end_to_end": _stats(end_to_end_times),
    }


@torch.no_grad()
def _run_plain_forward_loss(model, loss_fn, sample) -> None:
    batch = sample.to(model.device)
    ref = AtomicData.to_AtomicDataDict(batch)
    pred = model({k: v for k, v in ref.items()})
    loss = loss_fn(pred, ref)
    if isinstance(loss, torch.Tensor):
        _ = loss.detach()


def benchmark_plain_pipeline(
    input_json: str,
    dataset_root: str,
    samples: int = 10,
    repeats: int = 20,
    warmup: int = 2,
    device_override: str = None,
) -> Dict[str, object]:
    cutoff_options, train_options, common_options = _dataset_args_from_input(
        input_json=input_json,
        root_override=dataset_root,
        get_h0=False,
    )
    dataset = build_dataset(**cutoff_options, **train_options, **common_options)
    model = _build_plain_model_from_input(input_json=input_json, device_override=device_override)
    loss_fn = _build_plain_loss_from_input(input_json=input_json, model=model, device_override=device_override)

    n = min(samples, len(dataset))
    indices = list(range(n))

    for _ in range(warmup):
        for idx in indices:
            sample = dataset[idx]
            _run_plain_forward_loss(model, loss_fn, sample)
            _sync_device(model.device)

    dataset_get_times = []
    forward_loss_times = []
    end_to_end_times = []

    for _ in range(repeats):
        for idx in indices:
            t0 = time.perf_counter()
            sample = dataset[idx]
            t1 = time.perf_counter()
            _sync_device(model.device)
            t2 = time.perf_counter()
            _run_plain_forward_loss(model, loss_fn, sample)
            _sync_device(model.device)
            t3 = time.perf_counter()

            dataset_get_times.append((t1 - t0) * 1000.0)
            forward_loss_times.append((t3 - t2) * 1000.0)
            end_to_end_times.append((t3 - t0) * 1000.0)

    def _stats(values: List[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
        }

    return {
        "dataset_root": dataset_root,
        "dataset_size_bytes": path_size_bytes(dataset_root),
        "num_samples": n,
        "repeats": repeats,
        "dataset_get": _stats(dataset_get_times),
        "forward_loss": _stats(forward_loss_times),
        "end_to_end": _stats(end_to_end_times),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Helpers for H0 LMDB merging, prepacking, and benchmarking.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--delta-root", required=True)
    merge_parser.add_argument("--h0-root", required=True)
    merge_parser.add_argument("--output-root", required=True)
    merge_parser.add_argument("--h0-key", default=DEFAULT_H0_KEY)
    merge_parser.add_argument("--overwrite", action="store_true")

    precompute_parser = subparsers.add_parser("precompute")
    precompute_parser.add_argument("--input-root", required=True)
    precompute_parser.add_argument("--output-root", required=True)
    precompute_parser.add_argument("--input-json", required=True)
    precompute_parser.add_argument("--h0-key", default=DEFAULT_H0_KEY)
    precompute_parser.add_argument("--keep-raw-h0", action="store_true")
    precompute_parser.add_argument("--overwrite", action="store_true")

    precompute_main_parser = subparsers.add_parser("precompute-main")
    precompute_main_parser.add_argument("--input-root", required=True)
    precompute_main_parser.add_argument("--output-root", required=True)
    precompute_main_parser.add_argument("--input-json", required=True)
    precompute_main_parser.add_argument("--keep-raw-hamiltonian", action="store_true")
    precompute_main_parser.add_argument("--keep-raw-overlap", action="store_true")
    precompute_main_parser.add_argument("--overwrite", action="store_true")

    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument("--dataset-root", required=True)
    bench_parser.add_argument("--input-json", required=True)
    bench_parser.add_argument("--samples", type=int, default=10)
    bench_parser.add_argument("--repeats", type=int, default=20)
    bench_parser.add_argument("--warmup", type=int, default=2)
    bench_parser.add_argument("--device", default=None)
    bench_parser.add_argument("--output-json", default=None)

    bench_plain_parser = subparsers.add_parser("bench-plain")
    bench_plain_parser.add_argument("--dataset-root", required=True)
    bench_plain_parser.add_argument("--input-json", required=True)
    bench_plain_parser.add_argument("--samples", type=int, default=10)
    bench_plain_parser.add_argument("--repeats", type=int, default=20)
    bench_plain_parser.add_argument("--warmup", type=int, default=2)
    bench_plain_parser.add_argument("--device", default=None)
    bench_plain_parser.add_argument("--output-json", default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "merge":
        result = merge_lmdb_roots(
            delta_root=args.delta_root,
            h0_root=args.h0_root,
            output_root=args.output_root,
            h0_key=args.h0_key,
            overwrite=args.overwrite,
        )
    elif args.command == "precompute":
        result = materialize_h0_features(
            input_root=args.input_root,
            output_root=args.output_root,
            input_json=args.input_json,
            h0_key=args.h0_key,
            drop_raw_h0=not args.keep_raw_h0,
            overwrite=args.overwrite,
        )
    elif args.command == "precompute-main":
        result = materialize_main_features(
            input_root=args.input_root,
            output_root=args.output_root,
            input_json=args.input_json,
            drop_raw_hamiltonian=not args.keep_raw_hamiltonian,
            drop_raw_overlap=not args.keep_raw_overlap,
            overwrite=args.overwrite,
        )
    elif args.command == "bench-plain":
        result = benchmark_plain_pipeline(
            input_json=args.input_json,
            dataset_root=args.dataset_root,
            samples=args.samples,
            repeats=args.repeats,
            warmup=args.warmup,
            device_override=args.device,
        )
        if args.output_json is not None:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
    else:
        result = benchmark_h0_pipeline(
            input_json=args.input_json,
            dataset_root=args.dataset_root,
            samples=args.samples,
            repeats=args.repeats,
            warmup=args.warmup,
            device_override=args.device,
        )
        if args.output_json is not None:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
