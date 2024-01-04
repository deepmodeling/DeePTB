import inspect
import os
from importlib import import_module

from dptb.data.dataset import DefaultDataset
from dptb import data
from dptb.data.transforms import TypeMapper, OrbitalMapper
from dptb.data import AtomicDataset, register_fields
from dptb.utils import instantiate, get_w_prefix
from dptb.utils.tools import j_loader
from dptb.utils.argcheck import normalize_setinfo


def dataset_from_config(config, prefix: str = "dataset") -> AtomicDataset:
    """initialize database based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Examples see tests/data/test_dataset.py TestFromConfig
    and tests/datasets/test_simplest.py

    Args:

    config (dict, nequip.utils.Config): dict/object that store all the parameters
    prefix (str): Optional. The prefix of all dataset parameters

    Return:

    dataset (nequip.data.AtomicDataset)
    """

    config_dataset = config.get(prefix, None)
    if config_dataset is None:
        raise KeyError(f"Dataset with prefix `{prefix}` isn't present in this config!")

    if inspect.isclass(config_dataset):
        # user define class
        class_name = config_dataset
    else:
        try:
            module_name = ".".join(config_dataset.split(".")[:-1])
            class_name = ".".join(config_dataset.split(".")[-1:])
            class_name = getattr(import_module(module_name), class_name)
        except Exception:
            # ^ TODO: don't catch all Exception
            # default class defined in nequip.data or nequip.dataset
            dataset_name = config_dataset.lower()

            class_name = None
            for k, v in inspect.getmembers(data, inspect.isclass):
                if k.endswith("Dataset"):
                    if k.lower() == dataset_name:
                        class_name = v
                    if k[:-7].lower() == dataset_name:
                        class_name = v
                elif k.lower() == dataset_name:
                    class_name = v

    if class_name is None:
        raise NameError(f"dataset type {dataset_name} does not exists")

    # if dataset r_max is not found, use the universal r_max
    atomicdata_options_key = "AtomicData_options"
    prefixed_eff_key = f"{prefix}_{atomicdata_options_key}"
    config[prefixed_eff_key] = get_w_prefix(
        atomicdata_options_key, {}, prefix=prefix, arg_dicts=config
    )
    config[prefixed_eff_key]["r_max"] = get_w_prefix(
        "r_max",
        prefix=prefix,
        arg_dicts=[config[prefixed_eff_key], config],
    )

    config[prefixed_eff_key]["er_max"] = get_w_prefix(
        "er_max",
        prefix=prefix,
        arg_dicts=[config[prefixed_eff_key], config],
    )

    config[prefixed_eff_key]["oer_max"] = get_w_prefix(
        "oer_max",
        prefix=prefix,
        arg_dicts=[config[prefixed_eff_key], config],
    )

    # Build a TypeMapper from the config
    type_mapper, _ = instantiate(TypeMapper, prefix=prefix, optional_args=config)

    # Register fields:
    # This might reregister fields, but that's OK:
    instantiate(register_fields, all_args=config)

    instance, _ = instantiate(
        class_name,
        prefix=prefix,
        positional_args={"type_mapper": type_mapper},
        optional_args=config,
    )

    return instance


def build_dataset(set_options, common_options):

    dataset_type = set_options.get("type", "DefaultDataset")

    # input in set_option for Default Dataset:
    # "root": main dir storing all trajectory folders.
    #         that is, each subfolder of root contains a trajectory.
    # "prefix": optional, load selected trajectory folders.
    # "get_Hamiltonian": load the Hamiltonian file to edges of the graph or not.
    # "get_eigenvalues": load the eigenvalues to the graph or not.
    # "setinfo": MUST HAVE, the name of the json file used to build dataset.
    # Example:
    # "train": {
    #        "type": "DefaultDataset",
    #        "root": "foo/bar/data_files_here",
    #        "prefix": "traj",
    #        "setinfo": "with_pbc.json"
    #    }
    if dataset_type == "DefaultDataset":
        # See if we can get a OrbitalMapper.
        if "basis" in common_options:
            idp = OrbitalMapper(common_options["basis"])
        else:
            idp = None

        # Explore the dataset's folder structure.
        root = set_options["root"]
        prefix = set_options.get("prefix", None)
        include_folders = []
        for dir_name in os.listdir(root):
            if os.path.isdir(os.path.join(root, dir_name)):
                if prefix is not None:
                    if dir_name[:len(prefix)] == prefix:
                        include_folders.append(dir_name)
                else:
                    include_folders.append(dir_name)

        # We need to check the `setinfo.json` very carefully here.
        # Different `setinfo` points to different dataset, 
        # even if the data files in `root` are basically the same.
        info_files = {}

        # See if a public info is provided.
        if "info.json" in os.listdir(root):
            public_info = j_loader(os.path.join(root, "info.json"))
            public_info = normalize_setinfo(public_info)
            print("A public `info.json` file is provided, and will be used by the subfolders who do not have their own `info.json` file.")
        else:
            public_info = None

        # Load info in each trajectory folders seperately.
        for file in include_folders:
            if "info.json" in os.listdir(os.path.join(root, file)):
                # use info provided in this trajectory.
                info = j_loader(os.path.join(root, file, "info.json"))
                info = normalize_setinfo(info)
                info_files[file] = info
            elif public_info is not None:
                # use public info instead
                info_files[file] = public_info
            else:
                # no info for this file
                raise Exception(f"info.json is not properly provided for `{file}`.")
            
        # We will sort the info_files here.
        # The order itself is not important, but must be consistant for the same list.
        info_files = {key: info_files[key] for key in sorted(info_files)}
        
        dataset = DefaultDataset(
            root=root,
            type_mapper=idp,
            get_Hamiltonian=set_options.get("get_Hamiltonian", False),
            get_eigenvalues=set_options.get("get_eigenvalues", False),
            info_files = info_files
        )

    else:
        raise ValueError(f"Not support dataset type: {type}.")

    return dataset
