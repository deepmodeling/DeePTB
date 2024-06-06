import inspect
import os
from copy import deepcopy
import glob
from importlib import import_module

from dptb.data.dataset import DefaultDataset
from dptb.data.dataset._deeph_dataset import DeePHE3Dataset
from dptb.data.dataset._hdf5_dataset import HDF5Dataset
from dptb.data.dataset.lmdb_dataset import LMDBDataset
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


def build_dataset(
        # set_options
        root: str,
        type: str = "DefaultDataset",
        prefix: str = None,
        separator:str='.',
        get_Hamiltonian: bool = False,
        get_overlap: bool = False,
        get_DM: bool = False,
        get_eigenvalues: bool = False,

        # common_options
        orthogonal: bool = False,
        basis: str = None, 
        **kwargs,
        ):
    
    """
    Build a dataset based on the provided set options and common options.

    Args:
        - type (str): The type of dataset to build. Default is "DefaultDataset".
        - root (str): The main directory storing all trajectory folders.
        - prefix (str, optional): Load selected trajectory folders with the specified prefix.
        - get_Hamiltonian (bool, optional): Load the Hamiltonian file to edges of the graph or not.
        - get_eigenvalues (bool, optional): Load the eigenvalues to the graph or not.
        e.g.     
        type = "DefaultDataset",
        root = "foo/bar/data_files_here",
        prefix = "set"

        - basis (str, optional): The basis for the OrbitalMapper.

    Returns:
        dataset: The built dataset.

    Raises:
        ValueError: If the dataset type is not supported.
        Exception: If the info.json file is not properly provided for a trajectory folder.
    """
    dataset_type = type
    # See if we can get a OrbitalMapper.
    if basis is not None:
        idp = OrbitalMapper(basis=basis)
    else:
        idp = None

    if dataset_type in ["DefaultDataset", "DeePHDataset", "HDF5Dataset"]:

        # Explore the dataset's folder structure.
        #include_folders = []
        #for dir_name in os.listdir(root):
        #    dir_path = os.path.join(root, dir_name)
        #    if os.path.isdir(dir_path):
        #        # If the `processed_dataset` or other folder is here too, they do not have the proper traj data files.
        #        # And we will have problem in generating TrajData! 
        #        # So we test it here: the data folder must have `.dat` or `.traj` file.
        #        # If not, we will skip thi
        #        if glob.glob(os.path.join(dir_path, '*.dat')) or glob.glob(os.path.join(dir_path, '*.traj')):
        #            if prefix is not None:
        #                if dir_name[:len(prefix)] == prefix:
        #                    include_folders.append(dir_name)
        #            else:
        #                include_folders.append(dir_name)

        assert prefix is not None, "The prefix is not provided. Please provide the prefix to select the trajectory folders."
        prefix_folders = glob.glob(f"{root}/{prefix}{separator}*")
        include_folders=[]
        for idir in prefix_folders:
            if os.path.isdir(idir):
                if not glob.glob(os.path.join(idir, '*.dat')) and not glob.glob(os.path.join(idir, '*.traj')) and not glob.glob(os.path.join(idir, '*.h5')):
                    raise Exception(f"{idir} does not have the proper traj data files. Please check the data files.")
                include_folders.append(idir.split('/')[-1])
        
        assert isinstance(include_folders, list) and len(include_folders) > 0, "No trajectory folders are found. Please check the prefix."                
            
        # We need to check the `info.json` very carefully here.
        # Different `info` points to different dataset, 
        # even if the data files in `root` are basically the same.
        info_files = {}

        # See if a public info is provided.
        #if "info.json" in os.listdir(root):
        if os.path.exists(f"{root}/info.json"):
            public_info = j_loader(os.path.join(root, "info.json"))
            public_info = normalize_setinfo(public_info)
            print("A public `info.json` file is provided, and will be used by the subfolders who do not have their own `info.json` file.")
        else:
            public_info = None

        # Load info in each trajectory folders seperately.
        for file in include_folders:
            #if "info.json" in os.listdir(os.path.join(root, file)):
            if os.path.exists(f"{root}/{file}/info.json"):
                # use info provided in this trajectory.
                info = j_loader(f"{root}/{file}/info.json")
                info = normalize_setinfo(info)
                info_files[file] = info
            elif public_info is not None:
                # use public info instead
                # yaml will not dump correctly if this is not a deepcopy.
                info_files[file] = deepcopy(public_info)
            else:
                # no info for this file
                raise Exception(f"info.json is not properly provided for `{file}`.")
            
        # We will sort the info_files here.
        # The order itself is not important, but must be consistant for the same list.
        info_files = {key: info_files[key] for key in sorted(info_files)}
        
        if dataset_type == "DeePHDataset":
            dataset = DeePHE3Dataset(
                root=root,
                type_mapper=idp,
                get_Hamiltonian=get_Hamiltonian,
                get_eigenvalues=get_eigenvalues,
                info_files = info_files
            )
        elif dataset_type == "DefaultDataset":
            dataset = DefaultDataset(
                root=root,
                type_mapper=idp,
                get_Hamiltonian=get_Hamiltonian,
                get_overlap=get_overlap,
                get_DM=get_DM,
                get_eigenvalues=get_eigenvalues,
                info_files = info_files
            )
        else:
            dataset = HDF5Dataset(
                root=root,
                type_mapper=idp,
                get_Hamiltonian=get_Hamiltonian,
                get_overlap=get_overlap,
                get_DM=get_DM,
                get_eigenvalues=get_eigenvalues,
                info_files = info_files
            )

    elif dataset_type == "LMDBDataset":
        assert prefix is not None, "The prefix is not provided. Please provide the prefix to select the trajectory folders."
        prefix_folders = glob.glob(f"{root}/{prefix}*.lmdb")
        include_folders=[]
        for idir in prefix_folders:
            if os.path.isdir(idir):
                if not glob.glob(os.path.join(idir, '*.mdb')):
                    raise Exception(f"{idir} does not have the proper traj data files. Please check the data files.")
                include_folders.append(idir.split('/')[-1])
        
        assert isinstance(include_folders, list) and len(include_folders) == 1, "No trajectory folders are found. Please check the prefix."                

        # See if a public info is provided.
        #if "info.json" in os.listdir(root):
        
        if os.path.exists(f"{root}/info.json"):
            info = j_loader(f"{root}/info.json")
        else:
            print("Please provide a info.json file.")
            raise Exception("info.json is not properly provided for this dataset.")
        
        # We will sort the info_files here.
        # The order itself is not important, but must be consistant for the same list.
        
        dataset = LMDBDataset(
            root=os.path.join(root, include_folders[0]),
            type_mapper=idp,
            info=info,
            orthogonal=orthogonal,
            get_Hamiltonian=get_Hamiltonian,
            get_overlap=get_overlap,
            get_DM=get_DM,
            get_eigenvalues=get_eigenvalues,
        )

    else:
        raise ValueError(f"Not support dataset type: {type}.")

    return dataset