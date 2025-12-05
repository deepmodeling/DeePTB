# in-built packages
import os
import glob
import logging
import inspect
from typing import Union
from copy import deepcopy
from importlib import import_module
from pathlib import Path

# third-party packages
import torch

# local packages
from dptb import data
from dptb.data import AtomicDataset, register_fields
from dptb.data.dataset import DefaultDataset
from dptb.data.dataset._deeph_dataset import DeePHE3Dataset
from dptb.data.dataset._hdf5_dataset import HDF5Dataset
from dptb.data.dataset.lmdb_dataset import LMDBDataset
from dptb.data.transforms import TypeMapper, OrbitalMapper
from dptb.utils import instantiate, get_w_prefix
from dptb.utils.tools import j_loader
from dptb.utils.argcheck import normalize_setinfo, normalize_lmdbsetinfo
from dptb.utils.argcheck import collect_cutoffs 
from dptb.utils.argcheck import get_cutoffs_from_model_options

log = logging.getLogger(__name__)

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

class DatasetBuilder:
    def __init__(self):
        pass

    def __call__(self, 
        # set_options
        root: str,
        # dataset_options
        r_max: Union[float, int, dict],
        er_max: Union[float, int, dict] = None,
        oer_max: Union[float, int, dict] = None,
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
        # set cutoff radius
        self.r_max   = r_max   # default cutoff
        self.er_max  = er_max  # env cutoff
        self.oer_max = oer_max 

        # seems this warning will always show?
        self.if_check_cutoffs = False
        log.warning("The cutoffs in data and model are not checked. be careful!")

        # set and check the dataset type
        dataset_type = type
        assert dataset_type in ["DefaultDataset", "DeePHDataset", "HDF5Dataset", "LMDBDataset"], \
            f"The dataset type {dataset_type} is not supported. Please check the type."

        # See if we can get a OrbitalMapper.
        idp = None if basis is None else OrbitalMapper(basis=basis)

        # filter to get the valid folders
        assert prefix is not None, \
            "The prefix is not provided. Please provide the prefix to select the trajectory folders."
        valid_folders, delim = [], separator
        for folder in Path(root).glob(f"{prefix}{delim}*"):
            if folder.is_dir():
                assert any(folder.glob(f'*.{ext}') for ext in ['dat', 'traj', 'h5', 'mdb']), \
                    f'{folder} does not have the proper traj data files. Please check the data files.'
                valid_folders.append(folder.name)
        assert isinstance(valid_folders, list) and len(valid_folders) > 0, \
            "No trajectory folders are found. Please check the prefix."

        # We need to check the `info.json` very carefully here.
        # Different `info` points to different dataset, 
        # even if the data files in `root` are basically the same.
        info_files, public_info = {}, None
        # the info_files is a dict that maps the folder name to the info in dict format.

        # setting on the public (default) info that will be the default for subfolders without its
        # own info.json file.
        finfo = Path(root) / "info.json"
        if finfo.exists():
            public_info = j_loader(finfo)
            if dataset_type == "LMDBDataset":
                public_info = {}
                log.info("A public `info.json` file is provided, "
                         "but will not be used anymore for LMDBDataset.")
            else:
                public_info = normalize_setinfo(public_info)
                log.info("A public `info.json` file is provided, "
                         "and will be used by the subfolders who do not have their own `info.json` file.")

        # Load info in each trajectory folders seperately...
        for folder in valid_folders:
            # for LMDBDataset, the information is not set here
            if dataset_type == "LMDBDataset":
                info_files[folder] = {}
                continue
            
            # otherwise, at least the public info would be the default
            finfo = Path(root) / folder / "info.json"
            if not finfo.exists() and public_info is None: # no info in subfolder and no public info. then raise error.
                log.error(f"for {dataset_type} type, the info.json is not properly provided for `{folder}`")
                raise ValueError(f"for {dataset_type} type, the info.json is not properly provided for `{folder}`")
            
            # the case that info is individually provided
            if finfo.exists():
                info_files[folder] = normalize_setinfo(j_loader(finfo))
            else:
                # use public info instead
                assert public_info is not None
                # yaml will not dump correctly if this is not a deepcopy.
                info_files[folder] = deepcopy(public_info)


        # finally, we sort the info_files here.
        # The order itself is not important, but must be consistant for the same list.
        info_files = {key: info_files[key] for key in sorted(info_files)}
    
        # add the cutoff radius information for each valid folder (contains all the data files)
        for folder in info_files:
            info_files[folder].update({'r_max'  : r_max, 
                                       'er_max' : er_max, 
                                       'oer_max': oer_max})

        # after preprocessing, we can build the dataset
        if dataset_type == "DeePHDataset":
            return DeePHE3Dataset(root=root,
                                  type_mapper=idp,
                                  get_Hamiltonian=get_Hamiltonian,
                                  get_eigenvalues=get_eigenvalues,
                                  info_files=info_files)
        elif dataset_type == "DefaultDataset":
            return DefaultDataset(root=root,
                                  type_mapper=idp,
                                  get_Hamiltonian=get_Hamiltonian,
                                  get_overlap=get_overlap,
                                  get_DM=get_DM,
                                  get_eigenvalues=get_eigenvalues,
                                  info_files=info_files)
        elif dataset_type == "HDF5Dataset":
            return HDF5Dataset(root=root,
                               type_mapper=idp,
                               get_Hamiltonian=get_Hamiltonian,
                               get_overlap=get_overlap,
                               get_DM=get_DM,
                               get_eigenvalues=get_eigenvalues,
                               info_files=info_files)
        else:
            assert dataset_type == "LMDBDataset"
            return LMDBDataset(root=root,
                               type_mapper=idp,
                               orthogonal=orthogonal,
                               get_Hamiltonian=get_Hamiltonian,
                               get_overlap=get_overlap,
                               get_DM=get_DM,
                               get_eigenvalues=get_eigenvalues,
                               info_files=info_files)
    
    def from_model(self, 
               model, 
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
               **kwargs):
        """
        Build a dataset from a model.

        Args:
            - model (torch.nn.Module): The model to build the dataset from.
            - dataset_type (str, optional): The type of dataset to build. Default is "DefaultDataset".

        Returns:
            dataset: The built dataset.
        """
        # cutoff_options = collect_cutoffs(model.model_options)
        r_max, er_max, oer_max  = get_cutoffs_from_model_options(model.model_options)
        cutoff_options = {'r_max': r_max, 'er_max': er_max, 'oer_max': oer_max}
        
        dataset = self(
            root = root,
            **cutoff_options,
            type = type,
            prefix = prefix,
            separator = separator,
            get_Hamiltonian = get_Hamiltonian,
            get_overlap = get_overlap,
            get_DM = get_DM,
            get_eigenvalues = get_eigenvalues,
            orthogonal = orthogonal,
            basis = basis, 
            **kwargs,
        )

        return dataset

    def check_cutoffs(self,model:torch.nn.Module=None, **kwargs):
        if model is None:
            self.if_check_cutoffs = False
            log.warning("No model is provided. We can not check the cutoffs used in data and model are consistent.")
        else:
            self.if_check_cutoffs = True
            cutoff_options = collect_cutoffs(model.model_options)
            if isinstance(cutoff_options['r_max'],dict):
                assert isinstance(self.r_max,dict), "The r_max in model is a dict, but in dataset it is not."
                for key in cutoff_options['r_max']:
                    if key not in self.r_max:
                        log.error(f"The key {key} in r_max is not defined in dataset")
                        raise ValueError(f"The key {key} in r_max is not defined in dataset")
                    assert self.r_max >=  cutoff_options['r_max'][key], f"The r_max in model shoule be  smaller than in dataset for {key}."

            elif isinstance(cutoff_options['r_max'],float):
                assert isinstance(self.r_max,float), "The r_max in model is a float, but in dataset it is not."
                assert self.r_max >=  cutoff_options['r_max'], "The r_max in model shoule be  smaller than in dataset."        

            if isinstance(cutoff_options['er_max'],dict):
                assert isinstance(self.er_max,dict), "The er_max in model is a dict, but in dataset it is not."
                for key in cutoff_options['er_max']:
                    if key not in self.er_max:
                        log.error(f"The key {key} in er_max is not defined in dataset")
                        raise ValueError(f"The key {key} in er_max is not defined in dataset")
                    
                    assert self.er_max >=  cutoff_options['er_max'][key], f"The er_max in model shoule be  smaller than in dataset for {key}."

            elif isinstance(cutoff_options['er_max'],float):
                assert isinstance(self.er_max,float), "The er_max in model is a float, but in dataset it is not."
                assert self.er_max >=  cutoff_options['er_max'], "The er_max in model shoule be  smaller than in dataset."
            elif cutoff_options['er_max'] is None:
                assert self.er_max is None, "The er_max in model is None, but in dataset it is not."

            
            if isinstance(cutoff_options['oer_max'],dict):
                assert isinstance(self.oer_max,dict), "The oer_max in model is a dict, but in dataset it is not."
                for key in cutoff_options['oer_max']:
                    if key not in self.oer_max:
                        log.error(f"The key {key} in oer_max is not defined in dataset")
                        raise ValueError(f"The key {key} in oer_max is not defined in dataset")
                    
                    assert self.oer_max >=  cutoff_options['oer_max'][key], f"The oer_max in model shoule be  smaller than in dataset for {key}."
            elif isinstance(cutoff_options['oer_max'],float):
                assert isinstance(self.oer_max,float), "The oer_max in model is a float, but in dataset it is not."
                assert self.oer_max >=  cutoff_options['oer_max'], "The oer_max in model shoule be  smaller than in dataset."
            elif cutoff_options['oer_max'] is None:
                assert self.oer_max is None, "The oer_max in model is None, but in dataset it is not."
                
# note, compared to the previous build_dataset, this one is more flexible.    
# previous build_dataset is a function. now i define a class DataBuilder and re-defined __call__ function.
# then build_dataset is an instance of DataBuilder class. so i can use build_dataset.from_model() to build dataset from model.
# at the same time the previous way to use  build_dataset is still available. like build_dataset(...).

build_dataset = DatasetBuilder()

