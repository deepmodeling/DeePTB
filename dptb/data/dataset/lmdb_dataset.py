import numpy as np
import logging
import inspect
import itertools
import yaml
import hashlib
import math
from typing import Tuple, Dict, Any, List, Callable, Union, Optional

import torch

from torch_runstats.scatter import scatter_std, scatter_mean

from dptb.utils.torch_geometric import Batch, Dataset
from dptb.utils.tools import download_url, extract_zip

import dptb
from dptb.data import (
    AtomicData,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
)
from dptb.utils.batch_ops import bincount
from dptb.utils.regressor import solver
from dptb.utils.savenload import atomic_write
from ..transforms import TypeMapper
from ._base_datasets import AtomicDataset

class AtomicLMDBDataset(AtomicDataset):
    r"""Base class for all datasets that fit in memory.

    Please note that, as a ``pytorch_geometric`` dataset, it must be backed by some kind of disk storage.
    By default, the raw file will be stored at root/raw and the processed torch
    file will be at root/process.

    Subclasses must implement:
     - ``raw_file_names``
     - ``get_data()``

    Subclasses may implement:
     - ``download()`` or ``self.url`` or ``ClassName.URL``

    Args:
        root (str, optional): Root directory where the dataset should be saved. Defaults to current working directory.
        file_name (str, optional): file name of data source. only used in children class
        url (str, optional): url to download data source
        AtomicData_options (dict, optional): extra key that are not stored in data but needed for AtomicData initialization
        include_frames (list, optional): the frames to process with the constructor.
        type_mapper (TypeMapper): the transformation to map atomic information to species index. Optional
    """

    def __init__(
        self,
        root: str,
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        AtomicData_options: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
        type_mapper: Optional[TypeMapper] = None,
    ):
        # TO DO, this may be simplified
        # See if a subclass defines some inputs
        self.file_name = (
            getattr(type(self), "FILE_NAME", None) if file_name is None else file_name
        )
        self.url = getattr(type(self), "URL", url)

        self.AtomicData_options = AtomicData_options
        self.include_frames = include_frames

        self.data = None

        # !!! don't delete this block.
        # otherwise the inherent children class
        # will ignore the download function here
        class_type = type(self)
        if class_type != AtomicLMDBDataset:
            if "download" not in self.__class__.__dict__:
                class_type.download = AtomicLMDBDataset.download
            if "process" not in self.__class__.__dict__:
                class_type.process = AtomicLMDBDataset.process

        # Initialize the InMemoryDataset, which runs download and process
        # See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets
        # Then pre-process the data if disk files are not found
        super().__init__(root=root, type_mapper=type_mapper)
        if self.data is None:
            self.data, include_frames = torch.load(self.processed_paths[0])
            if not np.all(include_frames == self.include_frames):
                raise ValueError(
                    f"the include_frames is changed. "
                    f"please delete the processed folder and rerun {self.processed_paths[0]}"
                )

    def len(self):
        if self.data is None:
            return 0
        return self.data.num_graphs

    @property
    def raw_file_names(self):
        raise NotImplementedError()

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pth", "params.yaml"]

    def get_data(
        self,
    ) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], List[AtomicData]]:
        """Get the data --- called from ``process()``, can assume that ``raw_file_names()`` exist.

        Note that parameters for graph construction such as ``pbc`` and ``r_max`` should be included here as (likely, but not necessarily, fixed) fields.

        Returns:
        A dict:
            fields: dict
                mapping a field name ('pos', 'cell') to a list-like sequence of tensor-like objects giving that field's value for each example.
        Or:
            data_list: List[AtomicData]
        """
        raise NotImplementedError

    def download(self):
        if (not hasattr(self, "url")) or (self.url is None):
            # Don't download, assume present. Later could have FileNotFound if the files don't actually exist
            pass
        else:
            download_path = download_url(self.url, self.raw_dir)
            if download_path.endswith(".zip"):
                extract_zip(download_path, self.raw_dir)

    def process(self):
        data = self.get_data() ## get data returns either a list of AtomicData class or a data dict
        if isinstance(data, list):

            # It's a data list
            data_list = data
            if not (self.include_frames is None or data_list is None):
                data_list = [data_list[i] for i in self.include_frames] # 可以选择数据集中加载的序号
            assert all(isinstance(e, AtomicData) for e in data_list)
            assert all(AtomicDataDict.BATCH_KEY not in e for e in data_list)

            fields = {}

        elif isinstance(data, dict):
            # It's fields
            # Get our data
            fields = data

            # check keys
            all_keys = set(fields.keys())
            assert AtomicDataDict.BATCH_KEY not in all_keys
            # Check bad key combinations, but don't require that this be a graph yet.
            AtomicDataDict.validate_keys(all_keys, graph_required=False)

            # check dimesionality
            num_examples = set([len(a) for a in fields.values()])
            if not len(num_examples) == 1:
                raise ValueError(
                    f"This dataset is invalid: expected all fields to have same length (same number of examples), but they had shapes { {f: v.shape for f, v in fields.items() } }"
                )
            num_examples = next(iter(num_examples))

            include_frames = self.include_frames
            if include_frames is None:
                include_frames = range(num_examples)

            # Make AtomicData from it:
            if AtomicDataDict.EDGE_INDEX_KEY in all_keys:
                # This is already a graph, just build it
                constructor = AtomicData
            else:
                # do neighborlist from points
                constructor = AtomicData.from_points
                assert "r_max" in self.AtomicData_options
                assert AtomicDataDict.POSITIONS_KEY in all_keys

            data_list = [
                constructor(
                    **{
                        **{f: v[i] for f, v in fields.items()},
                        **self.AtomicData_options,
                    }
                )
                for i in include_frames
            ]


        else:
            raise ValueError("Invalid return from `self.get_data()`")

        # Batch it for efficient saving
        # This limits an AtomicInMemoryDataset to a maximum of LONG_MAX atoms _overall_, but that is a very big number and any dataset that large is probably not "InMemory" anyway
        data = Batch.from_data_list(data_list)
        del data_list
        del fields

        total_MBs = sum(item.numel() * item.element_size() for _, item in data) / (
            1024 * 1024
        )
        logging.info(
            f"Loaded data: {data}\n    processed data size: ~{total_MBs:.2f} MB"
        )
        del total_MBs

        # use atomic writes to avoid race conditions between
        # different trainings that use the same dataset
        # since those separate trainings should all produce the same results,
        # it doesn't matter if they overwrite each others cached'
        # datasets. It only matters that they don't simultaneously try
        # to write the _same_ file, corrupting it.
        with atomic_write(self.processed_paths[0], binary=True) as f:
            torch.save((data, self.include_frames), f)
        with atomic_write(self.processed_paths[1], binary=False) as f:
            yaml.dump(self._get_parameters(), f)

        logging.info("Cached processed data to disk")

        self.data = data

    def get(self, idx):
        return self.data.get_example(idx)