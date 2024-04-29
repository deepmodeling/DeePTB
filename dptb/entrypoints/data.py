import os
from typing import Dict, List, Optional, Any
from dptb.utils.tools import j_loader
import numpy as np
import glob
import shutil
from tqdm import tqdm
from dptb.utils.argcheck import normalize
from dptb.data.interfaces.abacus import recursive_parse
from dptb.utils.tools import setup_seed

def data(
        INPUT: str,
        parse: bool=False,
        split: bool=False,
        collect: bool=False,
        **kwargs
):
    jdata = j_loader(INPUT)


    if parse:
        if jdata["type"] == "ABACUS":

            # ABACUS parsing input like:
            # {  "type": "ABACUS",
            #    "parse_arguments": {
            #        "input_path": "alice_*/*_bob/system_No_*",
            #        "preprocess_dir": "charlie/david",
            #        "only_overlap": false, 
            #        "get_Hamiltonian": true, 
            #        "add_overlap": true, 
            #        "get_eigenvalues": true } }

            abacus_args = jdata["parse_arguments"]
            assert abacus_args.get("input_path") is not None, "ABACUS calculation results MUST be provided."
            assert abacus_args.get("preprocess_dir") is not None, "Please assign a dictionary to store preprocess files."

            print("Begin parsing ABACUS output...")
            recursive_parse(**abacus_args)
            print("Finished parsing ABACUS output.")
            
            ## write all h5 files to be used in building AtomicData
            #with open(os.path.join(abacus_args["preprocess_dir"], "AtomicData_file.txt"), "w") as f:
            #    for filename in h5_filenames:
            #        f.write(filename + "\n")

        else:
            raise Exception("Not supported software output.")
        
    if split:
        # Split the data into training and testing sets
        # {
        #    "dataset_dir": "",
        #    "prefix": "",
        #    "train_ratio": 0.6 
        #    "test_ratio": 0.2,
        #    "val_ratio": 0.2 
        # }

        # setup seed
        rds = np.random.RandomState(1)


        dataset_dir = jdata.get("dataset_dir")
        filenames = list(glob.glob(os.path.join(dataset_dir, jdata.get("prefix")+"*")))
        nfile = len(filenames)
        assert nfile > 0, "No file found in the dataset directory."

        train_ratio = jdata.get("train_ratio")
        test_ratio = jdata.get("test_ratio", 0.)
        val_ratio = jdata.get("val_ratio")

        assert train_ratio + test_ratio + val_ratio == 1.0, "The sum of train_ratio, test_ratio, and val_ratio must be 1.0."
        assert train_ratio > 0 and test_ratio >= 0 and val_ratio > 0, "All ratios must be positive."

        n_train = int(nfile * train_ratio)
        n_test = int(nfile * test_ratio)
        n_val = nfile - n_train - n_test

        indices = rds.choice(nfile, nfile, replace=False)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        os.mkdir(os.path.join(dataset_dir, "train"))
        os.mkdir(os.path.join(dataset_dir, "val"))

        for id in tqdm(train_indices, desc="Copying files to training sets..."):
            os.system(f"cp -r {filenames[id]} {dataset_dir}/train")
        
        for id in tqdm(val_indices, desc="Copying files to validation sets..."):
            os.system(f"cp -r {filenames[id]} {dataset_dir}/val")

        if n_test > 0:
            os.mkdir(os.path.join(dataset_dir, "test"))
            for id in tqdm(test_indices, desc="Copying files to testing sets..."):
                os.system(f"cp -r {filenames[id]} {dataset_dir}/test")

    if collect:
        # Collect sub-folders produced by parse into one dataset.
        # {
        #    "subfolders": "alice_*/*_bob/set.*/frame.*",  can be a list too.
        #    "name": "prefix_for_merged_dataset",
        #    "output_dir": "path_for_collected_subsets"
        # }
        # "subfolders" should always point to folders containing the `.dat` files.
        # IMPORTANT: collecting the `.traj` dataset folders are not supported yet.

        name = jdata.get("name")
        output_dir = jdata.get("output_dir")
        input_path = jdata.get("subfolders")

        if isinstance(input_path, list) and all(isinstance(item, str) for item in input_path):
            input_path = input_path
        else:
            input_path = glob.glob(input_path)

        subfolders = [item for item in input_path if os.path.isdir(item)]

        assert len(subfolders) > 0, "No sub-folders found in the provided path."

        os.mkdir(os.path.join(output_dir, name))  
        output_dir = os.path.join(output_dir, name)

        for idx, subfolder in enumerate(tqdm(subfolders, desc="Collecting files...")):
            # Check necessary data files.
            required_files = ['positions.dat', 'cell.dat', 'atomic_numbers.dat']
            files_exist = all(os.path.isfile(os.path.join(subfolder, f)) for f in required_files)

            if files_exist:
                new_folder_name = f"{name}.{idx}"
                new_folder_path = os.path.join(output_dir, new_folder_name)
                shutil.copytree(subfolder, new_folder_path)
            else:
                print(f"Warning: data missing in {subfolder}. Skipping.")
        
        print("Subfolders collected.")