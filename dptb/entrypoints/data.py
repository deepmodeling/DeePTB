import os
from typing import Dict, List, Optional, Any
from dptb.utils.tools import j_loader
import numpy as np
import glob
from tqdm import tqdm
from dptb.utils.argcheck import normalize
from dptb.data.interfaces.abacus import recursive_parse
from dptb.utils.tools import setup_seed

def data(
        INPUT: str,
        parse: bool=False,
        split: bool=False,
        **kwargs
):
    jdata = j_loader(INPUT)

    # ABACUS parsing input like:
    # {  "type": "ABACUS",
    #    "parse_arguments": {
    #        "input_path": "alice_*/*_bob/system_No_*",
    #        "preprocess_dir": "charlie/david",
    #        "only_overlap": false, 
    #        "get_Hamiltonian": true, 
    #        "add_overlap": true, 
    #        "get_eigenvalues": true } }

    if parse:
        if jdata["type"] == "ABACUS":
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
        setup_seed(seed=jdata.get("seed", 1021312))

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
        n_val = int(nfile * val_ratio)

        indices = np.random.choice(nfile, nfile, replace=False)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]

        os.mkdir(os.path.join(dataset_dir, "train"))
        os.mkdir(os.path.join(dataset_dir, "val"))

        for id in tqdm(train_indices, desc="Copying files to training sets..."):
            print("cp " + filenames[id] + " " + os.path.join(dataset_dir, "train")+" -r")
            os.system("cp " + filenames[id] + " " + os.path.join(dataset_dir, "train")+" -r")
        
        for id in tqdm(val_indices, desc="Copying files to validation sets..."):
            print("cp " + filenames[id] + " " + os.path.join(dataset_dir, "val")+" -r")
            os.system("cp " + filenames[id] + " " + os.path.join(dataset_dir, "val")+" -r")

        if n_test > 0:
            os.mkdir(os.path.join(dataset_dir, "test"))
            for id in tqdm(test_indices, desc="Copying files to testing sets..."):
                print("cp " + filenames[id] + " " + os.path.join(dataset_dir, "test")+" -r")
                os.system("cp " + filenames[id] + " " + os.path.join(dataset_dir, "test")+" -r")