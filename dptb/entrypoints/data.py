import os
from typing import Dict, List, Optional, Any
from dptb.utils.tools import j_loader
from dptb.utils.argcheck import normalize
from dptb.data.interfaces.abacus import recursive_parse

def data(
        INPUT: str,
        log_level: int,
        log_path: Optional[str],
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