import ase
import glob
import numpy as np
from ase.io.trajectory import Trajectory
import torch
from dptb.structure.structure import BaseStruct
from dptb.dataprocess.processor import Processor

def read_data(path, prefix, cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode:str='uniform', time_symm=True, **kwargs):
    """根据文件路径和prefix的读取文件夹下的数据文件,并存储为神经网络模型的输入格式数据
    """
    filenames  = {
        "xdat_file": "xdat.traj",
        "eigen_file": "eigs.npy",
        "kpoints_file" : "kpoints.npy"
    }
    filenames.update(kwargs)

    data_dirs = glob.glob(path + "/" + prefix + ".*")

    print(path + "/" + prefix + ".*")

    struct_list_sets = []
    kpoints_sets = []
    eigens_sets = []
    for ii in range(len(data_dirs)):
        struct_list = []
        asetrajs = Trajectory(filename=data_dirs[ii] + "/" + filenames['xdat_file'], mode='r')
        kpoints = np.load(data_dirs[ii] + "/" + filenames['kpoints_file'])
        eigs = np.load(data_dirs[ii] + "/" + filenames['eigen_file'])
        if len(eigs.shape)==2:
            eigs = eigs[np.newaxis]
        assert len(eigs.shape) == 3
        kpoints_sets.append(kpoints)
        eigens_sets.append(eigs)
        
        for iatom in asetrajs:

            struct = BaseStruct(atom=iatom, format='ase', cutoff=cutoff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles, onsitemode=onsitemode, time_symm=time_symm)
            struct_list.append(struct)
        struct_list_sets.append(struct_list)


    return struct_list_sets, kpoints_sets, eigens_sets


def get_data(path, prefix,batch_size, bond_cutoff, env_cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode:str='uniform', time_symm=True, device='cpu', dtype=torch.float32, **kwargs):
    """
        input: data params
        output: processor
    """
    
    struct_list_sets, kpoints_sets, eigens_sets = read_data(path, prefix, bond_cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode, time_symm, **kwargs)
    assert len(struct_list_sets) == len(kpoints_sets) == len(eigens_sets)
    processor_list = []

    for i in range(len(struct_list_sets)):
        processor_list.append(
            Processor(structure_list=struct_list_sets[i], batchsize=batch_size,
                        kpoint=kpoints_sets[i], eigen_list=eigens_sets[i], device=device, 
                        dtype=dtype, env_cutoff=onsite_cutoff, sorted_bond="st", sorted_env="st"))
    






