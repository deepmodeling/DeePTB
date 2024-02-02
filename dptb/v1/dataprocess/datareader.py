import ase
import glob
import numpy as np
import os
from ase.io.trajectory import Trajectory
import torch
from dptb.structure.structure import BaseStruct
from dptb.dataprocess.processor import Processor
from dptb.dataprocess.process_wannier import get_wannier_blocks
from dptb.utils.tools import j_loader
from dptb.utils.argcheck import normalize_bandinfo

def read_data(path, prefix, cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode:str='uniform', time_symm=True, use_wannier=False, **kwargs):
    """根据文件路径和prefix的读取文件夹下的数据文件,并存储为神经网络模型的输入格式数据
    """
    filenames  = {
        "xdat_file": "xdat.traj",
        "eigen_file": "eigs.npy",
        "kpoints_file" : "kpoints.npy",
        "bandinfo_file": "bandinfo.json",
        "wannier_file": "wannier90_hr.dat"
    }
    
    filenames.update(kwargs)

    data_dirs = glob.glob(path + "/" + prefix + ".*")

    print(path + "/" + prefix + ".*")

    struct_list_sets = []
    kpoints_sets = []
    eigens_sets = []
    bandinfo_sets = []
    wannier_sets = []
    for ii in range(len(data_dirs)):
        struct_list = []
        asetrajs = Trajectory(filename=data_dirs[ii] + "/" + filenames['xdat_file'], mode='r')
        assert len(asetrajs) > 0, "DataPath is not correct!"
        kpoints = np.load(data_dirs[ii] + "/" + filenames['kpoints_file'])
        eigs = np.load(data_dirs[ii] + "/" + filenames['eigen_file'])
        bandinfo = j_loader(data_dirs[ii] + "/" + filenames['bandinfo_file'])

        bandinfo = normalize_bandinfo(bandinfo)
        bandinfo_sets.append(bandinfo)
        if len(eigs.shape)==2:
            eigs = eigs[np.newaxis]
        assert len(eigs.shape) == 3
        kpoints_sets.append(kpoints)
        eigens_sets.append(eigs)

        for iatom in asetrajs:
            struct = BaseStruct(atom=iatom, format='ase', cutoff=cutoff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles, onsitemode=onsitemode, time_symm=time_symm)
            struct_list.append(struct)
        struct_list_sets.append(struct_list)

        if use_wannier:
            assert os.path.exists(data_dirs[ii] + "/" + filenames['wannier_file'])
            #wannier = np.load(data_dirs[ii] + "/" + filenames['wannier_file'], allow_pickle=True)
            assert len(struct_list) == 1, "wannier90_hr.dat should be calculated for one structure only!"
            wannier_proj = bandinfo['wannier_proj']
            orb_wan = bandinfo.get('orb_wan', None)
            wannier = get_wannier_blocks(file=data_dirs[ii] + "/" + filenames['wannier_file'],
                               struct=struct_list[0], wannier_proj_orbital=wannier_proj,orb_wan=orb_wan)
            wannier = [wannier]
        else:
            wannier = [None]
        
        if wannier[0] is None:
            wannier = [None] * eigs.shape[0]
        wannier_sets.append(wannier)
    
    return struct_list_sets, kpoints_sets, eigens_sets, bandinfo_sets, wannier_sets


def get_data(path, prefix, batch_size, bond_cutoff, env_cutoff, onsite_cutoff, proj_atom_anglr_m, proj_atom_neles, 
        sorted_onsite="st", sorted_bond="st", sorted_env="st", onsitemode:str='uniform', time_symm=True, device='cpu', dtype=torch.float32, if_shuffle=True, **kwargs):
    """
        input: data params
        output: processor
    """
    
    struct_list_sets, kpoints_sets, eigens_sets, bandinfo_sets, wannier_sets = read_data(path, prefix, bond_cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode, time_symm, **kwargs)
    assert len(struct_list_sets) == len(kpoints_sets) == len(eigens_sets) == len(bandinfo_sets) == len(wannier_sets)
    processor_list = []

    for i in range(len(struct_list_sets)):
        processor_list.append(
            Processor(structure_list=struct_list_sets[i], batchsize=batch_size,
                        kpoint=kpoints_sets[i], eigen_list=eigens_sets[i], wannier_list=wannier_sets[i], device=device, 
                        dtype=dtype, env_cutoff=env_cutoff, onsite_cutoff=onsite_cutoff, onsitemode=onsitemode, 
                        sorted_onsite=sorted_onsite, sorted_bond=sorted_bond, sorted_env=sorted_env, if_shuffle = if_shuffle, bandinfo=bandinfo_sets[i]))
    
    return processor_list
    






