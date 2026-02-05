import op2c
import torch
from typing import Union

from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict    
from dptb.data import _keys
from dptb.data.interfaces import block_to_feature
from dptb.data.AtomicDataDict import with_edge_vectors

def compute_overlap(data: AtomicDataDict, idp: OrbitalMapper, orb_dir, orb_names):
    ntype = idp.num_types
    op = op2c.Op2C(
        ntype=ntype, 
        nspin=1, # for current usage
        lspinorb=False,
        orb_dir=orb_dir, 
        orb_name=orb_names,
        log_file="op2c.log"
    )

    ovp_dict = {}

    if _keys.EDGE_VECTORS_KEY not in data:
        data = with_edge_vectors(data)

    edge_index = data[_keys.EDGE_INDEX_KEY].cpu()
    atom_types = data[_keys.ATOM_TYPE_KEY].cpu()
    edge_vectors = data[_keys.EDGE_VECTORS_KEY].cpu()
    cell_shifts = data[_keys.EDGE_CELL_SHIFT_KEY].cpu()

    for k in range(edge_index.shape[1]):
        i, j = edge_index[:, k].tolist()
        itype, jtype = atom_types[i].item(), atom_types[j].item()

        Rij = edge_vectors[k] * 1.8897259886 # angstrom to bohr
        Rij = Rij.tolist()
        Rvec = cell_shifts[k].int().tolist()

        inorb = idp.atom_norb[itype]
        jnorb = idp.atom_norb[jtype]

        S = op.overlap(itype, jtype, Rij, is_transpose=False)
        ovp_dict["{}_{}_{}_{}_{}".format(i, j, *Rvec)] = S.reshape(inorb, jnorb)
    
    for k in range(atom_types.shape[0]):
        itype = atom_types[k].item()
        S = op.overlap(itype, itype, [0, 0, 0], is_transpose=False)
        inorb = idp.atom_norb[itype]
        ovp_dict["{}_{}_{}_{}_{}".format(k, k, 0, 0, 0)] = S.reshape(inorb, inorb)

    block_to_feature(data, idp, False, ovp_dict, False)

    return data
    
    