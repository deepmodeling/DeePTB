import torch
from typing import Union
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict    
from dptb.data import _keys
from dptb.data.interfaces import block_to_feature
from dptb.data.AtomicDataDict import with_edge_vectors
try:
    from vbcsr import ImageContainer
    from vbcsr import AtomicData as AtomicData_vbcsr
except ImportError:
    print("VBCSR is not installed, therefore, the compute_overlap_image is not supported.")

try:
    import op2c
except ImportError:
    print("OP2C is not installed, therefore, the compute_overlap is not supported.")

def compute_overlap(data: AtomicDataDict, idp: OrbitalMapper, orb_dir, orb_names):
    ntype = idp.num_types
    op = op2c.Op2C(
        ntype=ntype, 
        nspin=1, # for current usage
        lspinorb=False,
        orb_dir=orb_dir, 
        orb_name=orb_names
    )

    device = data[_keys.EDGE_INDEX_KEY].device
    ovp_dict = {}

    if _keys.EDGE_VECTORS_KEY not in data:
        data = with_edge_vectors(data)
    if _keys.ATOM_TYPE_KEY not in data:
        idp(data)

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
    data[_keys.NODE_OVERLAP_KEY].to(device)
    data[_keys.EDGE_OVERLAP_KEY].to(device)

    return data

def compute_overlap_image(data: AtomicDataDict, idp: OrbitalMapper, orb_dir, orb_names):
    ntype = idp.num_types
    op = op2c.Op2C(
        ntype=ntype, 
        nspin=1, # for current usage
        lspinorb=False,
        orb_dir=orb_dir, 
        orb_name=orb_names
    )

    device = data[_keys.EDGE_INDEX_KEY].device
    ovp_dict = {}

    if _keys.EDGE_VECTORS_KEY not in data:
        data = with_edge_vectors(data)
    if _keys.ATOM_TYPE_KEY not in data:
        idp(data)

    edge_index = data[_keys.EDGE_INDEX_KEY].cpu()
    atom_types = data[_keys.ATOM_TYPE_KEY].cpu()
    edge_vectors = data[_keys.EDGE_VECTORS_KEY].cpu()
    cell_shifts = data[_keys.EDGE_CELL_SHIFT_KEY].cpu()

    natom = len(atom_types)
    nedge = len(edge_index[0])
    adata = AtomicData_vbcsr.from_distributed(
        natom, natom, 0, nedge, nedge,
        list(range(natom)), atom_types, edge_index.T, idp.atom_norb, cell_shifts,
        data[_keys.CELL_KEY], data[_keys.POSITIONS_KEY]
    )
    image_container = ImageContainer(adata, np.float64)

    for k in range(nedge):
        i, j = edge_index[:, k].tolist()
        itype, jtype = atom_types[i].item(), atom_types[j].item()

        Rij = edge_vectors[k] * 1.8897259886 # angstrom to bohr
        Rij = Rij.tolist()
        Rvec = cell_shifts[k].int().tolist()

        inorb = idp.atom_norb[itype]
        jnorb = idp.atom_norb[jtype]

        S = op.overlap(itype, jtype, Rij, is_transpose=False)
        image_container.add_block(i, j, Rvec, S.reshape(inorb, jnorb))
    
    for k in range(natom):
        itype = atom_types[k].item()
        S = op.overlap(itype, itype, [0, 0, 0], is_transpose=False)
        inorb = idp.atom_norb[itype]
        image_container.add_block(k, k, [0, 0, 0], S.reshape(inorb, inorb))

    image_container.assemble()

    return image_container
