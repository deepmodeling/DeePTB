from .from_deephe3.deephe3 import Net
import torch
import torch.nn as nn
import e3nn.o3 as o3
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicData, AtomicDataDict
from dptb.data.AtomicDataDict import with_edge_vectors, with_env_vectors, with_batch
from dptb.nn.embedding.emb import Embedding
from typing import Dict, Union, List, Tuple, Optional, Any


@Embedding.register("deeph-e3")
class E3DeePH(nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            n_atom: int=1, 
            irreps_embed: o3.Irreps=o3.Irreps("64e"), 
            lmax: int=3,
            irreps_mid: o3.Irreps=o3.Irreps("64x0e+32x1o+16x2e+8x3o+8x4e+4x5o"),
            n_layer: int=3, 
            rc: float=5.0, 
            n_basis: int=128,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
            ):
        
        super(E3DeePH, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device

        irreps_mid = o3.Irreps(irreps_mid)
        
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb")
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
            
        self.basis = self.idp.basis

        self.idp.get_irreps(no_parity=False)
        irreps_sh=o3.Irreps([(1, (i, (-1) ** i)) for i in range(lmax + 1)])
        # if not no_parity:
        #     irreps_sh=o3.Irreps([(1, (i, (-1) ** i)) for i in range(lmax + 1)])
        # else:
        #     irreps_sh=o3.Irreps([(1, (i, 1)) for i in range(lmax + 1)])

        self.net = Net(
                num_species=n_atom,
                irreps_embed_node=irreps_embed, 
                irreps_sh=irreps_sh,
                irreps_mid_node=irreps_mid, 
                irreps_post_node=self.idp.orbpair_irreps.sort()[0].simplify(), # it can be derived from the basis
                irreps_out_node=self.idp.orbpair_irreps, # it can be dervied from the basis
                irreps_edge_init=irreps_embed,
                irreps_mid_edge=irreps_mid, 
                irreps_post_edge=self.idp.orbpair_irreps.sort()[0].simplify(), # it can be dervied from the basis
                irreps_out_edge=self.idp.orbpair_irreps, # it can be dervied from the basis
                num_block=n_layer,
                r_max=rc, 
                use_sc=False,
                no_parity=False, 
                use_sbf=False,
                selftp=False, 
                edge_upd=True,
                only_ij=False,
                num_basis=n_basis
            )

        
        self.net.to(self.device)

        self.out_irreps = self.idp.orbpair_irreps
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)

        node_feature, edge_feature = self.net(data)
        data[AtomicDataDict.NODE_FEATURES_KEY] = node_feature
        data[AtomicDataDict.EDGE_FEATURES_KEY] = edge_feature

        return data

    @property
    def out_edge_irreps(self):
        return self.out_irreps

    @property
    def out_node_irreps(self):
        return self.out_irreps
