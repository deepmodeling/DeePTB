"""
This file refactor the SK and E3 Rotation in dptb/hamiltonian/transform_se3.py], it will take input of AtomicDataDict.Type
perform rotation from irreducible matrix element / sk parameters in EDGE/NODE FEATURE, and output the atomwise/ pairwise hamiltonian
as the new EDGE/NODE FEATURE. The rotation should also be a GNN module and speed uptable by JIT. The HR2HK should also be included here.
The indexmapping should ne passed here.
"""

import torch
from e3nn.o3 import wigner_3j, Irrep, xyz_to_angles, Irrep
from dptb.utils.constants import h_all_types, anglrMId
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import re
from torch_runstats.scatter import scatter
from dptb.nn.tensor_product import wigner_D
from dptb.nn.sktb.socbasic import get_soc_matrix_cubic_basis
from dptb.utils.tools import float2comlex
#TODO: 1. jit acceleration 2. GPU support 3. rotate AB and BA bond together.

# The `E3Hamiltonian` class is a PyTorch module that represents a Hamiltonian for a system with a
# given basis and can perform forward computations on input data.

class E3Hamiltonian(torch.nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            decompose: bool = False,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            rotation: bool = False, # for test only
            **kwargs,
            ) -> None:
        
        super(E3Hamiltonian, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if isinstance(device, str):
            device = torch.device(device)
        self.overlap = overlap
        self.dtype = dtype
        self.device = device
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb")
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
            
        self.basis = self.idp.basis
        self.cgbasis = {}
        self.decompose = decompose
        self.rotation = rotation
        if rotation:
            assert self.decompose, "Rotation is only supported when decompose is True."
        self.edge_field = edge_field
        self.node_field = node_field

        # initialize the CG basis
        self.idp.get_orbpairtype_maps()
        orbpairtypes = self.idp.orbpairtype_maps.keys()
        for orbpair in orbpairtypes:
            self._initialize_CG_basis(orbpair)
            

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        The forward function takes in atomic data and performs computations on the edge and node features
        based on the decompose flag. It performs the following operations:
        decompose = True:
            - the function will read the EDGE and NODE features and take them as hamiltonian blocks, the
            block will be decomposed into reduced matrix element that is irrelevant to the direction.
        decompose = False:
            - the function will read the EDGE and NODE features and take them as reduced matrix element, the
            function will transform the reduced matrix element into hamiltonian blocks with directional dependence.
        
        :param data: The `data` parameter is a dictionary that contains atomic data. It has the following
        keys:
        :type data: AtomicDataDict.Type
        :return: the updated `data` dictionary.
        """

        assert data[self.edge_field].shape[1] == self.idp.reduced_matrix_element
        if not self.overlap:
            assert data[self.node_field].shape[1] == self.idp.reduced_matrix_element

        n_edge = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        n_node = data[AtomicDataDict.NODE_FEATURES_KEY].shape[0]

        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        if not self.decompose:
            # The EDGE_FEATURES_KEY and NODE_FAETURE_KEY are the reduced matrix elements

            # compute hopping blocks
            for opairtype in self.idp.orbpairtype_maps.keys():
                # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                # for better performance
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                n_rme = (2*l1+1) * (2*l2+1) # number of reduced matrix element
                rme = data[self.edge_field][:, self.idp.orbpairtype_maps[opairtype]]
                rme = rme.reshape(n_edge, -1, n_rme)
                rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)
                
                HR = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                    rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
                
                # rotation
                # angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
                # rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l1+1, 2l1+1)
                # rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l2+1, 2l2+1)
                # HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R).reshape(n_edge, -1) # shape (N, n_pair * n_rme)
                HR = HR.permute(0,3,1,2).reshape(n_edge, -1)
                data[self.edge_field][:, self.idp.orbpairtype_maps[opairtype]] = HR

            # compute onsite blocks
            if not self.overlap:
                for opairtype in self.idp.orbpairtype_maps.keys():
                    # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                    # for better performance
                    l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                    
                    n_rme = (2*l1+1) * (2*l2+1) # number of reduced matrix element
                    rme = data[self.node_field][:, self.idp.orbpairtype_maps[opairtype]]
                    rme = rme.reshape(n_node, -1, n_rme)
                    rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)
                    
                    HR = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                        rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
                    HR = HR.permute(0,3,1,2).reshape(n_node, -1)

                    # the onsite block does not have rotation
                    data[self.node_field][:, self.idp.orbpairtype_maps[opairtype]] = HR
        
        else:
            for opairtype in self.idp.orbpairtype_maps.keys():
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                nL, nR = 2*l1+1, 2*l2+1
                HR = data[self.edge_field][:, self.idp.orbpairtype_maps[opairtype]]
                HR = HR.reshape(n_edge, -1, nL, nR) # shape (N, n_pair, nL, nR)

                # rotation
                if self.rotation:
                    angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
                    rot_mat_L = wigner_D(int(l1), angle[0], angle[1], torch.zeros_like(angle[0]))
                    rot_mat_R = wigner_D(int(l2), angle[0], angle[1], torch.zeros_like(angle[0]))
                    HR = torch.einsum("nml, nqmo, nok -> nlkq", rot_mat_L, HR, rot_mat_R) # shape (N, nL, nR, n_pair)
                else:
                    HR = HR.permute(0,2,3,1) # shape (N, nL, nR, n_pair)
                    
                rme = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                    HR[:,:,:,None,:], dim=(1,2)) # shape (N, n_rme, n_pair)
                rme = rme.transpose(1,2).reshape(n_edge, -1)

                data[self.edge_field][:, self.idp.orbpairtype_maps[opairtype]] = rme

            if not self.overlap:
                for opairtype in self.idp.orbpairtype_maps.keys():
                    # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                    # for better performance
                    l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                    nL, nR = 2*l1+1, 2*l2+1 # number of reduced matrix element
                    HR = data[self.node_field][:, self.idp.orbpairtype_maps[opairtype]]
                    HR = HR.reshape(n_node, -1, nL, nR).permute(0,2,3,1)# shape (N, nL, nR, n_pair)
                    
                    rme = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                        HR[:,:,:,None,:], dim=(1,2)) # shape (N, n_rme, n_pair)
                    rme = rme.transpose(1,2).reshape(n_node, -1)

                    # the onsite block doesnot have rotation
                    data[self.node_field][:, self.idp.orbpairtype_maps[opairtype]] = rme

        return data
            
    def _initialize_CG_basis(self, pairtype: str):
        """
        The function initializes a Clebsch-Gordan basis for a given pair type.
        
        :param pairtype: The parameter "pairtype" is a string that represents a pair of angular momentum
        quantum numbers. It is expected to have a length of 3, where the first and third characters
        represent the angular momentum quantum numbers of two particles, and the second character
        represents the type of interaction between the particles
        :type pairtype: str
        :return: the CG basis, which is a tensor containing the Clebsch-Gordan coefficients for the given
        pairtype.
        """
        self.cgbasis.setdefault(pairtype, None)

        l1, l2 = anglrMId[pairtype[0]], anglrMId[pairtype[2]]

        cg = []
        for l_ird in range(abs(l2-l1), l2+l1+1):
            cg.append(wigner_3j(int(l1), int(l2), int(l_ird), dtype=self.dtype, device=self.device) * (2*l_ird+1)**0.5)
        
        cg = torch.cat(cg, dim=-1)
        self.cgbasis[pairtype] = cg

        return cg
    
    
class SKHamiltonian_old(torch.nn.Module):
    # transform SK parameters to SK hamiltonian with E3 CG basis, strain is included.
    def __init__(
        self, 
        basis: Dict[str, Union[str, list]]=None,
        idp_sk: Union[OrbitalMapper, None]=None,
        dtype: Union[str, torch.dtype] = torch.float32, 
        device: Union[str, torch.device] = torch.device("cpu"),
        edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
        onsite: bool = False,
        strain: bool = False,
        soc: bool = False,
        **kwargs,
        ) -> None:
        super(SKHamiltonian, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if isinstance(device, str):
            device = torch.device(device)
        self.dtype = dtype
        self.device = device
        self.onsite = onsite
        self.soc = soc

        if basis is not None:
            self.idp_sk = OrbitalMapper(basis, method="sktb", device=device)
            if idp_sk is not None:
                assert idp_sk.basis == self.idp_sk.basis, "The basis of idp and basis should be the same."
        else:
            assert idp_sk is not None, "Either basis or idp should be provided."
            self.idp_sk = idp_sk
        # initilize a e3 indexmapping to help putting the orbital wise blocks into atom-pair wise format
        self.idp = OrbitalMapper(self.idp_sk.basis, method="e3tb", device=device)
        self.basis = self.idp.basis
        self.cgbasis = {}
        self.strain = strain
        self.edge_field = edge_field
        self.node_field = node_field

        self.idp_sk.get_orbpair_maps()
        self.idp_sk.get_skonsite_maps()
        self.idp.get_orbpair_maps()

        if self.soc:
            self.idp_sk.get_sksoc_maps()
            self.idp.get_orbpair_soc_maps()

        pairtypes = self.idp_sk.orbpairtype_maps.keys()
        
        for pairtype in pairtypes:
            # self.cgbasis.setdefault(pairtype, None)
            cg = self._initialize_CG_basis(pairtype)
            self.cgbasis[pairtype] = cg

        self.sk2irs = {
            's-s': torch.tensor([[1.]], dtype=self.dtype, device=self.device),
            's-p': torch.tensor([[1.]], dtype=self.dtype, device=self.device),
            's-d': torch.tensor([[1.]], dtype=self.dtype, device=self.device),
            'p-s': torch.tensor([[1.]], dtype=self.dtype, device=self.device),
            'p-p': torch.tensor([
                [3**0.5/3,2/3*3**0.5],[6**0.5/3,-6**0.5/3]
            ], dtype=self.dtype, device=self.device
            ),
            'p-d':torch.tensor([
                [(2/5)**0.5,(6/5)**0.5],[(3/5)**0.5,-2/5**0.5]
            ], dtype=self.dtype, device=self.device
            ),
            'd-s':torch.tensor([[1.]], dtype=self.dtype, device=self.device),
            'd-p':torch.tensor([
                [(2/5)**0.5,(6/5)**0.5],
                [(3/5)**0.5,-2/5**0.5]
            ], dtype=self.dtype, device=self.device
            ),
            'd-d':torch.tensor([
                [5**0.5/5, 2*5**0.5/5, 2*5**0.5/5],
                [2*(1/14)**0.5,2*(1/14)**0.5,-4*(1/14)**0.5],
                [3*(2/35)**0.5,-4*(2/35)**0.5,(2/35)**0.5]
                ], dtype=self.dtype, device=self.device
            )
        }
        if self.soc:
            self.soc_base_matrix = {
                's':get_soc_matrix_cubic_basis(orbital='s', device=self.device, dtype=self.dtype),
                'p':get_soc_matrix_cubic_basis(orbital='p', device=self.device, dtype=self.dtype),
                'd':get_soc_matrix_cubic_basis(orbital='d', device=self.device, dtype=self.dtype)
            }
            self.cdtype =  float2comlex(self.dtype)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # transform sk parameters to irreducible matrix element

        assert data[self.edge_field].shape[1] == self.idp_sk.reduced_matrix_element
        if self.onsite:
            assert data[self.node_field].shape[1] == self.idp_sk.n_onsite_Es
            n_node = data[self.node_field].shape[0]
            
        n_edge = data[self.edge_field].shape[0]
        

        edge_features = data[self.edge_field].clone()
        data[self.edge_field] = torch.zeros((n_edge, self.idp.reduced_matrix_element), dtype=self.dtype, device=self.device)

        # for hopping blocks
        for opairtype in self.idp_sk.orbpairtype_maps.keys():
            l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
            n_skp = min(l1, l2)+1 # number of reduced matrix element
            skparam = edge_features[:, self.idp_sk.orbpairtype_maps[opairtype]].reshape(n_edge, -1, n_skp)
            rme = skparam @ self.sk2irs[opairtype].T # shape (N, n_pair, n_rme)
            rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)
            
            H_z = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
            
            # rotation
            # when get the angle, the xyz vector should be transformed to yzx.
            angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
            # The roataion matrix is SO3 rotation, therefore Irreps(l,1), is used here.
            rot_mat_L = wigner_D(int(l1), angle[0], angle[1], torch.zeros_like(angle[0]))
            rot_mat_R = wigner_D(int(l2), angle[0], angle[1], torch.zeros_like(angle[0]))
            # rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l1+1, 2l1+1)
            # rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l2+1, 2l2+1)
            
            # Here The string to control einsum is important, the order of the index should be the same as the order of the tensor
            # H_z = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R) # shape (N, n_pair, 2l1+1, 2l2+1)
            HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R).reshape(n_edge, -1) # shape (N, n_pair * 2l2+1 * 2l2+1)
            
            if l1 < l2:
                HR = HR * (-1)**(l1+l2)
            
            data[self.edge_field][:, self.idp.orbpairtype_maps[opairtype]] = HR

        # compute onsite blocks
        if self.onsite:
            node_feature = data[self.node_field].clone()
            data[self.node_field] = torch.zeros(n_node, self.idp.reduced_matrix_element, dtype=self.dtype, device=self.device)

            for orbtype in self.idp_sk.skonsitetype_maps.keys():
                # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                # for better performance
                l = anglrMId[re.findall(r"[a-z]", orbtype)[0]]

                skparam = node_feature[:, self.idp_sk.skonsitetype_maps[orbtype]].reshape(n_node, -1, 1)
                HR = torch.eye(2*l+1, dtype=self.dtype, device=self.device)[None, None, :, :] * skparam[:,:, None, :] # shape (N, n_pair, 2l1+1, 2l2+1)
                # the onsite block doesnot have rotation

                data[self.node_field][:, self.idp.orbpairtype_maps[orbtype+"-"+orbtype]] = HR.reshape(n_node, -1)

        if self.soc:
            assert data[AtomicDataDict.NODE_SOC_SWITCH_KEY].all(), "The SOC switch is not turned on in data by soc is set to True."
            soc_feature = data[AtomicDataDict.NODE_SOC_KEY]
            data[AtomicDataDict.NODE_SOC_KEY] = torch.zeros(n_node, self.idp.reduced_soc_matrix_elemet, dtype= self.cdtype, device=self.device)
            for otype in self.idp_sk.sksoc_maps.keys():
                lsymbol = re.findall(r"[a-z]", otype)[0]
                l = anglrMId[lsymbol]
                socparam = soc_feature[:, self.idp_sk.sksoc_maps[otype]].reshape(n_node, -1, 1)
                HR = self.soc_base_matrix[lsymbol][None, None, :, :] * socparam[:,:, None, :]
                HR_upup_updn = HR[:,:,0:2*l+1,:]
                data[AtomicDataDict.NODE_SOC_KEY][:, self.idp.orbpair_soc_maps[otype+"-"+otype]] = HR_upup_updn.reshape(n_node, -1)

        # compute if strain effect is included
        # this is a little wired operation, since it acting on somekind of a edge(strain env) feature, and summed up to return a node feature.
        if self.strain:
            n_onsitenv = len(data[AtomicDataDict.ONSITENV_FEATURES_KEY])
            for opairtype in self.idp.orbpairtype_maps.keys(): # save all env direction and pair direction like sp and ps, but only get sp
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                # opairtype = opair[1]+"-"+opair[4]
                n_skp = min(l1, l2)+1 # number of reduced matrix element
                skparam = data[AtomicDataDict.ONSITENV_FEATURES_KEY][:, self.idp_sk.orbpairtype_maps[opairtype]].reshape(n_onsitenv, -1, n_skp)
                rme = skparam @ self.sk2irs[opairtype].T # shape (N, n_pair, n_rme)
                rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)

                H_z = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                    rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
                
                angle = xyz_to_angles(data[AtomicDataDict.ONSITENV_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
                rot_mat_L = wigner_D(int(l1), angle[0], angle[1], torch.zeros_like(angle[0]))
                rot_mat_R = wigner_D(int(l2), angle[0], angle[1], torch.zeros_like(angle[0]))
                # rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l1+1, 2l1+1)
                # rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l2+1, 2l2+1)

                HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R) # shape (N, n_pair, 2l1+1, 2l2+1)

                HR = scatter(src=HR, index=data[AtomicDataDict.ONSITENV_INDEX_KEY][0], dim=0, reduce="sum") # shape (n_node, n_pair, 2l1+1, 2l2+1)
                # A-B o1-o2 (A-B o2-o1)= (B-A o1-o2)
                
                data[self.node_field][:, self.idp.orbpairtype_maps[opairtype]] += HR.flatten(1, len(HR.shape)-1) # the index type [node/pair] should align with the index of for loop
            
        return data

    def _initialize_CG_basis(self, pairtype: str):
        """
        The function initializes a Clebsch-Gordan basis for a given pair type.
        
        :param pairtype: The parameter "pairtype" is a string that represents a pair of angular momentum
        quantum numbers. It is expected to have a length of 3, where the first and third characters
        represent the angular momentum quantum numbers of two particles, and the second character
        represents the type of interaction between the particles
        :type pairtype: str
        :return: the CG basis, which is a tensor containing the Clebsch-Gordan coefficients for the given
        pairtype.
        """
        

        irs_index = {
            's-s': [0],
            's-p': [1],
            's-d': [2],
            's-f': [3],
            'p-s': [1],
            'p-p': [0,6],
            'p-d': [1,11],
            'd-s': [2],
            'd-p': [1,11],
            'd-d': [0,6,20]
        }

        l1, l2 = anglrMId[pairtype[0]], anglrMId[pairtype[2]]
        assert l1 <=2 and l2 <=2, "Only support l<=2, ie. s, p, d orbitals at present."
        cg = []
        for l_ird in range(abs(l2-l1), l2+l1+1):
            cg.append(wigner_3j(int(l1), int(l2), int(l_ird), dtype=self.dtype, device=self.device) * (2*l_ird+1)**0.5)
        
        cg = torch.cat(cg, dim=-1)[:,:,irs_index[pairtype]]
        
        return cg
    
class SKHamiltonian(torch.nn.Module):
    # transform SK parameters to SK hamiltonian with E3 CG basis, strain is included.
    def __init__(
        self, 
        basis: Dict[str, Union[str, list]]=None,
        idp_sk: Union[OrbitalMapper, None]=None,
        dtype: Union[str, torch.dtype] = torch.float32, 
        device: Union[str, torch.device] = torch.device("cpu"),
        edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
        onsite: bool = False,
        strain: bool = False,
        soc: bool = False,
        **kwargs,
        ) -> None:
        super(SKHamiltonian, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if isinstance(device, str):
            device = torch.device(device)
        self.dtype = dtype
        self.device = device
        self.onsite = onsite
        self.soc = soc

        if basis is not None:
            self.idp_sk = OrbitalMapper(basis, method="sktb", device=device)
            if idp_sk is not None:
                assert idp_sk.basis == self.idp_sk.basis, "The basis of idp and basis should be the same."
        else:
            assert idp_sk is not None, "Either basis or idp should be provided."
            self.idp_sk = idp_sk
        # initilize a e3 indexmapping to help putting the orbital wise blocks into atom-pair wise format
        self.idp = OrbitalMapper(self.idp_sk.basis, method="e3tb", device=device)
        self.basis = self.idp.basis
        self.skbasis = {}
        self.strain = strain
        self.edge_field = edge_field
        self.node_field = node_field

        self.idp_sk.get_orbpair_maps()
        self.idp_sk.get_skonsite_maps()
        self.idp.get_orbpair_maps()

        if self.soc:
            self.idp_sk.get_sksoc_maps()
            self.idp.get_orbpair_soc_maps()

        pairtypes = self.idp_sk.orbpairtype_maps.keys()
        
        for pairtype in pairtypes:
            # self.cgbasis.setdefault(pairtype, None)
            bb = self._initialize_basis(pairtype)
            self.skbasis[pairtype] = bb

        if self.soc:
            self.soc_base_matrix = {
                's':get_soc_matrix_cubic_basis(orbital='s', device=self.device, dtype=self.dtype),
                'p':get_soc_matrix_cubic_basis(orbital='p', device=self.device, dtype=self.dtype),
                'd':get_soc_matrix_cubic_basis(orbital='d', device=self.device, dtype=self.dtype)
            }
            self.cdtype =  float2comlex(self.dtype)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # transform sk parameters to irreducible matrix element

        assert data[self.edge_field].shape[1] == self.idp_sk.reduced_matrix_element
        if self.onsite:
            assert data[self.node_field].shape[1] == self.idp_sk.n_onsite_Es
            n_node = data[self.node_field].shape[0]
            
        n_edge = data[self.edge_field].shape[0]
        

        edge_features = data[self.edge_field].clone()
        data[self.edge_field] = torch.zeros((n_edge, self.idp.reduced_matrix_element), dtype=self.dtype, device=self.device)

        # for hopping blocks
        for opairtype in self.idp_sk.orbpairtype_maps.keys():
            l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
            n_skp = min(l1, l2)+1 # number of reduced matrix element
            skparam = edge_features[:, self.idp_sk.orbpairtype_maps[opairtype]].reshape(n_edge, -1, n_skp)
            skparam = skparam.transpose(1,2) # shape (N, n_skp, n_pair)
            
            H_z = torch.sum(self.skbasis[opairtype][None,:,:,:,None] * \
                skparam[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
            
            # rotation
            # when get the angle, the xyz vector should be transformed to yzx.
            angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
            # The roataion matrix is SO3 rotation, therefore Irreps(l,1), is used here.
            rot_mat_L = wigner_D(int(l1), angle[0], angle[1], torch.zeros_like(angle[0]))
            rot_mat_R = wigner_D(int(l2), angle[0], angle[1], torch.zeros_like(angle[0]))
            # rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l1+1, 2l1+1)
            # rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l2+1, 2l2+1)
            
            # Here The string to control einsum is important, the order of the index should be the same as the order of the tensor
            # H_z = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R) # shape (N, n_pair, 2l1+1, 2l2+1)
            HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R).reshape(n_edge, -1) # shape (N, n_pair * 2l2+1 * 2l2+1)
            
            if l1 < l2:
                HR = HR * (-1)**(l1+l2)
            
            data[self.edge_field][:, self.idp.orbpairtype_maps[opairtype]] = HR

        # compute onsite blocks
        if self.onsite:
            node_feature = data[self.node_field].clone()
            data[self.node_field] = torch.zeros(n_node, self.idp.reduced_matrix_element, dtype=self.dtype, device=self.device)

            for orbtype in self.idp_sk.skonsitetype_maps.keys():
                # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                # for better performance
                l = anglrMId[re.findall(r"[a-z]", orbtype)[0]]

                skparam = node_feature[:, self.idp_sk.skonsitetype_maps[orbtype]].reshape(n_node, -1, 1)
                HR = torch.eye(2*l+1, dtype=self.dtype, device=self.device)[None, None, :, :] * skparam[:,:, None, :] # shape (N, n_pair, 2l1+1, 2l2+1)
                # the onsite block doesnot have rotation

                data[self.node_field][:, self.idp.orbpairtype_maps[orbtype+"-"+orbtype]] = HR.reshape(n_node, -1)

        if self.soc:
            assert data[AtomicDataDict.NODE_SOC_SWITCH_KEY].all(), "The SOC switch is not turned on in data by soc is set to True."
            soc_feature = data[AtomicDataDict.NODE_SOC_KEY]
            data[AtomicDataDict.NODE_SOC_KEY] = torch.zeros(n_node, self.idp.reduced_soc_matrix_elemet, dtype= self.cdtype, device=self.device)
            for otype in self.idp_sk.sksoc_maps.keys():
                lsymbol = re.findall(r"[a-z]", otype)[0]
                l = anglrMId[lsymbol]
                socparam = soc_feature[:, self.idp_sk.sksoc_maps[otype]].reshape(n_node, -1, 1)
                HR = self.soc_base_matrix[lsymbol][None, None, :, :] * socparam[:,:, None, :]
                HR_upup_updn = HR[:,:,0:2*l+1,:]
                data[AtomicDataDict.NODE_SOC_KEY][:, self.idp.orbpair_soc_maps[otype+"-"+otype]] = HR_upup_updn.reshape(n_node, -1)

        # compute if strain effect is included
        # this is a little wired operation, since it acting on somekind of a edge(strain env) feature, and summed up to return a node feature.
        if self.strain:
            n_onsitenv = len(data[AtomicDataDict.ONSITENV_FEATURES_KEY])
            for opairtype in self.idp.orbpairtype_maps.keys(): # save all env direction and pair direction like sp and ps, but only get sp
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                # opairtype = opair[1]+"-"+opair[4]
                n_skp = min(l1, l2)+1 # number of reduced matrix element
                skparam = data[AtomicDataDict.ONSITENV_FEATURES_KEY][:, self.idp_sk.orbpairtype_maps[opairtype]].reshape(n_onsitenv, -1, n_skp)
                skparam = skparam.transpose(1,2) # shape (N, n_skp, n_pair)

                H_z = torch.sum(self.skbasis[opairtype][None,:,:,:,None] * \
                    skparam[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
                
                angle = xyz_to_angles(data[AtomicDataDict.ONSITENV_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
                rot_mat_L = wigner_D(int(l1), angle[0], angle[1], torch.zeros_like(angle[0]))
                rot_mat_R = wigner_D(int(l2), angle[0], angle[1], torch.zeros_like(angle[0]))
                # rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l1+1, 2l1+1)
                # rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0].cpu(), angle[1].cpu(), torch.tensor(0., dtype=self.dtype)).to(self.device) # tensor(N, 2l2+1, 2l2+1)

                HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R) # shape (N, n_pair, 2l1+1, 2l2+1)

                HR = scatter(src=HR, index=data[AtomicDataDict.ONSITENV_INDEX_KEY][0], dim=0, reduce="sum") # shape (n_node, n_pair, 2l1+1, 2l2+1)
                # A-B o1-o2 (A-B o2-o1)= (B-A o1-o2)
                
                data[self.node_field][:, self.idp.orbpairtype_maps[opairtype]] += HR.flatten(1, len(HR.shape)-1) # the index type [node/pair] should align with the index of for loop
            
        return data

    def _initialize_basis(self, pairtype: str):
        """
        The function initializes a slater-koster used basis for a given pair type, to map the sk parameter 
        to the hamiltonian block rotated onto z-axis
        
        :param pairtype: The parameter "pairtype" is a string that represents a pair of angular momentum
        quantum numbers. It is expected to have a length of 3, where the first and third characters
        represent the angular momentum quantum numbers of two particles, and the second character
        represents the type of interaction between the particles
        :type pairtype: str
        :return: the basis, which contains a selection matrix with 0/1 as its value.
        pairtype.
        """

        basis = []
        l1, l2 = anglrMId[pairtype[0]], anglrMId[pairtype[2]]
        assert l1 <=5 and l2 <=5, "Only support l<=5, ie. s, p, d, f, g, h orbitals at present."
        for im in range(0, min(l1,l2)+1):
            mat = torch.zeros((2*l1+1, 2*l2+1), dtype=self.dtype, device=self.device)
            if im == 0:
                mat[l1,l2] = 1.
            else:
                mat[l1+im,l2+im] = 1.
                mat[l1-im, l2-im] = 1.
            basis.append(mat)
        basis = torch.stack(basis, dim=-1)
        
        return basis