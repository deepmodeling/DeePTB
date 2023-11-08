"""This file refactor the SK and E3 Rotation in dptb/hamiltonian/transform_se3.py], it will take input of AtomicDataDict.Type
    perform rotation from irreducible matrix element / sk parameters in EDGE/NODE FEATURE, and output the atomwise/ pairwise hamiltonian
    as the new EDGE/NODE FEATURE. The rotation should also be a GNN module and speed uptable by JIT. The HR2HK should also be included here.
    The indexmapping should ne passed here.
    """

import torch
from e3nn.o3 import wigner_3j, Irrep, xyz_to_angles, Irrep
from dptb.utils.constants import h_all_types, anglrMId
from typing import Tuple, Union, Dict
from dptb.utils.index_mapping import Index_Mapings_e3
from dptb.data import AtomicDataDict
from torch_scatter import scatter

#TODO: 1. jit acceleration 2. GPU support 3. rotate AB and BA bond together.

# The `E3Hamiltonian` class is a PyTorch module that represents a Hamiltonian for a system with a
# given basis and can perform forward computations on input data.
class E3Hamiltonian(torch.nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]],
            decompose: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")
            ) -> None:
        
        super(E3Hamiltonian, self).__init__()
        self.dtype = dtype
        self.device = device
        self.idp = Index_Mapings_e3(basis, method="e3tb")
        self.basis = self.idp.basis
        self.cgbasis = {}
        self.decompose = decompose

        # initialize the CG basis
        self.idp.get_nodetype_maps()
        self.idp.get_pairtype_maps()
        pairtypes = self.idp.pairtype_maps.keys()
        for pairtype in pairtypes:
            self._initialize_CG_basis(pairtype)
            

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

        assert data[AtomicDataDict.EDGE_FEATURES_KEY].shape[1] == self.idp.edge_reduced_matrix_element
        assert data[AtomicDataDict.NODE_FEATURES_KEY].shape[1] == self.idp.node_reduced_matrix_element

        n_edge = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        n_node = data[AtomicDataDict.NODE_FEATURES_KEY].shape[0]

        if not self.decompose:
            # The EDGE_FEATURES_KEY and NODE_FAETURE_KEY are the reduced matrix elements

            # compute hopping blocks
            for opairtype in self.idp.pairtype_maps.keys():
                # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                # for better performance
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                n_rme = (2*l1+1) * (2*l2+1) # number of reduced matrix element
                rme = data[AtomicDataDict.EDGE_FEATURES_KEY][:, self.idp.pairtype_maps[opairtype]]
                rme = rme.reshape(n_edge, -1, n_rme)
                rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)
                
                H_z = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                    rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
                
                # rotation
                angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
                rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l1+1, 2l1+1)
                rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l2+1, 2l2+1)
                HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R).reshape(n_edge, -1) # shape (N, n_pair * n_rme)
                
                data[AtomicDataDict.EDGE_FEATURES_KEY][:, self.idp.pairtype_maps[opairtype]] = HR

            # compute onsite blocks
            for opairtype in self.idp.nodetype_maps.keys():
                # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                # for better performance
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                n_rme = (2*l1+1) * (2*l2+1) # number of reduced matrix element
                rme = data[AtomicDataDict.NODE_FEATURES_KEY][:, self.idp.nodetype_maps[opairtype]]
                rme = rme.reshape(n_node, -1, n_rme)
                rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)
                
                HR = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                    rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
                HR = HR.permute(0,3,1,2).reshape(n_node, -1)

                # the onsite block doesnot have rotation
                data[AtomicDataDict.NODE_FEATURES_KEY][:, self.idp.nodetype_maps[opairtype]] = HR
        
        else:
            for opairtype in self.idp.pairtype_maps.keys():
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                nL, nR = 2*l1+1, 2*l2+1
                HR = data[AtomicDataDict.EDGE_FEATURES_KEY][:, self.idp.pairtype_maps[opairtype]]
                HR = HR.reshape(n_edge, -1, nL, nR) # shape (N, n_pair, nL, nR)

                # rotation
                angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
                rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l1+1, 2l1+1)
                rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l2+1, 2l2+1)
                H_z = torch.einsum("nml, nqmo, nok -> nlkq", rot_mat_L, HR, rot_mat_R) # shape (N, nL, nR, n_pair)

                rme = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                    H_z[:,:,:,None,:], dim=(1,2)) # shape (N, n_rme, n_pair)
                rme = rme.transpose(1,2).reshape(n_edge, -1)

                data[AtomicDataDict.EDGE_FEATURES_KEY][:, self.idp.pairtype_maps[opairtype]] = rme

            for opairtype in self.idp.nodetype_maps.keys():
                # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                # for better performance
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                nL, nR = 2*l1+1, 2*l2+1 # number of reduced matrix element
                HR = data[AtomicDataDict.NODE_FEATURES_KEY][:, self.idp.nodetype_maps[opairtype]]
                HR = HR.reshape(n_node, -1, nL, nR).permute(0,2,3,1)# shape (N, nL, nR, n_pair)
                
                rme = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                    HR[:,:,:,None,:], dim=(1,2)) # shape (N, n_rme, n_pair)
                rme = rme.transpose(1,2).reshape(n_node, -1)

                # the onsite block doesnot have rotation
                print(rme.shape, data[AtomicDataDict.NODE_FEATURES_KEY][:, self.idp.nodetype_maps[opairtype]].shape, opairtype)
                data[AtomicDataDict.NODE_FEATURES_KEY][:, self.idp.nodetype_maps[opairtype]] = rme
            
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
    
class SKHamiltonian(torch.nn.Module):
    # transform SK parameters to SK hamiltonian with E3 CG basis, strain is included.
    def __init__(
        self, 
        basis: Dict[str, Union[str, list]],
        dtype: Union[str, torch.dtype] = torch.float32, 
        device: Union[str, torch.device] = torch.device("cpu")
        ) -> None:
        super(SKHamiltonian, self).__init__()
        self.dtype = dtype
        self.device = device
        self.idp = Index_Mapings_e3(basis, method="sktb")
        # initilize a e3 indexmapping to help putting the orbital wise blocks into atom-pair wise format
        self.idp_e3 = Index_Mapings_e3(basis, method="e3tb")
        self.basis = self.idp.basis
        self.cgbasis = {}

        self.idp.get_nodetype_maps()
        self.idp.get_pairtype_maps()
        self.idp_e3.get_nodetype_maps()
        self.idp_e3.get_pairtype_maps()

        pairtypes = self.idp.pairtype_maps.keys()
        for pairtype in pairtypes:
            self._initialize_CG_basis(pairtype)

        self.sk2irs = {
            'ss': torch.tensor([[1.]], dtype=self.rot_type, device=self.device),
            'sp': torch.tensor([[1.]], dtype=self.rot_type, device=self.device),
            'sd': torch.tensor([[1.]], dtype=self.rot_type, device=self.device),
            'pp': torch.tensor([
                [3**0.5/3,2/3*3**0.5],[6**0.5/3,-6**0.5/3]
            ], dtype=self.dtype, device=self.device
            ),
            'pd':torch.tensor([
                [(2/5)**0.5,(6/5)**0.5],[(3/5)**0.5,-2/5**0.5]
            ], dtype=self.dtype, device=self.device
            ),
            'dd':torch.tensor([
                [5**0.5/5, 2*5**0.5/5, 2*5**0.5/5],
                [2*(1/14)**0.5,2*(1/14)**0.5,-4*(1/14)**0.5],
                [3*(2/35)**0.5,-4*(2/35)**0.5,(2/35)**0.5]
                ], dtype=self.dtype, device=self.device
            )
        }

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # transform sk parameters to irreducible matrix element

        assert data[AtomicDataDict.EDGE_FEATURES_KEY].shape[1] == self.idp.edge_reduced_matrix_element
        assert data[AtomicDataDict.NODE_FEATURES_KEY].shape[1] == self.idp.node_reduced_matrix_element

        n_edge = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        n_node = data[AtomicDataDict.NODE_FEATURES_KEY].shape[0]

        edge_features = data[AtomicDataDict.EDGE_FEATURES_KEY].clone()
        data[AtomicDataDict.EDGE_FEATURES_KEY] = torch.zeros(n_edge, self.idp_e3.edge_reduced_matrix_element)

        # for hopping blocks
        for opairtype in self.idp.pairtype_maps.keys():
            l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
            n_skp = min(l1, l2)+1 # number of reduced matrix element
            skparam = edge_features[:, self.idp.pairtype_maps[opairtype]].reshape(n_edge, -1, n_skp)
            rme = skparam @ self.sk2irs[opairtype].T # shape (N, n_pair, n_rme)
            rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)
            
            H_z = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
            
            # rotation
            angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
            rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l1+1, 2l1+1)
            rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l2+1, 2l2+1)
            HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R).reshape(n_edge, -1) # shape (N, n_pair * 2l2+1 * 2l2+1)
            
            data[AtomicDataDict.EDGE_FEATURES_KEY][:, self.idp_e3.pairtype_maps[opairtype]] = HR

        # compute onsite blocks
        node_feature = data[AtomicDataDict.NODE_FEATURES_KEY].clone()
        data[AtomicDataDict.NODE_FEATURES_KEY] = torch.zeros(n_node, self.idp_e3.node_reduced_matrix_element)

        for opairtype in self.idp.nodetype_maps.keys():
            # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
            # for better performance
            l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
            if l1 != l2:
                continue # off-diagonal term in sktb format
            else:
                skparam = node_feature[:, self.idp.nodetype_maps[opairtype]].reshape(n_node, -1, 1)
                
                HR = torch.eye(2*l1+1, dtype=self.dtype, device=self.device)[None, None, :, :] * skparam[:,:, None, :] # shape (N, n_pair, 2l1+1, 2l2+1)

                # the onsite block doesnot have rotation
                data[AtomicDataDict.NODE_FEATURES_KEY][:, self.idp_e3.nodetype_maps[opairtype]] = HR.reshape(n_node, -1)

        # compute if strain effect is included
        # this is a little wired operation, since it acting on somekind of a edge(strain env) feature, and summed up to return a node feature.
        if data.get(AtomicDataDict.ONSITENV_FEATURE_KEY, None):
            n_onsitenv = len(data[AtomicDataDict.ONSITENV_FEATURES_KEY])
            for opairtype in self.idp.nodetype_maps.keys():
                l1, l2 = anglrMId[opairtype[0]], anglrMId[opairtype[2]]
                n_skp = min(l1, l2)+1 # number of reduced matrix element
                skparam = data[AtomicDataDict.ONSITENV_FEATURES_KEY][:, self.idp.pairtype_maps[opairtype]].reshape(n_onsitenv, -1, n_skp)
                rme = skparam @ self.sk2irs[opairtype].T # shape (N, n_pair, n_rme)
                rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)

                H_z = torch.sum(self.cgbasis[opairtype][None,:,:,:,None] * \
                    rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
                
                angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY][:,[1,2,0]]) # (tensor(N), tensor(N))
                rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l1+1, 2l1+1)
                rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l2+1, 2l2+1)
                HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R).reshape(n_onsitenv, -1).sum # shape (N, n_pair * 2l2+1 * 2l2+1)

                HR = scatter(HR, data[AtomicDataDict.ONSITENV_INDEX_KEY], 0, None, "sum") # shape (n_node, n_pair * 2l2+1 * 2l2+1)

                data[AtomicDataDict.NODE_FEATURES_KEY][:, self.idp_e3.nodetype_maps[opairtype]] += HR

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

        irs_index = {
            's-s': [0],
            's-p': [1],
            's-d': [2],
            'p-p': [0,6],
            'p-d': [1,11],
            'd-d': [0,6,20]
        }

        l1, l2 = anglrMId[pairtype[0]], anglrMId[pairtype[2]]

        cg = []
        for l_ird in range(abs(l2-l1), l2+l1+1):
            cg.append(wigner_3j(int(l1), int(l2), int(l_ird), dtype=self.dtype, device=self.device) * (2*l_ird+1)**0.5)
        
        cg = torch.cat(cg, dim=-1)[:,:,irs_index[pairtype]]
        self.cgbasis[pairtype] = cg

        return cg