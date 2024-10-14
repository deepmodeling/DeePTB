from dptb.nn.dftb.sk_param import SKParam
from dptb.nn.dftb.hopping_dftb import HoppingIntp
import torch
from dptb.nn.sktb.hopping import HoppingFormula
from dptb.nn.sktb import OnsiteFormula, bond_length_list
from dptb.nn.sktb.cov_radiiDB import Covalent_radii
from dptb.nn.sktb.bondlengthDB import atomic_radius_v1
from dptb.utils.constants import atomic_num_dict
from functorch import vmap
import matplotlib.pyplot as plt
from torch.optim import Adam, LBFGS, RMSprop, SGD
from torch.optim.lr_scheduler import ExponentialLR
from dptb.nn.nnsk import NNSK
from dptb.nn.sktb.onsite import onsite_energy_database
from dptb.data.AtomicData import get_r_map
import numpy as np
from typing import Union
import logging

log = logging.getLogger(__name__)

class dftb:
    def __init__(self, basis, skdata, cal_rcuts=False):
        self.param = SKParam(basis=basis, skdata=skdata, cal_rcuts=cal_rcuts)
        self.atomic_r_min = self.param.atomic_r_min
        self.atomic_r_max = self.param.atomic_r_max
        self.idp_sk = self.param.idp_sk

        self.param = self.param.format_skparams(self.param.skdict)
        self.hopping = HoppingIntp(num_ingrls=self.param["Hopping"].shape[1])
        self.overlap = HoppingIntp(num_ingrls=self.param["Overlap"].shape[1])
        self.bond_types = self.idp_sk.bond_types
        self.bond_type_to_index = {bt: i for i, bt in enumerate(self.idp_sk.bond_types)}

    def __call__(self, r, bond_indices = None, mode="hopping"):
        out = []
        if bond_indices is None:
            bond_indices = torch.arange(len(self.idp_sk.bond_types))

        assert len(bond_indices) == len(r), "The bond_indices and r should have the same length."
        
        for i, ind in enumerate(bond_indices):
            out.append(self.hopping.get_skhij(rij=r[i], xx=self.param["Distance"], yy=self.param[mode[0].upper()+mode[1:]][ind]))
        
        return torch.stack(out)
    
class DFTB2NNSK:
    def __init__(self, basis, skdata, functype='poly2pow', rs=None, w=0.2, cal_rcuts=False, atomic_radius='cov'):
        self.dftb = dftb(basis=basis, skdata=skdata, cal_rcuts=cal_rcuts)
        self.functype = functype
        self.idp_sk = self.dftb.idp_sk
        # self.rs = rs
        self.w = w         
        self.nnsk_hopping = HoppingFormula(functype=self.functype)
        self.nnsk_overlap = HoppingFormula(functype=self.functype, overlap=True)
        self.hopping_params = torch.nn.Parameter(torch.randn(len(self.idp_sk.bond_types), self.dftb.hopping.num_ingrls, self.nnsk_hopping.num_paras))
        self.overlap_params = torch.nn.Parameter(torch.randn(len(self.idp_sk.bond_types), self.dftb.hopping.num_ingrls, self.nnsk_hopping.num_paras))

        if isinstance(atomic_radius, str):
            if atomic_radius == 'cov':
                atomic_radius = Covalent_radii
            elif atomic_radius == 'v1':
                atomic_radius = atomic_radius_v1
            else:
                raise ValueError("The atomic_radius should be either str of 'cov' or 'v1' or a dict.")
        else:
            assert isinstance(atomic_radius, dict), "The atomic_radius should be either str of 'cov' or 'v1' or a dict."
   
        for at in basis.keys():
            assert at in atomic_radius, f"The atomic radius for {at} is not provided."
            assert atomic_radius[at] is not None, f"The atomic radius for {at} is None."
        atomic_numbers = [atomic_num_dict[key] for key in basis.keys()]

        self.atomic_radius_list =  torch.zeros(int(max(atomic_numbers))) - 100
        for at in basis.keys():
            assert at in atomic_radius and atomic_radius[at] is not None, f"The atomic radius for {at} is not provided."
            radii = atomic_radius[at]
            
            self.atomic_radius_list[atomic_num_dict[at]-1] = radii

        if rs is None:
            if not cal_rcuts:
                log.warning('rs is not provided, the r_max and r_min will be calculated from the dftb parameters.'), 
            self.rs = self.dftb.atomic_r_max
            self.r_map = get_r_map(self.dftb.atomic_r_max)
            self.r_max = []
            self.r_min = []
            for ibt in self.idp_sk.bond_types:
                it = ibt.split('-')[0]
                jt = ibt.split('-')[1]
                #print(atomic_r_min_dict[it],atomic_r_min_dict[jt],atomic_r_min_dict[it] + atomic_r_min_dict[jt])
                self.r_max.append(self.dftb.atomic_r_max[it] + self.dftb.atomic_r_max[jt])
                self.r_min.append(self.dftb.atomic_r_min[it] + self.dftb.atomic_r_min[jt])
            self.r_max = torch.tensor(self.r_max, dtype=torch.float32).reshape(-1,1)
            self.r_min = torch.tensor(self.r_min, dtype=torch.float32).reshape(-1,1)

        else:
            if not cal_rcuts :
                assert isinstance(rs, (float, dict))
                self.rs = rs
                self.r_max = None
                self.r_min = None
            else:
                log.error("The rs is seted but the cal_rcuts is also set true. you should only set one of them.")
                raise ValueError("The rs is seted but the cal_rcuts is also set true. you should only set one of them.")
            

    def symmetrize(self):
        reflective_bonds = np.array([self.idp_sk.bond_to_type["-".join(self.idp_sk.type_to_bond[i].split("-")[::-1])] for i  in range(len(self.idp_sk.bond_types))])
        params = self.hopping_params.data
        reflect_params = params[reflective_bonds]
        for k in self.idp_sk.orbpair_maps.keys():
            iorb, jorb = k.split("-")
            if iorb == jorb:
                # This is to keep the symmetry of the hopping parameters for the same orbital pairs
                # As-Bs = Bs-As; we need to do this because for different orbital pairs, we only have one set of parameters, 
                # eg. we only have As-Bp and Bs-Ap, but not Ap-Bs and Bp-As; and we will use Ap-Bs = Bs-Ap and Bp-As = As-Bp to calculate the hopping integral
                self.hopping_params.data[:,self.idp_sk.orbpair_maps[k],:] = 0.5 * (params[:,self.idp_sk.orbpair_maps[k],:] + reflect_params[:,self.idp_sk.orbpair_maps[k],:])

        params = self.overlap_params.data
        reflect_params = params[reflective_bonds]
        for k in self.idp_sk.orbpair_maps.keys():
            iorb, jorb = k.split("-")
            if iorb == jorb:
                self.overlap_params.data[:,self.idp_sk.orbpair_maps[k],:] = 0.5 * (params[:,self.idp_sk.orbpair_maps[k],:] + reflect_params[:,self.idp_sk.orbpair_maps[k],:])
        
        return True
    
    def step(self, r):
        assert r.shape[0] == len(self.curr_bond_indices)
        r = r.reshape(-1)
        bond_ind_r_shp = self.curr_bond_indices.reshape(-1)

        edge_number = self.idp_sk.untransform_bond(bond_ind_r_shp).T
        r0 = self.atomic_radius_list[edge_number-1].sum(0)  # bond length r0 = r1 + r2. (r1, r2 are atomic radii of the two atoms)

        if isinstance(self.rs, dict):
            assert hasattr(self, "r_map")
            r_cutoffs = self.r_map[edge_number-1].sum(0)
            assert torch.allclose(r_cutoffs,self.r_max[bond_ind_r_shp].reshape(-1))
        else:
            assert isinstance(self.rs, (int,float))
            r_cutoffs = self.rs
        
        hopping = self.nnsk_hopping.get_skhij(
            rij=r,
            paraArray=self.hopping_params[bond_ind_r_shp], # [N_edge, n_pairs, n_paras],
            rs=r_cutoffs,
            w=self.w,
            r0=r0
            ) # [N_edge, n_pairs]
        
        overlap = self.nnsk_overlap.get_skhij(
            rij=r,
            paraArray=self.overlap_params[bond_ind_r_shp], # [N_edge, n_pairs, n_paras],
            rs=r_cutoffs,
            w=self.w,
            r0=r0
            )
        return hopping, overlap
    

    def viz(self, atom_a:str, atom_b:str=None, r_min:Union[float, int]=None, r_max:Union[float, int]=None, nsample=100):
        with torch.no_grad():
            if atom_b is None:
                atom_b = atom_a
            bond_type = atom_a + "-" + atom_b
            bond_index = torch.tensor([self.idp_sk.bond_types.index(bond_type)])
            self.curr_bond_indices = bond_index
            if r_min is None and r_max is None:
                assert self.r_min is not None and self.r_max is not None, "When both r_min and r_max  are None. cal_rcuts=True when initializing the DFTB2NNSK object."
                r_min_ = self.r_min[bond_index]
                r_max_ = self.r_max[bond_index]
            else:
                assert r_min is not None and r_max is not None, "bothr_min and r_max should be provided or both None."
                r_min_ = r_min
                r_max_ = r_max

            r = torch.linspace(0, 1, steps=100).reshape(1,-1).repeat(len(self.curr_bond_indices),1) * (r_max_ - r_min_) + r_min_

            hops = vmap(self.step,in_dims=1)(r)

            
            dftb_hopping = self.dftb(r, bond_indices = self.curr_bond_indices, mode="hopping").permute(1,0,2)
            dftb_overlap = self.dftb(r, bond_indices = self.curr_bond_indices, mode="overlap").permute(1,0,2)

            r = r.numpy()
            fig = plt.figure(figsize=(6,4))
            # hops[0] shape - [n_r, n_edge, n_skintegrals]

            for i in range(hops[0].shape[1]):
                plt.plot(r[i], hops[0][:,i, :-1].detach().numpy(), c="C"+str(i))
                plt.plot(r[i], hops[0][:,i, -1].detach().numpy(), c="C"+str(i))
                plt.plot(r[i], dftb_hopping[:,i, :-1].numpy(), c="C"+str(i), linestyle="--")
                plt.plot(r[i], dftb_hopping[:,i, -1].numpy(), c="C"+str(i), linestyle="--")
            plt.title("hoppings")
            plt.xlabel("r(angstrom)")
            plt.tight_layout()
            # plt.legend()
            plt.show()

            fig = plt.figure(figsize=(6,4))
            for i in range(hops[1].shape[1]):
                plt.plot(r[i], hops[1][:,i, :-1].detach().numpy(), c="C"+str(i))
                plt.plot(r[i], hops[1][:,i, -1].detach().numpy(), c="C"+str(i))
                plt.plot(r[i], dftb_overlap[:,i, :-1].numpy(), c="C"+str(i), linestyle="--")
                plt.plot(r[i], dftb_overlap[:,i, -1].numpy(), c="C"+str(i), linestyle="--")
            plt.title("overlaps")
            plt.xlabel("r(angstrom)")
            plt.tight_layout()
            # plt.legend()
            plt.show()
    
        
    def optimize(self, r_min=None, r_max=None, nsample=256, nstep=40000, lr=1e-1, dis_freq=1000, method="RMSprop", viz=False, max_elmt_batch=4):
        """
        Optimize the parameters of the neural network model.

        Args:
            r_min (float): The minimum value for the random range of r.
            r_max (float): The maximum value for the random range of r.
            nsample (int): The number of samples to generate for r.
            nstep (int): The number of optimization steps to perform.
            lr (float): The learning rate for the optimizer.
            dis_freq (int): The frequency at which to display the loss during optimization.
            method (str): The optimization method to use. Supported methods are "RMSprop" and "LBFGS".
            viz (bool): Whether to visualize the optimized results.
            max_elmt_batch (int): max_elmt_batch^2 defines The maximum number of bond types to optimize in each batch.
             ie. if max_elmt_batch=4, we will optimize 16 bond types in each batch.

        Returns:
            bool: True if the optimization is successful.

        Raises:
            NotImplementedError: If the specified optimization method is not supported.
        """

        if method=="RMSprop":
            optimizer = RMSprop([self.hopping_params, self.overlap_params], lr=lr, momentum=0.2)
        elif method=="LBFGS":
            optimizer = LBFGS([self.hopping_params, self.overlap_params], lr=lr)
        else:
            raise NotImplementedError
        
        lrscheduler = ExponentialLR(optimizer, gamma=0.9998)
        self.loss = torch.tensor(0.)


        def closure():
            optimizer.zero_grad()
            if r_min is None and r_max is None:
                assert self.r_min is not None and self.r_max is not None, "When both r_min and r_max  are None. cal_rcuts=True when initializing the DFTB2NNSK object."
                r_min_ = self.r_min[self.curr_bond_indices]
                r_max_ = self.r_max[self.curr_bond_indices]
            else:
                assert r_min is not None and r_max is not None, "bothr_min and r_max should be provided or both None."
                r_min_ = torch.tensor(r_min)
                r_max_ = r_max
            
            # r = torch.rand([len(self.curr_bond_indices),nsample]) * (r_max_ - r_min_) + r_min_
            # 用 gauss 分布的随机数，重点采样在 r_min 和 r_max范围中心区域的值
            # std = (r_max_ - r_min_)/8  4 sigma 展宽
            # mean = (r_max_ + r_min_)/2 中心为 r_max 和 r_min 的中点
            # r =torch.randn([len(self.curr_bond_indices),nsample]) * (r_max_ - r_min_)/8 + (r_max_ + r_min_)/2
            r = self.truncated_normal(shape=[len(self.curr_bond_indices),nsample], min_val=r_min_, max_val=r_max_, stdsigma=0.5)
            hopping, overlap = vmap(self.step,in_dims=1)(r)

            dftb_hopping = self.dftb(r, bond_indices = self.curr_bond_indices, mode="hopping").permute(1,0,2)
            dftb_overlap = self.dftb(r, bond_indices = self.curr_bond_indices, mode="overlap").permute(1,0,2)


            self.loss = (hopping - dftb_hopping).abs().mean() + \
                torch.nn.functional.mse_loss(hopping, dftb_hopping).sqrt() + \
                    15*torch.nn.functional.mse_loss(overlap, dftb_overlap).sqrt() + \
                        15*(overlap - dftb_overlap).abs().mean()
            self.loss.backward()
            return self.loss

        total_bond_types = len(self.idp_sk.bond_types)
    
        for istep in range(nstep):
            if istep % dis_freq == 0:
                print(f"step {istep}, loss {self.loss.item()}, lr {lrscheduler.get_last_lr()[0]}")
            # 如果 total_bond_types 太大, 会导致内存不够, 可以考虑分批次优化, 每次优化一部分的bond_types
            # 我们定义一次优化最大的bond_types数量为 max_elmt_batch^2    
            bond_indices_all = torch.randperm(total_bond_types)
            for i in range(0, total_bond_types, max_elmt_batch**2):
                curr_indices = torch.arange(i, min(i+max_elmt_batch**2, total_bond_types))
                self.curr_bond_indices = bond_indices_all[curr_indices]
                optimizer.step(closure)
            
            lrscheduler.step()
            self.symmetrize()
        if viz:
            self.viz(r_min=r_min, r_max=r_max)
        return True
    
    def to_nnsk(self, ebase=True):
        if ebase:
            nnsk = NNSK(
            idp_sk=self.dftb.idp_sk, 
            onsite={"method": "uniform"},
            hopping={"method": self.functype, "rs":self.rs, "w": self.w},
            overlap=True,
            )
        
            nnsk.hopping_param.data = self.hopping_params.data
            nnsk.overlap_param.data = self.overlap_params.data

            self.E_base = torch.zeros(self.idp_sk.num_types, self.idp_sk.n_onsite_Es)
            for asym, idx in self.idp_sk.chemical_symbol_to_type.items():
                self.E_base[idx] = torch.zeros(self.idp_sk.n_onsite_Es)
                for ot in self.idp_sk.basis[asym]:
                    fot = self.idp_sk.basis_to_full_basis[asym][ot]
                    self.E_base[idx][self.idp_sk.skonsite_maps[fot+"-"+fot]] = onsite_energy_database[asym][ot]
            
            nnsk.onsite_param.data = self.dftb.param["OnsiteE"] - self.E_base[torch.arange(len(self.idp_sk.type_names))].unsqueeze(-1)
        
        else:
            nnsk = NNSK(
            idp_sk=self.dftb.idp_sk, 
            onsite={"method": "uniform_noref"},
            hopping={"method": self.functype, "rs":self.rs, "w": self.w},
            overlap=True,
            )

            nnsk.hopping_param.data = self.hopping_params.data
            nnsk.overlap_param.data = self.overlap_params.data
            nnsk.onsite_param.data = self.dftb.param["OnsiteE"]

        return nnsk
    
    def to_json(self):
        nnsk = self.to_nnsk()
        return nnsk.to_json()
    
    @staticmethod
    def truncated_normal(shape, min_val, max_val, stdsigma=2):
        min_val = torch.as_tensor(min_val)
        max_val = torch.as_tensor(max_val)
        
        mean = (min_val + max_val) / 2
        #mean = (2 * min_val + mean) / 2
        std = (max_val - min_val) / (2 * stdsigma)
        u = torch.rand(shape)
        cdf_low = torch.erf((min_val - mean) / (std * 2.0**0.5)) / 2.0 + 0.5
        cdf_high = torch.erf((max_val - mean) / (std * 2.0**0.5)) / 2.0 + 0.5
        return torch.erfinv(2 * (cdf_low + u * (cdf_high - cdf_low)) - 1) * (2**0.5) * std + mean