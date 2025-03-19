from dptb.nn.dftb.sk_param import SKParam
from dptb.nn.dftb.hopping_dftb import HoppingIntp
import torch
from torch import nn
from dptb.nn.sktb.hopping import HoppingFormula
from dptb.nn.sktb import OnsiteFormula, bond_length_list
from dptb.nn.sktb.cov_radiiDB import Covalent_radii
from dptb.nn.sktb.bondlengthDB import atomic_radius_v1
from dptb.utils.constants import atomic_num_dict
import matplotlib.pyplot as plt
from torch.optim import Adam, LBFGS, RMSprop, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from dptb.nn.nnsk import NNSK
from dptb.nn.sktb.onsite import onsite_energy_database
from dptb.data.AtomicData import get_r_map, get_r_map_bondwise
import numpy as np
from typing import Union
import logging
import os, sys
from dptb.utils.tools import get_lr_scheduler, get_optimizer, setup_seed

log = logging.getLogger(__name__)

class dftb:
    def __init__(self, basis, skdata, cal_rcuts=False, device='cpu', dtype=torch.float32):
        self.device = device
        if isinstance(dtype, str):
            dtype =  getattr(torch, dtype)
        self.dtype = dtype
        self.param = SKParam(basis=basis, skdata=skdata, cal_rcuts=cal_rcuts)
        self.bond_r_min = self.param.bond_r_min
        self.bond_r_max = self.param.bond_r_max
        self.idp_sk = self.param.idp_sk

        self.param = self.param.format_skparams(self.param.skdict)
        self.hopping = HoppingIntp(num_ingrls=self.param["Hopping"].shape[1])
        self.overlap = HoppingIntp(num_ingrls=self.param["Overlap"].shape[1])
        self.bond_types = self.idp_sk.bond_types
        self.bond_type_to_index = {bt: i for i, bt in enumerate(self.idp_sk.bond_types)}

    def __call__(self, r, bond_indices = None, mode="hopping"):
        out = []
        if bond_indices is None:
            bond_indices = torch.arange(len(self.idp_sk.bond_types), device=self.device)

        assert len(bond_indices) == len(r), "The bond_indices and r should have the same length."
        
        for i, ind in enumerate(bond_indices):
            out.append(self.hopping.get_skhij(rij=r[i], xx=self.param["Distance"].to(device=self.device, dtype=self.dtype), 
                                              yy=self.param[mode[0].upper()+mode[1:]][ind].to(device=self.device, dtype=self.dtype)))
        
        return torch.stack(out)
    
class DFTB2NNSK(nn.Module):

    def __init__(self, basis, skdata, train_options, output='./', method='poly2pow', rs=None, w=0.2, cal_rcuts=False, atomic_radius='cov',seed=3982377700, device='cpu', dtype=torch.float32):
        if rs is None:
            assert cal_rcuts, "If rs is not provided, cal_rcuts should be False."
        super(DFTB2NNSK, self).__init__()
        self.device = device
        if isinstance(dtype, str):
            dtype =  getattr(torch, dtype)
        self.dtype = dtype
        
        self.dftb = dftb(basis=basis, skdata=skdata, cal_rcuts=cal_rcuts, device=self.device, dtype=self.dtype)
        self.basis = basis
        self.functype = method
        self.idp_sk = self.dftb.idp_sk
        # self.rs = rs
        self.w = w         
        self.nnsk_hopping = HoppingFormula(functype=self.functype)
        self.nnsk_overlap = HoppingFormula(functype=self.functype, overlap=True)
        self.hopping_params = torch.nn.Parameter(torch.randn(len(self.idp_sk.bond_types), self.dftb.hopping.num_ingrls, self.nnsk_hopping.num_paras, device=self.device, dtype=self.dtype))
        self.overlap_params = torch.nn.Parameter(torch.randn(len(self.idp_sk.bond_types), self.dftb.hopping.num_ingrls, self.nnsk_hopping.num_paras, device=self.device, dtype=self.dtype))
        self.atomic_radius = atomic_radius
        self.initialize_atomic_radius(basis, atomic_radius)
        self.initialize_rs_and_cutoffs(rs, cal_rcuts)

        self.mae_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.best_loss = float("inf")
        self.train_options = train_options
        self.output = output
        self.seed = seed 
        setup_seed(seed)
       
    def initialize_atomic_radius(self, basis, atomic_radius):
        if isinstance(atomic_radius, str):
            if atomic_radius == 'cov':
                atomic_radius_dict = Covalent_radii
            elif atomic_radius == 'v1':
                atomic_radius_dict = atomic_radius_v1
            else:
                raise ValueError("The atomic_radius should be either str of 'cov' or 'v1' or a dict.")
        else:
            assert isinstance(atomic_radius, dict), "The atomic_radius should be either str of 'cov' or 'v1' or a dict."
            atomic_radius_dict = atomic_radius

        atomic_numbers = [atomic_num_dict[key] for key in basis.keys()]
        self.atomic_radius_list =  torch.zeros(int(max(atomic_numbers)),device= self.device, dtype=self.dtype) - 100
        for at in basis.keys():
            assert at in atomic_radius_dict and atomic_radius_dict[at] is not None, f"The atomic radius for {at} is not provided."
            radii = atomic_radius_dict[at]
            self.atomic_radius_list[atomic_num_dict[at]-1] = radii

    def initialize_rs_and_cutoffs(self, rs, cal_rcuts):
        if not cal_rcuts:
            assert isinstance(rs, (float,int)), "If cal_rcuts is False, the rs should be a float"
            self.rs = rs
            self.r_max = None
            self.r_min = None 
        else:
            if rs is None:
                self.rs = self.dftb.bond_r_max
            else:
                assert isinstance(rs, dict)
                for k, v in self.dftb.bond_r_max.items():
                    assert k in rs, f"The bond type {k} is not in the rs dict."
                    assert rs[k] == v, f"The bond type rmax in {k} is not equal to the dftb bond_r_max."
                self.rs = rs    

            self.r_map = get_r_map_bondwise(self.dftb.bond_r_max).to(device=self.device, dtype=self.dtype)
            self.r_max = []
            self.r_min = []
            for ibt in self.idp_sk.bond_types:
                self.r_max.append(self.dftb.bond_r_max[ibt])
                self.r_min.append(self.dftb.bond_r_min[ibt])
            self.r_max = torch.tensor(self.r_max, device=self.device, dtype=self.dtype).reshape(-1,1)
            self.r_min = torch.tensor(self.r_min, device=self.device, dtype=self.dtype).reshape(-1,1)


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
    
    def save(self,filepath):
        state = {}
        config = {
            'basis': self.basis,
            'method': self.functype,
            'rs': self.rs,
            'w': self.w,
            'cal_rcuts': self.r_max is not None,
            'atomic_radius': self.atomic_radius,
            'device': self.device,
            'dtype': self.dtype,
            'seed': self.seed
        }
        state.update({"config": config})
        state.update({"model_state_dict": self.state_dict()})

        torch.save(state, f"{filepath}")
        log.info(f"The model is saved to {filepath}")

    def get_config(self):
        config = {
            'basis': self.basis,
            'method': self.functype,
            'rs': self.rs,
            'w': self.w,
            'cal_rcuts': self.r_max is not None,
            'atomic_radius': self.atomic_radius,
            'device': self.device,
            'dtype': self.dtype,
            'seed': self.seed
        }
        return config

    @classmethod
    def load(cls, ckpt, skdata, train_options, output='./'):
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"No file found at {ckpt}")

        state = torch.load(ckpt, weights_only=False)
        config = state['config']
        model = cls(skdata=skdata, train_options=train_options, output=output, **config)
        model.load_state_dict(state['model_state_dict'])
        return model 
    

    def step(self, r):
        assert r.shape[0] == len(self.curr_bond_indices)
        r = r.reshape(-1)
        bond_ind_r_shp = self.curr_bond_indices.reshape(-1)

        edge_number = self.idp_sk.untransform_bond(bond_ind_r_shp).T
        r0 = self.atomic_radius_list[edge_number-1].sum(0).to(device=self.device, dtype=self.dtype)  # bond length r0 = r1 + r2. (r1, r2 are atomic radii of the two atoms)

        if isinstance(self.rs, dict):
            assert hasattr(self, "r_map")
            # r_cutoffs = self.r_map[edge_number-1].sum(0)
            r_cutoffs = self.r_map[edge_number[0]-1, edge_number[1]-1]
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

    def forward(self, r, bond_indices):
        self.curr_bond_indices = bond_indices
        hopping, overlap = torch.func.vmap(self.step,in_dims=1)(r)

        dftb_hopping = self.dftb(r, bond_indices = self.curr_bond_indices, mode="hopping").permute(1,0,2)
        dftb_overlap = self.dftb(r, bond_indices = self.curr_bond_indices, mode="overlap").permute(1,0,2)

        return hopping, overlap, dftb_hopping, dftb_overlap
    
            
    def optimize(self, r_min=None, r_max=None, nstep=None, nsample=None, lr=None, dis_freq=None, viz=False):
        """
        Optimize the parameters of the neural network model.

        Args:
            r_min (float): The minimum value for the random range of r.
            r_max (float): The maximum value for the random range of r.
            nsample (int): The number of samples to generate for r.
            nstep (int): The number of optimization steps to perform.
            lr (float): The learning rate for the optimizer.
            Freq (int): The frequency using in the function :
                        1. frequency to display the loss during optimization.
                        2. frequency to save the model.
                        3. for CosineAnnealingLR, it is the T_max = 5*Freq.
            method (str): The optimization method to use. Supported methods are "RMSprop" and "LBFGS".
            viz (bool): Whether to visualize the optimized results.
            max_elmt_batch (int): max_elmt_batch^2 defines The maximum number of bond types to optimize in each batch.
             ie. if max_elmt_batch=4, we will optimize 16 bond types in each batch.

        Returns:
            bool: True if the optimization is successful.

        Raises:
            NotImplementedError: If the specified optimization method is not supported.
        """    
        if lr is not None:
            self.train_options["optimizer"]["lr"] = lr
        max_elmt_batch = self.train_options.get("max_elmt_batch", 4)
        if nstep is None:
            nstep = int(self.train_options["nstep"])
        if dis_freq is None:
            dis_freq = int(self.train_options["dis_freq"])
        if nsample is None:
            nsample = int(self.train_options.get("n_samples",256))
        
        save_freq = self.train_options.get("save_freq", 1)

        optimizer = get_optimizer(model_param=[self.hopping_params, self.overlap_params], **self.train_options["optimizer"])
        lrscheduler = get_lr_scheduler(optimizer=optimizer, **self.train_options["lr_scheduler"])  # add optmizer

        self.loss = torch.tensor(0., device=self.device, dtype=self.dtype)
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
            
            # 用 gauss 分布的随机数，重点采样在 r_min 和 r_max范围中心区域的值
            r = self.truncated_normal(shape=[len(self.curr_bond_indices),nsample], min_val=r_min_, max_val=r_max_, stdsigma=0.5, device=self.device, dtype=self.dtype)
           
            hopping, overlap, dftb_hopping, dftb_overlap = self(r, bond_indices=self.curr_bond_indices)

            # self.loss = (hopping - dftb_hopping).abs().mean() + \
            #    torch.nn.functional.mse_loss(hopping, dftb_hopping).sqrt() + \
            #        15*torch.nn.functional.mse_loss(overlap, dftb_overlap).sqrt() + \
            #            15*(overlap - dftb_overlap).abs().mean()
            
            self.loss_hop_mae =  self.mae_loss(hopping, dftb_hopping)
            self.loss_hop_rmse = torch.sqrt(self.mse_loss(hopping, dftb_hopping))
            self.loss_ovl_mae = self.mae_loss(overlap, dftb_overlap)
            self.loss_ovl_rmse = torch.sqrt(self.mse_loss(overlap, dftb_overlap))

            self.loss = self.loss_hop_mae + self.loss_hop_rmse + 15*self.loss_ovl_mae + 15*self.loss_ovl_rmse

            self.loss.backward()
            return self.loss

        total_bond_types = len(self.idp_sk.bond_types)
    
        for istep in range(nstep):
            # 如果 total_bond_types 太大, 会导致内存不够, 可以考虑分批次优化, 每次优化一部分的bond_types
            # 我们定义一次优化最大的bond_types数量为 max_elmt_batch^2    
            bond_indices_all = torch.randperm(total_bond_types, device=self.device)
            total_loss = 0
            total_hop_mae = 0
            total_hop_rmse = 0
            total_ovl_mae = 0
            total_ovl_rmse = 0

            for i in range(0, total_bond_types, max_elmt_batch**2):
                curr_indices = torch.arange(i, min(i+max_elmt_batch**2, total_bond_types),device=self.device)
                self.curr_bond_indices = bond_indices_all[curr_indices]
                optimizer.step(closure)
                total_loss += self.loss.item()
                total_hop_mae += self.loss_hop_mae.item()
                total_hop_rmse += self.loss_hop_rmse.item()
                total_ovl_mae += self.loss_ovl_mae.item()
                total_ovl_rmse += self.loss_ovl_rmse.item()

                if istep % dis_freq == 0:
                    loginfo = (f"Batch {istep:6d}, subset [{i:3d}{min(i+max_elmt_batch**2, total_bond_types):3d}]: "
                             f"Loss {self.loss.item():7.4f}, "
                             f"Hop MAE {self.loss_hop_mae.item():7.4f}, "
                             f"Hop RMSE {self.loss_hop_rmse.item():7.4f}, "
                             f"Ovl MAE {self.loss_ovl_mae.item():7.4f}, "
                             f"Ovl RMSE {self.loss_ovl_rmse.item():7.4f}, "
                             f"LR {lrscheduler.get_last_lr()[0]:8.6f}")
                    log.info(loginfo)
                        
            if istep % dis_freq == 0 and total_bond_types > max_elmt_batch**2:
                total_loss = total_loss / ((total_bond_types + max_elmt_batch**2 - 1) // max_elmt_batch**2)
                total_hop_mae = total_hop_mae / ((total_bond_types + max_elmt_batch**2 - 1) // max_elmt_batch**2)
                total_hop_rmse = total_hop_rmse / ((total_bond_types + max_elmt_batch**2 - 1) // max_elmt_batch**2)
                total_ovl_mae = total_ovl_mae / ((total_bond_types + max_elmt_batch**2 - 1) // max_elmt_batch**2)
                total_ovl_rmse = total_ovl_rmse / ((total_bond_types + max_elmt_batch**2 - 1) // max_elmt_batch**2)

                loginfo=(f"Batch {istep} Summary: "
                         f"Loss {total_loss:.4f}, "
                         f"Hop MAE {total_hop_mae:.4f}, "
                         f"Hop RMSE {total_hop_rmse:.4f}, "
                         f"Ovl MAE {total_ovl_mae:.4f}, "
                         f"Ovl RMSE {total_ovl_rmse:.4f}, "
                         f"LR {lrscheduler.get_last_lr()[0]:.6f}")

                log.info('--'*15)
                log.info(loginfo)

            lrscheduler.step()
            self.symmetrize()

            if total_loss < min(self.best_loss, 1):
                self.best_loss = total_loss
                self.save(f'{self.output}/best_df2sk.pth')
            if istep % save_freq == 0 or istep == nstep-1:
                self.save(f'{self.output}/lastest_df2sk.pth')
        if viz:
            self.viz(r_min=r_min, r_max=r_max)
        return True
    
        
    def viz(self, atom_a:str, atom_b:str=None, show_int=True, r_min:Union[float, int]=None, r_max:Union[float, int]=None, nsample=100):
        with torch.no_grad():
            if atom_b is None:
                atom_b = atom_a
            bond_type = f"{atom_a}-{atom_b}"
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

            if not show_int:
                # hops[0].shape[1] == 1, since we only plot one bond type.
                for i in range(hops[0].shape[1]):
                    plt.plot(r[i], hops[0][:,i, :-1].detach().numpy(), c="C"+str(i))
                    plt.plot(r[i], hops[0][:,i, -1].detach().numpy(), c="C"+str(i),label="nn:"+bond_type)
                    plt.plot(r[i], dftb_hopping[:,i, :-1].numpy(), c="C"+str(i), linestyle="--")
                    plt.plot(r[i], dftb_hopping[:,i, -1].numpy(), c="C"+str(i), linestyle="--",label="skf:"+bond_type)
                plt.title("hoppings")
                plt.xlabel("r(angstrom)")
                plt.tight_layout()
                plt.legend()
                plt.savefig(f"{self.output}/hopping_{bond_type}.png")
                plt.show()

                fig = plt.figure(figsize=(6,4))
                for i in range(hops[1].shape[1]):
                    plt.plot(r[i], hops[1][:,i, :-1].detach().numpy(), c="C"+str(i))
                    plt.plot(r[i], hops[1][:,i, -1].detach().numpy(), c="C"+str(i),label="nn:"+bond_type)
                    plt.plot(r[i], dftb_overlap[:,i, :-1].numpy(), c="C"+str(i), linestyle="--")
                    plt.plot(r[i], dftb_overlap[:,i, -1].numpy(), c="C"+str(i), linestyle="--",label="skf:"+bond_type)
                plt.title("overlaps")
                plt.xlabel("r(angstrom)")
                plt.tight_layout()
                plt.legend()
                plt.savefig(f"{self.output}/overlap_{bond_type}.png")
                plt.show()
            else:
                assert hops[0].shape[1] ==1
                hopps = hops[0][:,0,:].detach().numpy()
                dftb_hopps = dftb_hopping[:,0,:].numpy()
                fig = plt.figure(figsize=(6,4))
                ic=-1
                for k, v in self.idp_sk.orbpairtype_maps.items():
                    hopps_ibt = hopps[:, v]
                    dftb_hopps_ibt = dftb_hopps[:, v]
                    for ii in range(hopps_ibt.shape[1]):
                        ic+=1
                        plt.plot(r[0], hopps_ibt[:,ii], c="C"+str(ic), label=f"nn:{k}-{ii}")
                        plt.plot(r[0], dftb_hopps_ibt[:,ii], c="C"+str(ic), linestyle="--")

                plt.title("hoppings")
                plt.xlabel("r(angstrom)")
                plt.tight_layout()
                plt.legend(ncol=2)
                plt.savefig(f"{self.output}/hopping_{bond_type}.png")
                plt.show()

                ovlps = hops[1][:,0,:].detach().numpy()
                dftb_ovlps = dftb_overlap[:,0,:].numpy()
                fig = plt.figure(figsize=(6,4))
                ic=-1
                for k, v in self.idp_sk.orbpairtype_maps.items():
                    ovlps_ibt = ovlps[:, v]
                    dftb_ovlps_ibt = dftb_ovlps[:, v]
                    for ii in range(ovlps_ibt.shape[1]):
                        ic+=1
                        plt.plot(r[0], ovlps_ibt[:,ii], c="C"+str(ic), label=f"nn:{k}-{ii}")
                        plt.plot(r[0], dftb_ovlps_ibt[:,ii], c="C"+str(ic), linestyle="--")
                
                plt.title("overlaps")
                plt.xlabel("r(angstrom)")
                plt.tight_layout()
                plt.legend(ncol=2)
                plt.savefig(f"{self.output}/overlap_{bond_type}.png")
                plt.show()
        return True


    def to_nnsk(self, ebase=False):
        if ebase:
            self.nnsk = NNSK(
            idp_sk=self.dftb.idp_sk, 
            onsite={"method": "uniform"},
            hopping={"method": self.functype, "rs":self.rs, "w": self.w},
            overlap=True,
            atomic_radius = self.atomic_radius,
            device=self.device,
            dtype=self.dtype
            )
        
            self.nnsk.hopping_param.data = self.hopping_params.data
            self.nnsk.overlap_param.data = self.overlap_params.data

            self.E_base = torch.zeros(self.idp_sk.num_types, self.idp_sk.n_onsite_Es)
            for asym, idx in self.idp_sk.chemical_symbol_to_type.items():
                self.E_base[idx] = torch.zeros(self.idp_sk.n_onsite_Es)
                for ot in self.idp_sk.basis[asym]:
                    fot = self.idp_sk.basis_to_full_basis[asym][ot]
                    self.E_base[idx][self.idp_sk.skonsite_maps[fot+"-"+fot]] = onsite_energy_database[asym][ot]
            
            self.nnsk.onsite_param.data = self.dftb.param["OnsiteE"] - self.E_base[torch.arange(len(self.idp_sk.type_names))].unsqueeze(-1)
        
        else:
            self.nnsk = NNSK(
            idp_sk=self.dftb.idp_sk, 
            onsite={"method": "uniform_noref"},
            hopping={"method": self.functype, "rs":self.rs, "w": self.w},
            overlap=True,
            atomic_radius = self.atomic_radius,
            device=self.device,
            dtype=self.dtype
            )

            self.nnsk.hopping_param.data = self.hopping_params.data
            self.nnsk.overlap_param.data = self.overlap_params.data
            self.nnsk.onsite_param.data = self.dftb.param["OnsiteE"]

        return self.nnsk

    def to_pth(self):
        if not hasattr(self, "nnsk"):
            self.to_nnsk()
        self.nnsk.save(f"{self.output}/nnsk_from_skf.pth")        
        
        return True 
        
    def to_json(self):
        if not hasattr(self, "nnsk"):
            self.to_nnsk()
        return self.nnsk.to_json()
    
    @staticmethod
    def truncated_normal(shape, min_val, max_val, stdsigma=2,device='cpu', dtype=torch.float32):
        min_val = torch.as_tensor(min_val, device=device, dtype=dtype)
        max_val = torch.as_tensor(max_val, device=device, dtype=dtype)
        
        mean = (min_val + max_val) / 2
        #mean = (2 * min_val + mean) / 2
        std = (max_val - min_val) / (2 * stdsigma)
        u = torch.rand(shape, device=device, dtype=dtype)
        cdf_low = torch.erf((min_val - mean) / (std * 2.0**0.5)) / 2.0 + 0.5
        cdf_high = torch.erf((max_val - mean) / (std * 2.0**0.5)) / 2.0 + 0.5
        return torch.erfinv(2 * (cdf_low + u * (cdf_high - cdf_low)) - 1) * (2**0.5) * std + mean