from dptb.negf.RGF import recursive_gf
import logging
from dptb.utils.constants import eV
import torch
import os
from dptb.negf.utils import update_kmap, update_temp_file
from dptb.negf.density import Ozaki
from dptb.utils.constants import *
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from dptb.negf.utils import gauss_xw, leggauss

log = logging.getLogger(__name__)

class Device(object):
    def __init__(self, hamiltonian, structure, results_path, e_T=300, efermi=0.) -> None:
        self.green = 0
        self.hamiltonian = hamiltonian
        self.structure = structure # ase Atoms
        self.results_path = results_path
        self.cdtype = torch.complex128
        self.device = "cpu"
        self.kBT = k * e_T / eV
        self.e_T = e_T
        self.efermi = efermi
        self.mu = self.efermi
    
    def set_leadLR(self, lead_L, lead_R):
        self.lead_L = lead_L
        self.lead_R = lead_R
        self.mu = self.efermi - 0.5*(self.lead_L.voltage + self.lead_R.voltage)

    def green_function(self, e, kpoint, etaDevice=0., block_tridiagonal=True):
        assert len(np.array(kpoint).reshape(-1)) == 3
        if not isinstance(e, torch.Tensor):
            e = torch.tensor(e, dtype=torch.complex128)

        self.block_tridiagonal = block_tridiagonal
        self.kpoint = kpoint

        # if V is not None:
        #     HD_ = self.attachPotential(HD, SD, V)
        # else:
        #     HD_ = HD

        if os.path.exists(os.path.join(self.results_path, "POTENTIAL.pth")):
            self.V = torch.load(os.path.join(self.results_path, "POTENTIAL.pth"))
        elif abs(self.mu - self.efermi) < 1e-7:
            self.V = self.efermi - self.mu
        else:
            self.V = 0.
        
        if not hasattr(self, "hd") or not hasattr(self, "sd"):
            self.hd, self.sd, _, _, _, _ = self.hamiltonian.get_hs_device(kpoint, self.V, block_tridiagonal)
        s_in = [torch.zeros(i.shape).cdouble() for i in self.hd]
        
        # for i, e in tqdm(enumerate(ee), desc="Compute green functions: "):
        
        tags = ["g_trans", \
               "grd", "grl", "gru", "gr_left", \
               "gnd", "gnl", "gnu", "gin_left", \
               "gpd", "gpl", "gpu", "gip_left"]
        
        seL = self.lead_L.se
        seR = self.lead_R.se
        seinL = seL * self.lead_L.fermi_dirac(e+self.mu).reshape(-1)
        seinR = seR * self.lead_R.fermi_dirac(e+self.mu).reshape(-1)
        s01, s02 = s_in[0].shape
        se01, se02 = seL.shape
        idx0, idy0 = min(s01, se01), min(s02, se02)

        s11, s12 = s_in[-1].shape
        se11, se12 = seR.shape
        idx1, idy1 = min(s11, se11), min(s12, se12)
        
        green_ = {}

        s_in[0][:idx0,:idy0] = s_in[0][:idx0,:idy0] + seinL[:idx0,:idy0]
        s_in[-1][-idx1:,-idy1:] = s_in[-1][-idx1:,-idy1:] + seinR[-idx1:,-idy1:]
        ans = recursive_gf(e, hl=[], hd=self.hd, hu=[],
                            sd=self.sd, su=[], sl=[], 
                            left_se=seL, right_se=seR, seP=None, s_in=s_in,
                            s_out=None, eta=etaDevice, chemiPot=self.mu)
        s_in[0][:idx0,:idy0] = s_in[0][:idx0,:idy0] - seinL[:idx0,:idy0]
        s_in[-1][-idx1:,-idy1:] = s_in[-1][-idx1:,-idy1:] - seinR[-idx1:,-idy1:]
            # green shape [[g_trans, grd, grl,...],[g_trans, ...]]
        
        for t in range(len(tags)):
            green_[tags[t]] = ans[t]

        self.green = green_
        # self.green = update_temp_file(update_fn=fn, file_path=GFpath, ee=ee, tags=tags, info="Computing Green's Function")

    def _cal_current_(self, espacing):
        v_L = self.lead_L.voltage
        v_R = self.lead_R.voltage

        # check the energy grid satisfied the requirement
        xl = min(v_L, v_R)-4*self.kBT
        xu = max(v_L, v_R)+4*self.kBT

        def fcn(e):
            self.green_function()

        cc = leggauss(fcn=self._cal_tc_)
        
        int_grid, int_weight = gauss_xw(xl=xl, xu=xu, n=int((xu-xl)/espacing))

        self.__CURRENT__ = simpson((self.lead_L.fermi_dirac(self.ee+self.mu) - self.lead_R.fermi_dirac(self.ee+self.mu)) * self.tc, self.ee)

    def _cal_current_nscf_(self, ee, tc):
        f = lambda x,mu: 1 / (1 + torch.exp((x - mu) / self.kBT))

        emin = ee.min()
        emax = ee.max()
        vmin = emin + 4*self.kBT
        vmax = emax - 4*self.kBT
        vm = 0.5 * (vmin+vmax)
        vmid = vm - vmin
        
        vv = torch.linspace(start=0., end=vmid, steps=int(vmid / 0.1)+1) * 2
        cc = []

        for dv in vv * 0.5:
            I = simpson((f(ee+self.mu, self.lead_L.efermi-vm+dv) - f(ee+self.mu, self.lead_R.efermi-vm-dv)) * tc, ee)
            cc.append(I)

        return vv, cc
    
    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp((x - self.mu) / self.kBT))


    def _cal_tc_(self):

        tx, ty = self.g_trans.shape
        lx, ly = self.lead_L.se.shape
        rx, ry = self.lead_R.se.shape
        x0 = min(lx, tx)
        x1 = min(rx, ty)

        gammaL = torch.zeros(size=(tx, tx), dtype=self.cdtype, device=self.device)
        gammaL[:x0, :x0] += self.lead_L.gamma[:x0, :x0]
        gammaR = torch.zeros(size=(ty, ty), dtype=self.cdtype, device=self.device)
        gammaR[-x1:, -x1:] += self.lead_R.gamma[-x1:, -x1:]

        tc = torch.mm(torch.mm(gammaL, self.g_trans), torch.mm(gammaR, self.g_trans.conj().T)).diag().real.sum(-1)

        return tc
    
    def _cal_dos_(self):
        dos = 0
        for jj in range(len(self.grd)):
            temp = self.grd[jj] @ self.sd[jj] # taking each diagonal block with all energy e together
            dos -= temp.imag.diag().sum(-1) / pi

        return dos * 2

    def _cal_ldos_(self):
        ldos = []
        sd = self.hamiltonian.get_hs_device(kpoint=self.kpoint, V=self.V, block_tridiagonal=self.block_tridiagonal)[1]
        for jj in range(len(self.grd)):
            temp = self.grd[jj] @ sd[jj] # taking each diagonal block with all energy e together
            ldos.append(-temp.imag.diag() / pi) # shape(Nd(diagonal elements))

        return torch.cat(ldos, dim=0).contiguous()

    def _cal_local_current_(self):
        # current only support non-block-triagonal format
        v_L = self.lead_L.voltage
        v_R = self.lead_R.voltage

        # check the energy grid satisfied the requirement
        
        na = len(self.norbs_per_atom)
        local_current = torch.zeros(na, na)
        hd = self.hamiltonian.get_hs_device(kpoint=self.kpoint, V=self.V, block_tridiagonal=self.block_tridiagonal)[0][0]

        for i in range(na):
            for j in range(na):
                if i != j:
                    id = self.get_index(i)
                    jd = self.get_index(j)
                    ki = hd[id[0]:id[1], jd[0]:jd[1]] @ (1j*self.gnd[0][jd[0]:jd[1],id[0]:id[1]])
                    kj = hd[jd[0]:jd[1], id[0]:id[1]] @ (1j*self.gnd[0][id[0]:id[1],jd[0]:jd[1]])
                    local_current[i,j] = ki.real.diag().sum() - kj.real.diag().sum()
        
        return local_current.contiguous()
    
    def _cal_density_(self, dm_options):
        dm = Ozaki(**dm_options)
        DM_eq, DM_neq = dm.integrate(device=self.device, kpoint=self.kpoint)

        return DM_eq, DM_neq
    
    @property
    def current_nscf(self):
        if not hasattr(self, "__CURRENT_NSCF__"):
            self._cal_current_nscf_()
            return self.__V_NSCF__, self.__CURRENT_NSCF__
        
        else:
            return self.__V_NSCF__, self.__CURRENT_NSCF__


    @property
    def dos(self):
        return self._cal_dos_()
        
    @property
    def current(self):
        if not hasattr(self, "__CURRENT__"):
            self._cal_current_()
            return self.__CURRENT__
        
        else:
            return self.__CURRENT__
    
    @property
    def ldos(self):
        if not hasattr(self, "__LDOS__"):
            self._cal_ldos_()
            return self.__LDOS__
           
        else:
            return self.__LDOS__
            
    @property
    def tc(self):
        return self._cal_tc_()
        
    @property
    def lcurrent(self):
        if not hasattr(self, "__LCURRENT__"):
            self._cal_local_current_()

            return self.__LCURRENT__
        else:
            return self.__LCURRENT__


    @property
    def g_trans(self):
        return self.green["g_trans"] # [n,n]
    
    @property
    def grd(self):
        return self.green["grd"] # [[n,n]]
    
    @property
    def grl(self):
        return self.green["grl"]
    
    @property
    def gru(self):
        return self.green["gru"]
    
    @property
    def gr_left(self):
        return self.green["gr_left"]
    
    @property
    def gnd(self):
        return self.green["gnd"]
    
    @property
    def gnl(self):
        return self.green["gnl"]
    
    @property
    def gnu(self):
        return self.green["gnu"]
    
    @property
    def gin_left(self):
        return self.green["gin_left"]
    
    @property
    def gpd(self):
        return self.green["gpd"]
    
    @property
    def gpl(self):
        return self.green["gpl"]
    
    @property
    def gpu(self):
        return self.green["gpu"]
    
    @property
    def gip_left(self):
        return self.green["gip_left"]
    
    @property
    def norbs_per_atom(self):
        return self.hamiltonian.device_norbs

    @property
    def positions(self):
        return self.structure.positions
    
    def get_index(self, iatom):
        start = sum(self.norbs_per_atom[:iatom])
        end = start + self.norbs_per_atom[iatom]

        return [start, end]
    
    def get_index_block(self, iatom):
        pass