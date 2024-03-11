import numpy as np 
import pyamg #TODO: later add it to optional dependencies,like sisl
from pyamg.gallery import poisson
from dptb.utils.constants import elementary_charge
from dptb.utils.constants import Boltzmann, eV2J
from scipy.constants import epsilon_0 as eps0  #TODO:later add to untils.constants.py
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
#eps0 = 8.854187817e-12 # in the unit of F/m
# As length in deeptb is in the unit of Angstrom, the unit of eps0 is F/Angstrom
eps0 = eps0*1e-10 # in the unit of F/Angstrom


class Grid(object):
    # define the grid in 3D space
    def __init__(self,xg,yg,zg,xa,ya,za):
        # xg,yg,zg are the coordinates of the basic grid points
        # self.xg = np.around(xg,decimals=5);self.yg = np.around(yg,decimals=5);self.zg = np.around(zg,decimals=5)
        self.xg = xg
        self.yg = yg
        self.zg = zg
        # xa,ya,za are the coordinates of the atoms
        # atom should be within the grid
        assert np.min(xa) >= np.min(xg) and np.max(xa) <= np.max(xg)
        assert np.min(ya) >= np.min(yg) and np.max(ya) <= np.max(yg)
        assert np.min(za) >= np.min(zg) and np.max(za) <= np.max(zg)

        self.Na = len(xa) # number of atoms
        uxa = np.unique(xa);uya = np.unique(ya);uza = np.unique(za)
        # x,y,z are the coordinates of the grid points
        self.xall = np.unique(np.concatenate((uxa,self.xg),0)) # unique results are sorted
        self.yall = np.unique(np.concatenate((uya,self.yg),0))
        self.zall = np.unique(np.concatenate((uza,self.zg),0))
        self.shape = (len(self.xall),len(self.yall),len(self.zall))

        print('unique len of zall:',len(np.unique(self.zall)))

        # create meshgrid
        xmesh,ymesh,zmesh = np.meshgrid(self.xall,self.yall,self.zall)
        xmesh = xmesh.flatten()
        ymesh = ymesh.flatten()
        zmesh = zmesh.flatten()
        self.grid_coord = np.array([xmesh,ymesh,zmesh]).T #(Np,3)
        sorted_indices = np.lexsort((xmesh,ymesh,zmesh))
        self.grid_coord = self.grid_coord[sorted_indices] # sort the grid points firstly along x, then y, lastly z        
        ## check the number of grid points
        self.Np = int(len(self.xall)*len(self.yall)*len(self.zall))
        assert self.Np == len(xmesh)
        assert self.grid_coord.shape[0] == self.Np

        print('Number of grid points: ',self.Np,' grid shape: ',self.grid_coord.shape,' Number of atoms: ',self.Na)

        # find the index of the atoms in the grid
        self.atom_index_dict = self.find_atom_index(xa,ya,za)


        # create surface area for each grid point along x,y,z axis
        # each grid point corresponds to a Voronoi cell(box)
        surface_grid = np.zeros((self.Np,3))
        x_vorlen = self.cal_vorlen(self.xall);y_vorlen = self.cal_vorlen(self.yall);z_vorlen = self.cal_vorlen(self.zall)
        
        XD,YD = np.meshgrid(x_vorlen,y_vorlen)
        ## surface along x-axis (yz-plane)
        ax,bx = np.meshgrid(YD.flatten(),z_vorlen)
        surface_grid[:,0] = abs((ax*bx).flatten())
        ## surface along y-axis (xz-plane) 
        ay,by = np.meshgrid(XD.flatten(),z_vorlen)
        surface_grid[:,1] = abs((ay*by).flatten())
        ## surface along z-axis (xy-plane)
        az,_ = np.meshgrid((XD*YD).flatten(),self.zall)
        surface_grid[:,2] = abs(az.flatten())

        self.surface_grid = surface_grid  # grid points order are the same as that of  self.grid_coord
        



    def find_atom_index(self,xa,ya,za):
        # find the index of the atoms in the grid
        swap = {}
        for atom_index in range(self.Na):
            for gp_index in range(self.Np):
                if xa[atom_index]==self.grid_coord[gp_index][0] and \
                    ya[atom_index]==self.grid_coord[gp_index][1] and \
                    za[atom_index]==self.grid_coord[gp_index][2]:
                    swap.update({atom_index:gp_index})
        return swap
    
    def cal_vorlen(self,x):
        # compute the length of the Voronoi segment of a one-dimensional array x
        xd = np.zeros(len(x))
        xd[0] = abs(x[0]-x[1])/2
        xd[-1] = abs(x[-1]-x[-2])/2
        for i in range(1,len(x)-1):
            xd[i] = (abs(x[i]-x[i-1])+abs(x[i]-x[i+1]))/2
        return xd


class region(object):
    def __init__(self,x_range,y_range,z_range):
        self.xmin,self.xmax = float(x_range[0]),float(x_range[1])
        self.ymin,self.ymax = float(y_range[0]),float(y_range[1])
        self.zmin,self.zmax = float(z_range[0]),float(z_range[1])
    
class Gate(region):
    def __init__(self,x_range,y_range,z_range):
        # Gate region
        super().__init__(x_range,y_range,z_range)
        # Fermi_level of gate (in unit eV)
        self.Ef = 0.0        
        # self.xmin,self.xmax = float(xrange[0]),float(xrange[1])
        # self.ymin,self.ymax = float(yrange[0]),float(yrange[1])
        # self.zmin,self.zmax = float(zrange[0]),float(zrange[1])

class Dielectric(region):
    def __init__(self,x_range,y_range,z_range):
        # dielectric region
        super().__init__(x_range,y_range,z_range)
        # dielectric permittivity
        self.eps = 1.0
        # # gate region
        # self.xmin,self.xmax = float(xrange[0]),float(xrange[1])
        # self.ymin,self.ymax = float(yrange[0]),float(yrange[1])
        # self.zmin,self.zmax = float(zrange[0]),float(zrange[1])


class Interface3D(object):
    def __init__(self,grid,gate_list,dielectric_list):
        assert grid.__class__.__name__ == 'Grid'

        
        for i in range(0,len(gate_list)):
            if not gate_list[i].__class__.__name__ == 'Gate':
                raise ValueError('Unknown region type in Gate list: ',gate_list[i].__class__.__name__)
        for i in range(0,len(dielectric_list)):
            if not dielectric_list[i].__class__.__name__ == 'Dielectric':
                raise ValueError('Unknown region type in Dielectric list: ',dielectric_list[i].__class__.__name__)
            
        self.grid = grid
        self.eps = np.ones(grid.Np) # dielectric permittivity
        self.phi,self.phi_old = np.zeros(grid.Np),np.zeros(grid.Np) # potential
        self.free_charge,self.fixed_charge  = np.zeros(grid.Np),np.zeros(grid.Np)  # free charge density and fixed charge density 

        self.Temperature = 300.0 # temperature in unit of Kelvin
        self.kBT = Boltzmann*self.Temperature/eV2J # thermal energy in unit of eV

        # store the boundary information: xmin,xmax,ymin,ymax,zmin,zmax,gate
        self.boudnary_points = {i:"in" for i in range(self.grid.Np)} # initially set all points as internal
        self.boundary_points_get()

        self.lead_gate_potential = np.zeros(grid.Np) # no gate potential initially, all grid points are set to zero
        self.potential_eps_get(gate_list+dielectric_list)
        


    def boundary_points_get(self):
        # set the boundary points
        xmin,xmax = np.min(self.grid.xall),np.max(self.grid.xall)
        ymin,ymax = np.min(self.grid.yall),np.max(self.grid.yall)
        zmin,zmax = np.min(self.grid.zall),np.max(self.grid.zall)
        internal_NP = 0
        for i in range(self.grid.Np):
            if self.grid.grid_coord[i,0] == xmin: self.boudnary_points[i] = "xmin"
            elif self.grid.grid_coord[i,0] == xmax: self.boudnary_points[i] = "xmax"
            elif self.grid.grid_coord[i,1] == ymin: self.boudnary_points[i] = "ymin"   
            elif self.grid.grid_coord[i,1] == ymax: self.boudnary_points[i] = "ymax" 
            elif self.grid.grid_coord[i,2] == zmin: self.boudnary_points[i] = "zmin"  
            elif self.grid.grid_coord[i,2] == zmax: self.boudnary_points[i] = "zmax"
            else: internal_NP +=1
                
        # internal_NP = 0
        # for i in range(self.grid.Np):
        #     if self.boudnary_points[i] == "in":
        #         internal_NP += 1
        self.internal_NP = internal_NP
    
    def potential_eps_get(self,region_list):
        # set the gate potential
        # ingore the lead potential temporarily
        gate_point = 0
        for i in range(len(region_list)):    
            # find gate region in grid
            index=np.nonzero((region_list[i].xmin<=self.grid.grid_coord[:,0])&
                             (region_list[i].xmax>=self.grid.grid_coord[:,0])&
                        (region_list[i].ymin<=self.grid.grid_coord[:,1])&
                        (region_list[i].ymax>=self.grid.grid_coord[:,1])&
                        (region_list[i].zmin<=self.grid.grid_coord[:,2])&
                        (region_list[i].zmax>=self.grid.grid_coord[:,2]))[0]
            if region_list[i].__class__.__name__ == 'Gate': 
                #attribute gate potential to the corresponding grid points
                self.boudnary_points.update({index[i]: "Gate" for i in range(len(index))})
                self.lead_gate_potential[index] = region_list[i].Ef 
                gate_point += len(index)
            elif region_list[i].__class__.__name__ == 'Dielectric':
                # attribute dielectric permittivity to the corresponding grid points
                self.eps[index] = region_list[i].eps
            else:
                raise ValueError('Unknown region type: ',region_list[i].__class__.__name__)
        print('Number of gate points: ',gate_point)
        
    def to_pyamg(self,dtype=np.float64):
        # convert to amg format A,b matrix
        # if dtype == None:
        #     dtype = np.float64
        # A = poisson(self.grid.shape,format='csr',dtype=dtype)
        Jacobian = csr_matrix(np.zeros((self.grid.Np,self.grid.Np),dtype=dtype))
        B = np.zeros(Jacobian.shape[0],dtype=Jacobian.dtype)
        self.NR_construct_Jacobian(Jacobian)
        self.NR_construct_B(B)
        
        return Jacobian,B
    
    
    def to_scipy(self,dtype=np.float64):
        # convert to amg format A,b matrix
        # A = poisson(self.grid.shape,format='csr',dtype=dtype)
        Jacobian = csr_matrix(np.zeros((self.grid.Np,self.grid.Np),dtype=dtype))
        B = np.zeros(Jacobian.shape[0],dtype=Jacobian.dtype)
        self.NR_construct_Jacobian(Jacobian)
        self.NR_construct_B(B)
        # self.construct_poisson(A,b)
        return Jacobian,B
    
    def construct_poisson(self,A,b):
        # construct the Poisson equation by adding boundary conditions and free charge to the matrix A and vector b
        Nx = self.grid.shape[0];Ny = self.grid.shape[1];Nz = self.grid.shape[2]
        for gp_index in range(self.grid.Np):
            if self.boudnary_points[gp_index] == "in":
                # flux_xm = self.grid.surface_grid[gp_index,0]*eps0*(self.eps[gp_index-1]+self.eps[gp_index])*0.5\
                # *(self.phi[gp_index-1]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index,0]-self.grid.grid_coord[gp_index-1,0])
                # flux_xp = self.grid.surface_grid[gp_index,0]*eps0*(self.eps[gp_index+1]+self.eps[gp_index])*0.5\
                # *(self.phi[gp_index+1]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index+1,0]-self.grid.grid_coord[gp_index,0])
                
                # flux_ym = self.grid.surface_grid[gp_index,1]*eps0*(self.eps[gp_index-Nx]+self.eps[gp_index])*0.5\
                # *(self.phi[gp_index-Nx]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index-Nx,1]-self.grid.grid_coord[gp_index,1])
                # flux_yp = self.grid.surface_grid[gp_index,1]*eps0*(self.eps[gp_index+Nx]+self.eps[gp_index])*0.5\
                # *(self.phi[gp_index+Nx]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index+Nx,1]-self.grid.grid_coord[gp_index,1])

                # flux_zm = self.grid.surface_grid[gp_index,2]*eps0*(self.eps[gp_index-Nx*Ny]+self.eps[gp_index])*0.5\
                # *(self.phi[gp_index-Nx*Ny]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index-Nx*Ny,2]-self.grid.grid_coord[gp_index,2])
                # flux_zp = self.grid.surface_grid[gp_index,2]*eps0*(self.eps[gp_index+Nx*Ny]+self.eps[gp_index])*0.5\
                # *(self.phi[gp_index+Nx*Ny]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index+Nx*Ny,2]-self.grid.grid_coord[gp_index,2])
                flux_xm = self.grid.surface_grid[gp_index,0]*eps0*(self.eps[gp_index-1]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index,0]-self.grid.grid_coord[gp_index-1,0])
                flux_xp = self.grid.surface_grid[gp_index,0]*eps0*(self.eps[gp_index+1]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index+1,0]-self.grid.grid_coord[gp_index,0])
                
                flux_ym = self.grid.surface_grid[gp_index,1]*eps0*(self.eps[gp_index-Nx]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index-Nx,1]-self.grid.grid_coord[gp_index,1])
                flux_yp = self.grid.surface_grid[gp_index,1]*eps0*(self.eps[gp_index+Nx]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index+Nx,1]-self.grid.grid_coord[gp_index,1])

                flux_zm = self.grid.surface_grid[gp_index,2]*eps0*(self.eps[gp_index-Nx*Ny]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index-Nx*Ny,2]-self.grid.grid_coord[gp_index,2])
                flux_zp = self.grid.surface_grid[gp_index,2]*eps0*(self.eps[gp_index+Nx*Ny]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index+Nx*Ny,2]-self.grid.grid_coord[gp_index,2])

                # add flux term to matrix A
                A[gp_index,gp_index] = -(flux_xm+flux_xp+flux_ym+flux_yp+flux_zm+flux_zp)
                A[gp_index,gp_index-1] = flux_xm
                A[gp_index,gp_index+1] = flux_xp
                A[gp_index,gp_index-Nx] = flux_ym
                A[gp_index,gp_index+Nx] = flux_yp
                A[gp_index,gp_index-Nx*Ny] = flux_zm
                A[gp_index,gp_index+Nx*Ny] = flux_zp

                # b[gp_index] = -elementary_charge*self.free_charge[gp_index]\
                #     *np.exp(-np.sign(self.free_charge[gp_index])*(self.phi[gp_index]-self.phi_old[gp_index])/self.kBT)\
                #     -elementary_charge*self.fixed_charge[gp_index]

                # TODO: remove damping factor temporarily
                # b[gp_index] = -elementary_charge*self.free_charge[gp_index]\
                #     *np.exp(-np.sign(self.free_charge[gp_index])*(self.phi[gp_index]-self.phi_initial[gp_index])/self.kBT)\
                #     -elementary_charge*self.fixed_charge[gp_index]
                b[gp_index] = -elementary_charge*self.free_charge[gp_index] -elementary_charge*self.fixed_charge[gp_index]

                # free charge and fixed charge are number of electrons in real-space grid points
                # the above free_charge form accelerate the convergence of the Poisson equation
                # only internal points have non-zero free_charge and fixed_charge

            else:# boundary points
                A[gp_index,gp_index] = 1.0
                assert b[gp_index] == 0.0
                if self.boudnary_points[gp_index] == "xmin":   
                    A[gp_index,gp_index+1] = -1.0
                elif self.boudnary_points[gp_index] == "xmax":
                    A[gp_index,gp_index-1] = -1.0
                elif self.boudnary_points[gp_index] == "ymin":
                    A[gp_index,gp_index+Nx] = -1.0
                elif self.boudnary_points[gp_index] == "ymax":
                    A[gp_index,gp_index-Nx] = -1.0
                elif self.boudnary_points[gp_index] == "zmin":
                    A[gp_index,gp_index+Nx*Ny] = -1.0
                elif self.boudnary_points[gp_index] == "zmax":
                    A[gp_index,gp_index-Nx*Ny] = -1.0
                elif self.boudnary_points[gp_index] == "Gate":
                    b[gp_index] = -1*self.lead_gate_potential[gp_index]

                #TODO: add lead potential. For dptb-negf, we only need to change zmin and zmax as lead


    def solve_pyamg(self,A,b,tolerance=1e-7,accel=None):
        # solve the Poisson equation
        print('Solve Poisson equation by pyamg')
        pyamg_solver = pyamg.aggregation.smoothed_aggregation_solver(A, max_levels=1000)
        del A
        # print('Poisson equation solver: ',pyamg_solver)
        residuals = []

        def callback(x):
        # residuals calculated in solver is a pre-conditioned residual
        # residuals.append(np.linalg.norm(b - A.dot(x)) ** 0.5)
            print(
                "    {:4d}  residual = {:.5e}   x0-residual = {:.5e}".format(
                    len(residuals) - 1, residuals[-1], residuals[-1] / residuals[0]
                )
            )

        x = pyamg_solver.solve(
            b,
            tol=tolerance,
            # callback=callback,
            residuals=residuals,
            accel=accel,
            cycle="W",
            maxiter=1e3,
        )
        return x


    def solve_poisson(self,method='pyamg',tolerance=1e-7):
        # solve poisson equation:
        if method == 'pyamg':
            print('Solve Poisson equation by pyamg')
            # A,b = self.to_pyamg()
            # self.phi = self.solve_poisson_pyamg(A,b,tolerance)
            # max_diff = np.max(abs(self.phi-self.phi_old))
            # return max_diff
        elif method == 'scipy':
            print('Solve Poisson equation by scipy')
        else:
            raise ValueError('Unknown Poisson solver: ',method)
        # NR iteration
        self.phi_initial = self.phi.copy() # tilde_phi in paper
        norm_avp = 1.0; NR_circle_count = 0
        while norm_avp > 1e-3 and NR_circle_count < 100:
            Jacobian,B = self.to_scipy()
            norm_B = np.linalg.norm(B)
            if method == 'scipy':   
                delta_phi = spsolve(Jacobian,B)
            elif method == 'pyamg':
                delta_phi = self.solve_pyamg(Jacobian,B,tolerance=1e-5)
            else:
                print('Unknown Poisson solver: ',method)
            self.phi_oldstep = self.phi.copy()
            
            max_diff_NR = np.max(abs(delta_phi))
            print('max_diff_NR: ',max_diff_NR)
            norm_avp = np.linalg.norm(delta_phi)
            self.phi += delta_phi
            if norm_avp > 1e-3:
                _,B = self.to_scipy()
                norm_B_new = np.linalg.norm(B)
                print('norm_B_new: ',norm_B_new)
                control_count = 1
                while norm_B_new > norm_B and control_count < 2:
                    self.phi -= delta_phi/np.power(2,control_count)
                    _,B = self.to_scipy()
                    norm_B_new = np.linalg.norm(B)
                    control_count += 1    
                    print('control_count: ',control_count,'  norm_B_new: ',norm_B_new)           
            NR_circle_count += 1
            print('NR circle: ',NR_circle_count,'  norm_avp: ', norm_avp)
        max_diff = np.max(abs(self.phi-self.phi_old))
        return max_diff




    def NR_construct_Jacobian(self,J):
        # construct the Jacobian matrix for the Poisson equation
        
        Nx = self.grid.shape[0];Ny = self.grid.shape[1];Nz = self.grid.shape[2]
        for gp_index in range(self.grid.Np):
            if self.boudnary_points[gp_index] == "in":
                flux_xm = self.grid.surface_grid[gp_index,0]*eps0*(self.eps[gp_index-1]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index,0]-self.grid.grid_coord[gp_index-1,0])
                flux_xp = self.grid.surface_grid[gp_index,0]*eps0*(self.eps[gp_index+1]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index+1,0]-self.grid.grid_coord[gp_index,0])
                
                flux_ym = self.grid.surface_grid[gp_index,1]*eps0*(self.eps[gp_index-Nx]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index-Nx,1]-self.grid.grid_coord[gp_index,1])
                flux_yp = self.grid.surface_grid[gp_index,1]*eps0*(self.eps[gp_index+Nx]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index+Nx,1]-self.grid.grid_coord[gp_index,1])

                flux_zm = self.grid.surface_grid[gp_index,2]*eps0*(self.eps[gp_index-Nx*Ny]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index-Nx*Ny,2]-self.grid.grid_coord[gp_index,2])
                flux_zp = self.grid.surface_grid[gp_index,2]*eps0*(self.eps[gp_index+Nx*Ny]+self.eps[gp_index])*0.5\
                /abs(self.grid.grid_coord[gp_index+Nx*Ny,2]-self.grid.grid_coord[gp_index,2])

                # add flux term to matrix A
                J[gp_index,gp_index] = -(flux_xm+flux_xp+flux_ym+flux_yp+flux_zm+flux_zp)\
                    +elementary_charge*self.free_charge[gp_index]*(-np.sign(self.free_charge[gp_index]))/self.kBT*np.exp(-np.sign(self.free_charge[gp_index])*(self.phi[gp_index]-self.phi_old[gp_index])/self.kBT)
                J[gp_index,gp_index-1] = flux_xm
                J[gp_index,gp_index+1] = flux_xp
                J[gp_index,gp_index-Nx] = flux_ym
                J[gp_index,gp_index+Nx] = flux_yp
                J[gp_index,gp_index-Nx*Ny] = flux_zm
                J[gp_index,gp_index+Nx*Ny] = flux_zp

            else:# boundary points
                J[gp_index,gp_index] = elementary_charge
                
                if self.boudnary_points[gp_index] == "xmin":   
                    J[gp_index,gp_index+1] = -1.0*elementary_charge
                elif self.boudnary_points[gp_index] == "xmax":
                    J[gp_index,gp_index-1] = -1.0*elementary_charge
                elif self.boudnary_points[gp_index] == "ymin":
                    J[gp_index,gp_index+Nx] = -1.0*elementary_charge
                elif self.boudnary_points[gp_index] == "ymax":
                    J[gp_index,gp_index-Nx] = -1.0*elementary_charge
                elif self.boudnary_points[gp_index] == "zmin":
                    J[gp_index,gp_index+Nx*Ny] = -1.0*elementary_charge
                elif self.boudnary_points[gp_index] == "zmax":
                    J[gp_index,gp_index-Nx*Ny] = -1.0*elementary_charge
                elif self.boudnary_points[gp_index] == "Gate":
                    J[gp_index,gp_index] = elementary_charge
        
    def NR_construct_B(self,B):
    # construct the -B matrix in NR iteration
    # Note that the sign of B has been changed for convenience in later NR iteration
    # return -B
        Nx = self.grid.shape[0];Ny = self.grid.shape[1];Nz = self.grid.shape[2]
        for gp_index in range(self.grid.Np):
            if self.boudnary_points[gp_index] == "in":
                flux_xm = self.grid.surface_grid[gp_index,0]*eps0*(self.eps[gp_index-1]+self.eps[gp_index])*0.5\
                *(self.phi[gp_index-1]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index,0]-self.grid.grid_coord[gp_index-1,0])
                flux_xp = self.grid.surface_grid[gp_index,0]*eps0*(self.eps[gp_index+1]+self.eps[gp_index])*0.5\
                *(self.phi[gp_index+1]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index+1,0]-self.grid.grid_coord[gp_index,0])
                
                flux_ym = self.grid.surface_grid[gp_index,1]*eps0*(self.eps[gp_index-Nx]+self.eps[gp_index])*0.5\
                *(self.phi[gp_index-Nx]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index-Nx,1]-self.grid.grid_coord[gp_index,1])
                flux_yp = self.grid.surface_grid[gp_index,1]*eps0*(self.eps[gp_index+Nx]+self.eps[gp_index])*0.5\
                *(self.phi[gp_index+Nx]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index+Nx,1]-self.grid.grid_coord[gp_index,1])

                flux_zm = self.grid.surface_grid[gp_index,2]*eps0*(self.eps[gp_index-Nx*Ny]+self.eps[gp_index])*0.5\
                *(self.phi[gp_index-Nx*Ny]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index-Nx*Ny,2]-self.grid.grid_coord[gp_index,2])
                flux_zp = self.grid.surface_grid[gp_index,2]*eps0*(self.eps[gp_index+Nx*Ny]+self.eps[gp_index])*0.5\
                *(self.phi[gp_index+Nx*Ny]-self.phi[gp_index])/abs(self.grid.grid_coord[gp_index+Nx*Ny,2]-self.grid.grid_coord[gp_index,2])
 
                # add flux term to matrix B
                B[gp_index] = (flux_xm+flux_xp+flux_ym+flux_yp+flux_zm+flux_zp)
                B[gp_index] += elementary_charge*self.free_charge[gp_index]*np.exp(-np.sign(self.free_charge[gp_index])*(self.phi[gp_index]-self.phi_old[gp_index])/self.kBT)\
                    +elementary_charge*self.fixed_charge[gp_index]


            else:# boundary points
                        
                if self.boudnary_points[gp_index] == "xmin":   
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index+1])*elementary_charge
                elif self.boudnary_points[gp_index] == "xmax":
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index-1])*elementary_charge
                elif self.boudnary_points[gp_index] == "ymin":
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index+Nx])*elementary_charge
                elif self.boudnary_points[gp_index] == "ymax":
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index-Nx])*elementary_charge
                elif self.boudnary_points[gp_index] == "zmin":
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index+Nx*Ny])*elementary_charge
                elif self.boudnary_points[gp_index] == "zmax":
                    B[gp_index] = (self.phi[gp_index]-self.phi[gp_index-Nx*Ny])*elementary_charge
                elif self.boudnary_points[gp_index] == "Gate":
                    B[gp_index] = (self.phi[gp_index]+self.lead_gate_potential[gp_index])*elementary_charge
                #TODO: add lead potential. For dptb-negf, we only need to change zmin and zmax as lead

            if B[gp_index]!=0: # for convenience change the sign of B in later NR iteration
                B[gp_index] = -B[gp_index]
        


    

        
        