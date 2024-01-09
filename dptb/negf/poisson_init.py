import numpy as np 
import pyamg
from pyamg.gallery import poisson
from utils.constants import elementary_charge as q
from utils.constants import Boltzmann
from scipy.constants import epsilon_0 as eps0  #TODO:later add to untils.constants.py




class Grid(object):
    # define the grid in 3D space
    def __init__(self,xg,yg,zg,xa,ya,za):
        # xg,yg,zg are the coordinates of the basic grid points
        self.xg = np.around(xg,decimals=5);self.yg = np.around(yg,decimals=5);self.zg = np.around(zg,decimals=5)
        # xa,ya,za are the coordinates of the atoms
        # atom should be within the grid
        assert (xa-np.min(xg)).all() and (xa-np.max(xg)).all()
        assert (ya-np.min(yg)).all() and (ya-np.max(yg)).all()
        assert (za-np.min(zg)).all() and (za-np.max(zg)).all()

        self.Na = len(xa) # number of atoms
        uxa = np.unique(xa);uya = np.unique(ya);uza = np.unique(za)
        # x,y,z are the coordinates of the grid points
        self.xall = np.unique(np.concatenate((uxa,self.xg),0)) # unique results are sorted
        self.yall = np.unique(np.concatenate((uya,self.yg),0))
        self.zall = np.unique(np.concatenate((uza,self.zg),0))
        self.shape = (len(self.xall),len(self.yall),len(self.zall))


        # create meshgrid
        xmesh,ymesh,zmesh = np.meshgrid(self.xall,self.yall,self.zall)
        self.xmesh = xmesh.flatten()
        self.ymesh = ymesh.flatten()
        self.zmesh = zmesh.flatten()
        self.grid_coord = np.array([self.xmesh,self.ymesh,self.zmesh]).T #(Np,3)
        sorted_indices = np.lexsort((self.xmesh , self.ymesh , self.zmesh))
        self.grid_coord = self.grid_coord[sorted_indices] # sort the grid points firstly along x, then y, lastly z        
        ## check the number of grid points
        self.Np = int(len(self.xall)*len(self.yall)*len(self.zall))
        assert self.Np == len(self.xmesh)
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
                if xa[atom_index]==self.xmesh[gp_index] and ya[atom_index]==self.ymesh[gp_index] and za[atom_index]==self.zmesh[gp_index]:
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


    
class Gate(object):
    def __init__(self,xmin,xmax,ymin,ymax,zmin,zmax):
        self.Ef = 0.0
        # gate region
        self.xmin = xmin; self.xmax = xmax
        self.ymin = ymin; self.ymax = ymax
        self.zmin = zmin; self.zmax = zmax

class Dielectric(object):
    def __init__(self,xmin,xmax,ymin,ymax,zmin,zmax):
        self.eps = 1.0
        # gate region
        self.xmin = xmin; self.xmax = xmax
        self.ymin = ymin; self.ymax = ymax
        self.zmin = zmin; self.zmax = zmax







class Interface3D(object):
    def __init__(self,grid,gate_list,dielectric_list):
        assert grid.__class__.__name__ == 'Grid'

        
        for i in range(0,len(gate_list)):
            if not gate_list[i].__class__.__name__ == 'Gate':
                raise ValueError('Unknown region type: ',gate_list[i].__class__.__name__)
        for i in range(0,len(dielectric_list)):
            if not dielectric_list[i].__class__.__name__ == 'Dielectric':
                raise ValueError('Unknown region type: ',dielectric_list[i].__class__.__name__)
            
        self.grid = grid
        self.eps = np.zeros(grid.Np) # dielectric permittivity
        self.phi = np.zeros(grid.Np) # potential
        self.phi_old = np.zeros(grid.Np) # potential in the previous iteration
        self.free_charge = np.zeros(grid.Np) # free charge density
        self.fixed_charge = np.zeros(grid.Np) # fixed charge density

        self.Temperature = 300.0 # temperature in unit of Kelvin
        self.kBT = Boltzmann*self.Temperature # thermal energy

        # store the boundary information: xmin,xmax,ymin,ymax,zmin,zmax,gate
        self.boudnary_points = {i:"in" for i in range(self.grid.Np)} # initially set all points as internal
        self.boudnary_points_get()

        self.lead_gate_potential = np.zeros(grid.Np) # no gate potential initially, all grid points are set to zero
        self.potential_eps_get(gate_list)
        self.potential_eps_get(dielectric_list)


    def boudnary_points_get(self):
        # set the boundary points
        for i in range(self.grid.Np):
            if self.grid.xmesh[i] == np.min(self.grid.xall):
                self.boudnary_points[i] = "xmin"
            elif self.grid.xmesh[i] == np.max(self.grid.xall):
                self.boudnary_points[i] = "xmax"
            elif self.grid.ymesh[i] == np.min(self.grid.yall):
                self.boudnary_points[i] = "ymin"
            elif self.grid.ymesh[i] == np.max(self.grid.yall):
                self.boudnary_points[i] = "ymax"
            elif self.grid.zmesh[i] == np.min(self.grid.zall):
                self.boudnary_points[i] = "zmin"
            elif self.grid.zmesh[i] == np.max(self.grid.zall):
                self.boudnary_points[i] = "zmax"
        internal_NP = 0
        for i in range(self.grid.Np):
            if self.boudnary_points[i] == "in":
                internal_NP += 1
        self.internal_NP = internal_NP
    
    def potential_eps_get(self,region_list):
        # set the gate potential
        # ingore the lead potential temporarily
        for i in range(len(region_list)):    
            # find gate region in grid
            index=np.nonzero((region_list[i].xmin<=self.grid.grid_coord[0])&(region_list[i].xmax>=self.grid.grid_coord[0])&
                        (region_list[i].ymin<=self.grid.grid_coord[1])&(region_list[i].ymax>=self.grid.grid_coord[1])&
                        (region_list[i].zmin<=self.grid.grid_coord[2])&(region_list[i].zmax>=self.grid.grid_coord[2]))
            if region_list[i].__class__.__name__ == 'Gate': #attribute gate potential to the corresponding grid points
                self.boudnary_points[index] = "Gate"
                self.lead_gate_potential[index] = region_list[i].Ef 
            elif region_list[i].__class__.__name__ == 'Dielectric':
                self.eps[index] = region_list[i].eps
            else:
                raise ValueError('Unknown region type: ',region_list[i].__class__.__name__)
        
    def to_pyamg(self,dtype=None):
        # convert to amg format A,b matrix
        if dtype == None:
            dtype = np.float64
        A = poisson(self.grid.shape,format='csr',dtype=dtype)
        b = np.zeros(A.shape[0],dtype=A.dtype)
        A.data[:] = 0  # set all elements to zero
        # later we set non-zero elements to A, the indices and indptr are not changed as the default grid order in pyamg
        # is the same as that of self.grid.grid_coord
        self.construct_poisson(A,b)
        
        return A,b
    
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

                b[gp_index] = -q*self.free_charge[gp_index]\
                    *np.exp(-np.sign(self.free_charge[gp_index])*(self.phi[gp_index]-self.phi_old[gp_index])/self.kBT)\
                    -q*self.fixed_charge[gp_index]
                # the above free_charge form accelerate the convergence of the Poisson equation
                # only internal points have non-zero free_charge and fixed_charge

            else:# boundary points
                A[gp_index,gp_index] = 1.0
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


    def solve_poisson_pyamg(self,A,b,tolerance=1e-12,accel=None):
        # solve the Poisson equation
        print('Solve Poisson equation by pyamg')
        pyamg_solver = pyamg.aggregation.smoothed_aggregation_solver(A, max_levels=1000)
        del A
        print('Poisson equation solver: ',pyamg_solver)
        residuals = []

        def callback(x):
        # residuals calculated in solver is a pre-conditioned residual
        # residuals.append(np.linalg.norm(b - A.dot(x)) ** 0.5)
            print(
                "    {:4d}  residual = {:.5e}   x0-residual = {:.5e}".format(
                    len(residuals) - 1, residuals[-1], residuals[-1] / residuals[0]
                )
            )

        x = pyamg_solver(
            b,
            tol=tolerance,
            callback=callback,
            residuals=residuals,
            accel=accel,
            cycle="W",
            maxiter=1e7,
        )
        print("Done solving the Poisson equation!")
        return x


    def solve_poisson(self,method='pyamg'):
        # solve poisson equation:
        if method == 'pyamg':
            A,b = self.to_pyamg()
            self.phi = self.solve_poisson_pyamg(A,b)
            self.phi_old = self.phi.copy()
        else:
            raise ValueError('Unknown Poisson solver: ',method)



        


    

        
        