import numpy as np 
from pyamg.gallery import poisson

class grid(object):
    # define the grid in 3D space
    def __init__(self,xg,yg,zg,xa,ya,za):
        # xg,yg,zg are the coordinates of the basic grid points
        self.xg = np.around(xg,decimals=5);self.yg = np.around(yg,decimals=5);self.zg = np.around(zg,decimals=5)
        # xa,ya,za are the coordinates of the atoms
        # atom should be within the grid
        assert xa.all() >= np.min(xg) and xa.all() <= np.max(xg)
        assert ya.all() >= np.min(yg) and ya.all() <= np.max(yg)
        assert za.all() >= np.min(zg) and za.all() <= np.max(zg)

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


        self.Np = int(len(self.xall)*len(self.yall)*len(self.zall))
        assert self.Np == len(self.xmesh)
        assert self.grid_coord.shape[0] == self.Np

        print('Number of grid points: ',self.Np,' grid shape: ',self.grid_coord.shape,' Number of atoms: ',self.Na)

        self.atom_index = self.find_atom_index(xa,ya,za)


        # create surface area for each grid point
        surface_grid = np.zeros((self.Np,3))
        x_vorlen = self.cal_vorlen(self.xall);y_vorlen = self.cal_vorlen(self.yall);z_vorlen = self.cal_vorlen(self.zall)
        
        ## surface along x-axis (yz-plane)
        XD,YD = np.meshgrid(x_vorlen,y_vorlen)
        ax,bx = np.meshgrid(YD.flatten(),z_vorlen)
        surface_grid[:,0] = abs((ax*bx).flatten())
        ## surface along y-axis (xz-plane) 
        



    def find_atom_index(self,xa,ya,za):
        # find the index of the atoms in the grid
        swap = {}
        for i in range(self.Na):
            for j in range(self.Np):
                if xa[i]==self.xmesh[j] and ya[i]==self.ymesh[j] and za[i]==self.zmesh[j]:
                    swap.update({i:j})
        return swap
    
    def cal_vorlen(self,x):
        # compute the length of the Voronoi segment of a one-dimensional array x
        xd = np.zeros(len(x))
        xd[0] = abs(x[0]-x[1])/2
        xd[-1] = abs(x[-1]-x[-2])/2
        for i in range(1,len(x)-1):
            xd[i] = (abs(x[i]-x[i-1])+abs(x[i]-x[i+1]))/2
        return xd


class gate(object):
    def __init__(self,xmin,xmax,ymin,ymax,zmin,zmax):
        self.Ef = 0.0
        # gate region
        self.xmin = xmin; self.xmax = xmax
        self.ymin = ymin; self.ymax = ymax
        self.zmin = zmin; self.zmax = zmax

class medium(object):
    def __init__(self,xmin,xmax,ymin,ymax,zmin,zmax):
        self.eps = 1.0
        # gate region
        self.xmin = xmin; self.xmax = xmax
        self.ymin = ymin; self.ymax = ymax
        self.zmin = zmin; self.zmax = zmax

class interface3D(object):
    def __init__(self,grid,*args):
        assert grid.__class__.__name__ == 'grid'

        region_name = ['gate','medium']
        for i in range(0,len(args)):
            if not args[i].__class__.__name__ in region_name:
                raise ValueError('Unknown region type: ',args[i].__class__.__name__)

        self.grid = grid
        self.eps = np.zeros(grid.Np) # dielectric permittivity
        self.phi = np.zeros(grid.Np) # potential
        self.free_charge = np.zeros(grid.Np) # free charge density
        self.fixed_charge = np.zeros(grid.Np) # fixed charge density

        self.boudnary_points = {i:"in" for i in range(self.grid.Np)} # initially set all points as internal
        self.boudnary_points_init()

        self.lead_gate_potential = np.zeros(grid.Np) # no gate potential initially
        self.gate_potential_eps_init(args)


    def boudnary_points_init(self):
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
    
    def gate_potential_eps_init(self,args):
        # set the gate potential
        # ingore the lead potential temporarily
        for i in range(len(args)):
            if args[i].__class__.__name__ == 'gate' or args[i].__class__.__name__ == 'medium':
                
                # find gate region in grid
                index=np.nonzero((args[i].xmin<=self.grid.grid_coord[0])&(args[i].xmax>=self.grid.grid_coord[0])&
                            (args[i].ymin<=self.grid.grid_coord[1])&(args[i].ymax>=self.grid.grid_coord[1])&
                            (args[i].zmin<=self.grid.grid_coord[2])&(args[i].zmax>=self.grid.grid_coord[2]))
                if args[i].__class__.__name__ == 'gate': #attribute gate potential to the corresponding grid points
                    self.lead_gate_potential[index] = args[i].Ef 
                else:
                    self.eps[index] = args[i].eps
        
    def to_pyamg(self,dtype=None):
        # convert to amg format A,b matrix
        if dtype == None:
            dtype = np.float64
        A = poisson(self.grid.shape,format='csr',dtype=dtype)
        b = np.zeros(A.shape[0],dtype=A.dtype)
        self.set_amg_boundary(A,b)
        
        return A,b
    
    def set_amg_boundary(self,A,b):

        def Dirichlet(idx,A,b): #第一类边界条件
            # Default pyamg Poisson matrix has Dirichlet BC
            b[idx] = 0.0  #为何要将边界点的值设为0



        # def Neumann(idx_bc, idx_p1): #第二类边界条件
        #     # Set all boundary equations to 0 
        #     s = _a.array_arange(A.indptr[idx_bc], A.indptr[idx_bc + 1])
        #     A.data[s] = 0
        #     # force the boundary cells to equal the neighbouring cell
        #     A[idx_bc, idx_bc] = 1
        #     A[idx_bc, idx_p1] = -1
        #     A.eliminate_zeros()
        #     b[idx_bc] = 0.0

        


    

        
        