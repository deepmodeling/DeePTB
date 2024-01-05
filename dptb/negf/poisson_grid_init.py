import numpy as np 


class grid(object):
    def __init__(self,xg,yg,zg,xa,ya,za):
        # xg,yg,zg are the coordinates of the basic grid points
        self.xg = np.around(xg,decimals=5)
        self.yg = np.around(yg,decimals=5)
        self.zg = np.around(zg,decimals=5)
        # xa,ya,za are the coordinates of the atoms
        self.Na = len(xa) # number of atoms
        uxa = np.unique(xa)
        uya = np.unique(ya)
        uza = np.unique(za)
        # x,y,z are the coordinates of the grid points
        self.xall = np.unique(np.concatenate((uxa,self.xg),0)) # unique results are sorted
        self.yall = np.unique(np.concatenate((uya,self.yg),0))
        self.zall = np.unique(np.concatenate((uza,self.zg),0))

        assert len(self.xall) == len(self.yall)
        assert len(self.yall) == len(self.zall)
        # create meshgrid
        xmesh,ymesh,zmesh = np.meshgrid(self.xall,self.yall,self.zall)
        self.xmesh = xmesh.flatten()
        self.ymesh = ymesh.flatten()
        self.zmesh = zmesh.flatten()

        self.Np = int(len(self.xall)*len(self.yall)*len(self.zall))
        assert self.Np == len(self.xmesh)

        print('Number of grid points: ',self.Np)

        self.atom_index = self.find_atom_index(xa,ya,za)

    def find_atom_index(self,xa,ya,za):
        # find the index of the atoms in the grid
        swap = {}
        for i in range(self.Na):
            for j in range(self.Np):
                if xa[i]==self.xmesh[j] and ya[i]==self.ymesh[j] and za[i]==self.zmesh[j]:
                    swap.update({i:j})
        return swap

class gate(object):
    def __init__(self):
        self.Ef = 0.0




class interface3D(object):
    def __init__(self,grid,*args):
        self.grid = grid

        region_name = ['gate','medium']
        
        for i in range(0,len(args)):
            if not args[i].__class__.__name__ in region_name:
                raise ValueError('Unknown region type: ',args[i])
        