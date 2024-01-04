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
        self.x = np.unique(np.concatenate((uxa,self.xg),0)).sort()
        self.y = np.unique(np.concatenate((uya,self.yg),0)).sort()
        self.z = np.unique(np.concatenate((uza,self.zg),0)).sort()
        self.Np = int(len(self.x)*len(self.y)*len(self.z))
        print('Number of grid points: ',self.Np)

        self.atom_index = self.find_atom_index(xa,ya,za)

    def find_atom_index(self,xa,ya,za):
        # find the index of the atoms in the grid
        swap = np.zeros(self.Na)
        for i in range(self.Na):
            for j in range(self.Np):
                if xa[i]==self.x[j] and ya[i]==self.y[j] and za[i]==self.z[j]:
                    swap[i] = j
        return swap
