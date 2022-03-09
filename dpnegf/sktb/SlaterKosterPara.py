import os
import re
import ase
import ase.atom
import numpy as  np
from scipy.interpolate import interp1d


class SlaterKosterInt(object):
    def __init__(self,paras):
        self.MaxShells  = 3  # s p d only! f will be add on later.
        self.NumHvals   = 10  # dd1 dd2 dd3, pd1 pd2, pp1 pp2, sd1, sp1, ss1. spd only. for spdf 20.
        self.AnglrMID   = {'s':0,'p':1,'d':2,'f':3}
        self.atomic_num_dict = ase.atom.atomic_numbers
        self.atomic_num_dict_r =  dict(zip(self.atomic_num_dict.values(), self.atomic_num_dict.keys()))
        
        self.AtomTypes  = self.get_uniq_symbol(paras.ProjAtomType)
        self.NAtomTypes = len(self.AtomTypes)
        # self.AnglrM     = paras.ProjAnglrM
        self.AnglrM = []
        for ii in self.AtomTypes:
            self.AnglrM.append(paras.ProjAnglrM[ii])

        self.SKFilePath = paras.SKFilePath
        self.Separator  = paras.Separator
        self.Suffix     = paras.Suffix
        self.atomic_num_dict = ase.atom.atomic_numbers
        self.atomic_num_dict_r =  dict(zip(self.atomic_num_dict.values(), self.atomic_num_dict.keys()))
        
        for ii in range(self.NAtomTypes):
            print('# Atom type: ' + self.AtomTypes[ii] + ', ID : %d' %(ii+1))
        
        if len(self.AnglrM) != self.NAtomTypes:
            print('Error, should input the angularmonentum for every atom type.')
            exit()
            
        self.SiteE = np.zeros([self.NAtomTypes,self.MaxShells])
        self.HubdU = np.zeros([self.NAtomTypes,self.MaxShells])
        self.Occu  = np.zeros([self.NAtomTypes,self.MaxShells])
        
        SKfile = []
        for itype in self.AtomTypes:
            sktmp = []
            for jtype in self.AtomTypes:
                curfilename = self.SKFilePath + '/' + itype + self.Separator  + jtype + self.Suffix
                if not os.path.exists(curfilename):
                    print('Didn\'t find the skfile ' + curfilename)
                sktmp.append(curfilename)
            SKfile.append(sktmp)
        self.SKfiles = np.asarray(SKfile)

        self.AnglrMind = []
        self.Num_Orbs  = []
        for ii in range(self.NAtomTypes):
            aglids = []
            orb_tmp = []
            for ia in range(len(self.AnglrM[ii])):
                aglidtmp = self.AnglrMID[self.AnglrM[ii][ia]]
                aglids.append(aglidtmp)
                orb_tmp.append(2*aglidtmp +1)
            self.AnglrMind.append(aglids)
            self.Num_Orbs.append(orb_tmp)

    def get_uniq_symbol(self, atomsymbols):
       atom_num=[]
       for it in atomsymbols:
           atom_num.append(self.atomic_num_dict[it])
       # uniq and sort.
       uniq_atom_num = sorted(np.unique(atom_num),reverse=True)
       # assert(len(uniq_atom_num) == len(atomsymbols))
       atomtype = []
       for ia in uniq_atom_num:
           atomtype.append(self.atomic_num_dict_r[ia])

       return atomtype

    @staticmethod
    def Formatreadline(line):
        lsplit = re.split(',|;| +|\n|\t',line)
        lsplit = list(filter(None,lsplit))
        lstr = []
        for ii in range(len(lsplit)):
            strtmp = lsplit[ii]
            if re.search('\*',strtmp):
                strspt = re.split('\*|\n',strtmp)
                strspt = list(filter(None,strspt))
                strfull = int(strspt[0]) * [strspt[1]] 
                lstr +=strfull
            else:
                lstr += [strtmp]
        return lstr
        
    def ReadSKfiles(self):
        self.GridDist = []
        self.NGrid    = []
        self.HSintgrl  = []
        for ii in range(self.NAtomTypes):
            gridtmp  = []
            ngridtmp = []
            HSintgrltmp = []
            for jj in range(self.NAtomTypes):
                filename = self.SKfiles[ii,jj]
                print('# Reading SlaterKoster File......' )
                print('# ' + filename)
                fr = open(filename)
                data = fr.readlines()
                fr.close()
                # Line 1
                datline = self.Formatreadline(data[0])
                gridDist, ngrid = float(datline[0]), int(datline[1])
                ngrid = ngrid -1
                gridtmp.append(gridDist)
                ngridtmp.append(ngrid)
                
                HSvals = np.zeros([ngrid, self.NumHvals*2]) 

                if ii == jj and ( self.AtomTypes[ii] == self.AtomTypes[jj] ):
                    print('# This file is a Homo-nuclear case!')
                    # Line 2 for Homo-nuclear case 
                    datline = self.Formatreadline(data[1])
                    for ish in range(self.MaxShells):
                        # Ed Ep Es, spe, Ud Up Us, Od Op Os. 
                        # order from d p s -> s p d.
                        self.SiteE[ii,ish] = float(datline[2-ish])
                        self.HubdU[ii,ish] = float(datline[6-ish])
                        self.Occu[ii,ish]  = float(datline[9-ish]) 
                        
                    for il in range(3,3+ngrid):
                        datline = self.Formatreadline(data[il])
                        HSvals[il-3]  = np.array([float(val) for val in datline[0:2*self.NumHvals]])
                else:
                    print('# This is for Hetero-nuclear case!')
                    for il in range(2,2+ngrid): 
                        datline = self.Formatreadline(data[il])
                        HSvals[il-2]  = np.array([float(val) for val in datline[0:2*self.NumHvals]])
                HSintgrltmp.append(HSvals)
            self.GridDist.append(gridtmp)
            self.NGrid.append(ngridtmp)
            self.HSintgrl.append(HSintgrltmp)
        self.GridDist = np.asarray(self.GridDist)
        self.NGrid    = np.asarray(self.NGrid)
        self.HSintgrl = np.asarray(self.HSintgrl)
        
        
    def IntpSKfunc(self):
        self.MaxDistail = 1.0
        self.IntpSKf = []
        for ia in range(self.NAtomTypes):
            functmp = []
            for ja in range(self.NAtomTypes):
                xlist = np.arange(1,self.NGrid[ia,ja]+1)*self.GridDist[ia,ja]
                xlist = np.append(xlist,[xlist[-1]+self.MaxDistail],axis=0)
                target = self.HSintgrl[ia,ja]
                target = np.append(target,np.zeros([1,2*self.NumHvals]),axis=0)
                intpfunc = interp1d(xlist, target , axis=0)
                functmp.append(intpfunc)
            self.IntpSKf.append(functmp)

    def IntpSK(self,itype,jtype,dist):
        if hasattr(self,'IntpSKf'):
            self.IntpSKfunc()
        eps = 1.0E-6
        maxlength =  self.NGrid[itype,jtype] * self.GridDist[itype,jtype] + self.MaxDistail - eps
        minlength =  self.GridDist[itype,jtype] 
        if dist > maxlength:
            res = np.zeros(2*self.NumHvals)
        elif dist < minlength:
            print('The bond distance is extremely small. : %f' %dist)
            exit()
        else:
            res = self.IntpSKf[itype][jtype](dist)
            
        return res
