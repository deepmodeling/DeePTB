import numpy as np
from collections  import OrderedDict

import torch as th
import torch.nn as nn


class ForwardNet(nn.Module):
    def __init__(self, insize, nneurons, afunc='tanh'):
        super(ForwardNet, self).__init__()
        assert(len(nneurons)>=1)
        self.nneurons = [insize] + nneurons
        if afunc.lower() == 'tanh':
            self.AFunc  = nn.Tanh()
        elif afunc.lower() == 'sigmode':
            self.AFunc  = nn.Sigmoid()
        elif afunc.lower() == 'relu':
            self.AFunc  =  nn.ReLU()
        else:
            print('no active function ')
        
        nnorderdict = OrderedDict()
        if len(nneurons) == 1:
            nnorderdict['input'] = nn.Linear(insize,nneurons[0])
        elif len(nneurons) > 1:
            nnorderdict['input']  = nn.Linear(insize,nneurons[0])
            nnorderdict['activein'] = self.AFunc
            for i in range(1,len(nneurons)-1):
                nnorderdict['hidden'+str(i)] = nn.Linear(nneurons[i-1],nneurons[i])
                nnorderdict['active'+str(i)] = self.AFunc   
            nnorderdict['output'] = nn.Linear(nneurons[-2],nneurons[-1])
        else:
            print('Error in nneuron settings.')
            
        self.nnet = nn.Sequential(nnorderdict)
        
    def forward(self,inputs):
        out = self.nnet(inputs)
        return out




class ResidualBlock(nn.Module):
    def __init__(self, insize, outsize, afunc='tanh'):
        super(ResidualBlock,self).__init__()
        self.insize = insize
        self.outsize = outsize
        if afunc.lower() == 'tanh':
            self.AFunc  = nn.Tanh()
        elif afunc.lower() == 'sigmode':
            self.AFunc  = nn.Sigmoid()
        elif afunc.lower() == 'relu':
            self.AFunc  =  nn.ReLU()
        else:
            print('no active function ')

        self.linear = nn.Linear(insize,outsize)

    def forward(self,inputs):
        redsidual = inputs
        out = self.linear(inputs)
        out = self.AFunc(out)
        
        if self.outsize == self.insize:
            out = out + redsidual
        elif self.outsize == self.insize * 2: 
            out = out + th.cat([redsidual,redsidual],axis=1)

        return out


class ResNet(nn.Module):
    def __init__(self, block, insize, nneurons, afunc='tanh'):
        super(ResNet,self).__init__()
        assert(len(nneurons)>=1)
        self.afunctag = afunc
        self.nneurons = [insize] + nneurons
        
        nnorderdict = OrderedDict()
        #layer = []
        for i in range(1,len(self.nneurons)):
        #    layer.append(block(self.nneurons[i-1],self.nneurons[i]))
            nnorderdict['embeding'+str(i)] = block(self.nneurons[i-1],self.nneurons[i])
        #self.resnet = nn.Sequential(*layer)
        self.resnet = nn.Sequential(nnorderdict)

    def forward(self,inputs):
        out = self.resnet(inputs)
        return out


class BuildNN(object):
    def __init__(self,envtype, bondtype, proj_anglr_m):
        print('# allocate nn models')
        self.AnglrMID = {'s':0,'p':1,'d':2,'f':3}
        # bond and env type can get from stuct class. 
        self.envtype  = envtype
        self.bondtype = bondtype
        # projected angular momentum. get from struct class.
        self.ProjAnglrM = proj_anglr_m

    def BuildEnvNN(self, env_neurons, afunc='tanh'):
        # env_neurons  = np.asarray(env_neurons)
        assert(len(env_neurons) >=2 )
        insize       = env_neurons[0]
        env_neurons2 = list(env_neurons[1:])
        envmodel = {}
        for ib in self.bondtype:
            for ii in self.envtype:
                envmodel[ib+ii] = ResNet(block=ResidualBlock,
                    insize=insize, nneurons=env_neurons2 ,afunc=afunc)

        return envmodel

    def BuildBondNN(self,bond_neurons,afunc='tanh'):
        # bond_neurons = np.asarray(bond_neurons)
        assert(len(bond_neurons) >=2 )
        insize  = bond_neurons[0]
        bond_neurons2 = list(bond_neurons[1:])
        self.bond_index_map, self.bond_num_hops = self.Bond_Ind_Mapings()
        
        bondmodel = {}

        for ii  in range(len(self.bondtype)):
            itype = self.bondtype[ii]
            for jj in range(ii,len(self.bondtype)):
                jtype = self.bondtype[jj]
                outsize = self.bond_num_hops[itype+jtype]
                # print("# bond type " + itype + ' --> ' + jtype)
                bondmodel[itype+jtype] = ForwardNet(insize=insize,
                    nneurons=bond_neurons2 + [outsize], afunc=afunc)

        return bondmodel



    def BuildOnsiteNN(self,onsite_neurons,afunc='tanh'):
        #onsite_neurons = np.asarray(onsite_neurons)
        assert(len(onsite_neurons)>=2)
        insize  = onsite_neurons[0]
        onsite_neurons2 = list(onsite_neurons[1:])
        self.onsite_index_map, self.onsite_num = self.Onsite_Ind_Mapings()
        onsitemodel={}
        for ii in self.bondtype:
            outsize = self.onsite_num[ii]
            onsitemodel[ii] = ForwardNet(insize=insize,
                nneurons=onsite_neurons2 + [outsize], afunc=afunc)

        return onsitemodel



    def Bond_Ind_Mapings(self):
        """ define a rule for mapping between the hoppings and the output of NN.
        eg. the 0-th element is assigned to the s-s sigam bond, etc. 
        """
        bond_index_map = {}
        bond_num_hops ={}
        for it in range(len(self.bondtype)):
            for jt in range(len(self.bondtype)):
                itype = self.bondtype[it]
                jtype = self.bondtype[jt]
                orbdict = {}
                ist = 0
                numhops = 0
                for ish in self.ProjAnglrM[it]:
                    for jsh in self.ProjAnglrM[jt]:
                        ishid = self.AnglrMID[ish]
                        jshid = self.AnglrMID[jsh]
                        # the same type atoms of the bond.
                        if it == jt:
                            if ishid > jshid:
                                orbdict[ish+jsh] = orbdict[jsh + ish]
                                continue
                            else:
                                numhops += min(ishid,jshid)+1
                                orbdict[ish+jsh] = np.arange(ist,ist+ min(ishid,jshid)+1).tolist()

                        elif it < jt: 
                            numhops += min(ishid,jshid)+1  
                            orbdict[ish+jsh] = np.arange(ist,ist+ min(ishid,jshid)+1).tolist()
                        else:
                            numhops += min(ishid,jshid)+1
                            orbdict[ish+jsh] = bond_index_map[jtype+itype][jsh+ish]
                
                        # orbdict[ish+jsh] = paralist
                        ist += min(ishid,jshid)+1
                        # print (itype, jtype, ish+jsh, ishid, jshid,paralist)
                bond_index_map[itype + jtype] = orbdict
                bond_num_hops[itype + jtype]  = numhops
                
        for key in bond_index_map.keys():
            print('# '+key+':', bond_num_hops[key] , ' independent hoppings')
            print('## ',end='')
            ic=1
            for key2 in bond_index_map[key]:

                print('' + key2 +':',bond_index_map[key][key2],'   ',end='')
                if ic%6==0:
                    print('\n## ',end='')
                ic+=1
            print()
        
        return bond_index_map, bond_num_hops
    
    def Onsite_Ind_Mapings(self):
        onsite_index_map = {}
        onsite_num ={}
        for it in range(len(self.bondtype)):
            itype = self.bondtype[it]
            orbdict = {}
            ist = 0
            numhops = 0
            for ish in self.ProjAnglrM[it]:
                ishid = self.AnglrMID[ish]
                #orbdict[ish] = [ist]
                orbdict[ish] = np.arange(ist, ist + 2 * ishid + 1).tolist()
                ist += 2*ishid + 1
                numhops += 2*ishid + 1
            onsite_index_map[itype] = orbdict
            onsite_num[itype]  = numhops
        
        for key in onsite_index_map.keys():
            print('# '+key+':', onsite_index_map[key] , ' independent onsite Es')
            print('## ',end='')
            ic=1
            for key2 in onsite_index_map[key]:

                print('' + key2 +':',onsite_index_map[key][key2],'   ',end='')
                if ic%6==0:
                    print('\n## ',end='')
                ic+=1
            print()

        return onsite_index_map, onsite_num