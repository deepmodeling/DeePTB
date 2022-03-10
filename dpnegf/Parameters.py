# -*- coding: utf-8 -*-
import re, sys
import json
import argparse
class Paras(object):
    """ 
    a class store all the input parameters.
    """
    def __init__(self, file, command:str='train', nn_off:str=None):
        """  initial the input json and check the input file.

        Parameters
        ----------
        file object to open input.json file.
        args: arguments from command line.
        """

        self.input_json = json.load(file)
        self.GUI = False

        if nn_off == None:
            nn_off_tag = False
        elif nn_off.lower() in ['n','f','no','false']:
            nn_off_tag = False
        elif nn_off.lower() in ['y','yes','t','true']:
            nn_off_tag = True
        else:
            print('nn_off should be yes y true t or no n false f.')
            print('We didnot recognize the nn_off tag, set it to defalt value: False.')
            nn_off_tag = False

        self.SKfileparas()
        if command == 'train':
            self.trainparas()
        elif command == 'test':
            self.testparas()
        elif command == 'negf':
            if nn_off_tag:
                self.sktbnegfparas()
            else:
                self.nntbnegfparas()
            

    def SKfileparas(self):
        """  initial the default sk files parameters.
        """
        self.SKFilePath = "./slakos"
        self.Separator = "-"
        self.Suffix = ".skf"

    def trainparas(self):
        """ initial default parameters and user define parameters for NNTB .

        Attributes
        ----------
        istrain, istest, ispredict : used NN to train, test or predict.
        SpinDeg: the spin degeneracy of the band structure.
        active_func: activation function used in NN.
        train_data_path: path for train data.
        prefix: the prefix of train datas.
        valddir: the directory of validation data.
        withref: train the NN with or without reference structure.
        refdir : directory reference of data
        ref_ratio: ratio in loss of reference data.
        xdatfile: structure data file name.
        eigfile: eigenvalues data file name. 
        kpfile: kpoints data file name. 
        num_epoch: epoch number in training
        batch_size: batch size in traing.
        valid_size: validation batch size.
        start_learning_rate: 
        decay_rate: decay rate of learning rate.
        decay_step: decay learning rate every decay_step.
        savemodel: save model or not.
        save_epoch: save model every save_epoch.
        save_checkpoint: the checkpoint name.
        sort_strength: strength in soft sort.

        # no default keys:
        AtomType: atoms types in the whole structure. 
        ProjAtomType: the atoms which inclued in TB orbitals.
        ProjAnglrM: each ProjAtomType, the angular momentum used in TB.
        ValElec: number of valence electrons for each ProjAtomType.
        CutOff: cutoff for band.
        EnvCutOff: cutoff for local environment.
        NumEnv: the number of atoms inclued in local environment.
        Envnet: env embedding network.
        Envout: out2 of env embedding network.
        Bondnet: bond network
        onsite_net: on energy network.
        energy_window: energy_window for training.
        """

        self.istrain = True
        self.istest = False
        self.ispredict = False
        # 1 h_sk(1+nn) 2: hsk + nn.
        self.correction_mode = 1 
        self.SpinDeg = 2
        self.active_func = "tanh"
        self.train_data_path = './'
        self.prefix = 'set'
        self.valddir = "./valddir"
        self.withref = False
        self.refdir = './refdir'
        # self.testdir = './test'
        self.ref_ratio = 0.5
        self.xdatfile = 'xdat.traj'
        self.eigfile = 'eigs.npy'
        self.kpfile = 'kpoints.npy'
        self.num_epoch = 100
        self.batch_size = 1
        self.valid_size = 1
        self.start_learning_rate = 0.001
        self.decay_rate = 0.99
        self.decay_step = 2
        self.savemodel = True
        self.save_epoch = 4
        self.save_checkpoint = './checkpoint.pl'
        self.read_checkpoint = self.save_checkpoint
        self.sort_strength = [1, 0.01]
        self.corr_strength = [1, 1]
        self.use_E_win = True
        # w.r.t Fermi level.
        self.energy_max = 1.0
        self.use_I_win = False
        self.band_max = 1
        self.band_min = 0

        no_default_keys = ['AtomType',
                                'ProjAtomType',
                                'ProjAnglrM',
                                'ValElec',
                                'CutOff',
                                'EnvCutOff',
                                'NumEnv',
                                'Envnet',
                                'Envout',
                                'Bondnet',
                                'onsite_net']

        # check all the no default keys in input json file :

        for ikey in no_default_keys:
            if not ikey in self.input_json.keys():
                print('input json file must have ' + ikey)
                sys.exit()
        
        # read all the default keys in inpus json.
        for ikey in self.input_json.keys():
            # skip the key starting with '_' wich is comment.
            if not re.match('_', ikey):
                exec ('self.' + ikey + '= self.input_json[ikey]')


    def testparas(self):
        """ initial default parameters and user define parameters for NNTB .

        Attributes
        ----------
        istrain, istest, ispredict : used NN to train, test or predict.
        SpinDeg: the spin degeneracy of the band structure.
        active_func: activation function used in NN.
        xdatfile: structure data file name.
        eigfile: eigenvalues data file name. 
        kpfile: kpoints data file name. 
        num_epoch: epoch number in training
        batch_size: batch size in traing.
        valid_size: validation batch size.
        read_checkpoint: the checkpoint name.
        sort_strength: strength in soft sort.

        # no default keys:
        AtomType: atoms types in the whole structure. 
        ProjAtomType: the atoms which inclued in TB orbitals.
        ProjAnglrM: each ProjAtomType, the angular momentum used in TB.
        ValElec: number of valence electrons for each ProjAtomType.
        CutOff: cutoff for band.
        EnvCutOff: cutoff for local environment.
        NumEnv: the number of atoms inclued in local environment.
        Envnet: env embedding network.
        Envout: out2 of env embedding network.
        Bondnet: bond network
        onsite_net: on energy network.
        energy_window: energy_window for training.
        """

        self.istrain = False
        self.istest = True
        self.ispredict = False
        # 1 h_sk(1+nn) 2: hsk + nn.
        self.correction_mode = 1 
        self.SpinDeg = 2
        self.active_func = "tanh"
        self.testdir = './test'
        self.xdatfile = 'xdat.traj'
        self.eigfile = 'eigs.npy'
        self.kpfile = 'kpoints.npy'
        self.num_epoch = 100
        self.batch_size = 1
        self.read_checkpoint = './checkpoint.pl'
        self.sort_strength = [1, 0.01]
        self.corr_strength = [1, 1]
        self.use_E_win = True
        # w.r.t Fermi level.
        self.energy_max = 1.0
        self.use_I_win = False
        self.band_max = 1
        self.band_min = 0

        no_default_keys = ['AtomType',
                                'ProjAtomType',
                                'ProjAnglrM',
                                'ValElec',
                                'CutOff',
                                'EnvCutOff',
                                'NumEnv',
                                'Envnet',
                                'Envout',
                                'Bondnet',
                                'onsite_net']

        # check all the no default keys in input json file :

        for ikey in no_default_keys:
            if not ikey in self.input_json.keys():
                print('input json file must have ' + ikey)
                sys.exit()
        
        # read all the default keys in inpus json.
        for ikey in self.input_json.keys():
            # skip the key starting with '_' wich is comment.
            if not re.match('_', ikey):
                exec ('self.' + ikey + '= self.input_json[ikey]')

    def bandparas(self):
        """ initial default parameters and user define parameters for NNTB .

        Attributes
        ----------
        SpinDeg: the spin degeneracy of the band structure.
        read_checkpoint: the checkpoint name.
        nkpoints : total kpoints in the band kpath.
        band_range: energy range of band plot. default os None, and will be set to max and min of band.
        # no default keys:
        AtomType: atoms types in the whole structure. 
        ProjAtomType: the atoms which inclued in TB orbitals.
        ProjAnglrM: each ProjAtomType, the angular momentum used in TB.
        ValElec: number of valence electrons for each ProjAtomType.
        CutOff: cutoff for band.
        EnvCutOff: cutoff for local environment.
        NumEnv: the number of atoms inclued in local environment.
        Envnet: env embedding network.
        Envout: out2 of env embedding network.
        Bondnet: bond network
        onsite_net: on energy network.
        """
        # 1 h_sk(1+nn) 2: hsk + nn.
        self.SpinDeg = 2
        self.read_checkpoint = './checkpoint.pl'
        self.struct ='struct.vasp'
        self.format = 'vasp'
        self.band_range=None
        self.nkpoints = 120

        no_default_keys = ['AtomType',
                                'ProjAtomType',
                                'ProjAnglrM',
                                'ValElec',
                                'CutOff',
                                'EnvCutOff',
                                'NumEnv',
                                'Envnet',
                                'Envout',
                                'Bondnet',
                                'onsite_net',
                                'HighSymKps',
                                'KPATH']

        # check all the no default keys in input json file :

        for ikey in no_default_keys:
            if not ikey in self.input_json.keys():
                print('input json file must have ' + ikey)
                sys.exit()
        
        # read all the default keys in inpus json.
        for ikey in self.input_json.keys():
            # skip the key starting with '_' wich is comment.
            if not re.match('_', ikey):
                exec ('self.' + ikey + '= self.input_json[ikey]')

    def SKtbparas(self):
        """ initial default parameters e and user define parameters for sktb band structure calculations.
        
        Attributes
        ----------
        BZmesh: BZ sampling mesh.
        NKpLine: number of kpoints for each line in kpath.
        BandPlotRange: energy range for ploting band structure.
        """
        self.BZmesh = [1,1,1]
        self.NKpLine = 31
        no_default_keys = ['HighSymKps',
                            'KPATH',
                            'BandPlotRange']

        # check all the no default keys in input json file :
        for ikey in no_default_keys:
            if not ikey in self.input_json.keys():
                print('input json file must have ' + ikey)
                sys.exit()
        
        # read all the default keys in inpus json.
        for ikey in self.input_json.keys():
            # skip the key starting with '_' wich is comment.
            if not re.match('_', ikey):
                exec ('self.' + ikey + '= self.input_json[ikey]')
    

    def nntbnegfparas(self):
        """ initial default parameters e and user define parameters for negf calculations.
        
        Attributes
        ----------
        Contacts: The contacts in the whole device.
        ContactsPot: Bias potentials on the contacts.
        DeviceRegion: atom's indices in the central device region.
        ContactsRegions: atom's indices in contacts.
        PrinLayNunit: unit cell given in the device stucture. can be 1  or 2.
        """
        self.TBmodel = 'nntb'
        self.istrain = False
        self.istest = False
        self.ispredict = False
        self.correction_mode = 1
        self.SpinDeg = 2
        self.read_checkpoint = './checkpoint.pl'
        self.active_func = "tanh"

        self.Processors = 1
        self.SaveSurface = False
        self.SaveSelfEnergy = False
        self.CalDeviceDOS = False
        self.CalDevicePDOS = False
        self.CalTrans = False
        

        self.kmesh = [1,1,1]
        self.eta = 0.001
        self.max_iteration = 100
        self.epsilon = 1.0E-6
        self.Emin = -1
        self.Emax = 1
        self.NumE = 100
        self.use_E_win = True
        # w.r.t Fermi level.
        self.energy_max = 1.0
        self.use_I_win = False
        self.band_max = 1
        self.band_min = 0
        self.DeviceFermi = 0
        self.ContactFermi = 0

        no_default_keys=['AtomType',
                        'ProjAtomType',
                        'ProjAnglrM',
                        'ValElec',
                        'CutOff',
                        'EnvCutOff',
                        'NumEnv',
                        'Envnet',
                        'Envout',
                        'Bondnet',
                        'onsite_net',
                        'DeviceRegion',
                        'Contacts',
                        'ContactsRegions',
                        'ContactsPot',
                        'PrinLayNunit'
                        ]
        
        # check all the no default keys in input json file :
        for ikey in no_default_keys:
            if not ikey in self.input_json.keys():
                print('input json file must have ' + ikey)
                sys.exit()
        
        # read all the default keys in inpus json.
        for ikey in self.input_json.keys():
            # skip the key starting with '_' wich is comment.
            if not re.match('_', ikey):
                exec ('self.' + ikey + '= self.input_json[ikey]')


    def sktbnegfparas(self):
        """ initial default parameters e and user define parameters for negf calculations.
        
        Attributes
        ----------
        Contacts: The contacts in the whole device.
        ContactsPot: Bias potentials on the contacts.
        DeviceRegion: atom's indices in the central device region.
        ContactsRegions: atom's indices in contacts.
        PrinLayNunit: unit cell given in the device stucture. can be 1  or 2.
        """
        self.TBmodel = 'sktb'
        self.istrain = False
        self.istest = False
        self.ispredict = False
        self.correction_mode = 1
        self.SpinDeg = 2
        self.read_checkpoint = './checkpoint.pl'
        self.active_func = "tanh"

        self.Processors = 1
        self.SaveSurface = False
        self.SaveSelfEnergy = False
        self.CalDeviceDOS = False
        self.CalDevicePDOS = False
        self.CalTrans = False
        

        self.kmesh = [1,1,1]
        self.eta = 0.001
        self.max_iteration = 100
        self.epsilon = 1.0E-6
        self.Emin = -1
        self.Emax = 1
        self.NumE = 100
        self.use_E_win = True
        # w.r.t Fermi level.
        self.energy_max = 1.0
        self.use_I_win = False
        self.band_max = 1
        self.band_min = 0
        self.DeviceFermi = 0
        self.ContactFermi = 0

        no_default_keys=['AtomType',
                        'ProjAtomType',
                        'ProjAnglrM',
                        'ValElec',
                        'CutOff',
                        'EnvCutOff',
                        'NumEnv',
                        'Envnet',
                        'Envout',
                        'Bondnet',
                        'onsite_net',
                        'DeviceRegion',
                        'Contacts',
                        'ContactsRegions',
                        'ContactsPot',
                        'PrinLayNunit'
                        ]
        
        # check all the no default keys in input json file :
        for ikey in no_default_keys:
            if not ikey in self.input_json.keys():
                print('input json file must have ' + ikey)
                sys.exit()
        
        # read all the default keys in inpus json.
        for ikey in self.input_json.keys():
            # skip the key starting with '_' wich is comment.
            if not re.match('_', ikey):
                exec ('self.' + ikey + '= self.input_json[ikey]')






