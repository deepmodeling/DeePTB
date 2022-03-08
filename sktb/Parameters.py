import json

class Paras(object):
    def __init__(self,fp):
        input = json.load(fp)
        self.SKFilePath = input['SKFilePath']
        self.Separator  = input['Separator']
        self.Suffix     = input['Suffix']
        self.CutOff     = input['CutOff']
        self.StructFileName = input['StructFileName']
        self.StructFileFmt  = input['StructFileFmt']
        self.AtomType = input['AtomType']
        self.ProjAtomType   = input['ProjAtomType']
        self.ProjAnglrM     = input['ProjAnglrM']
        self.ValElec        = input['ValElec']
        self.Task           = input['Task']
        self.GUI            = input['GUI']
        
        if input['Task'].lower() == 'band':
            self.BZmesh = input['BZmesh']
            self.HighSymKps = input['HighSymKps']
            self.KPATHstr = input['KPATH']
            self.KPATH = []
            for ik in self.KPATHstr:
                self.KPATH.append(self.HighSymKps[ik])
            self.NKpLine = input['NKpLine']
            self.BandPlotRang = input['BandPlotRange']
        
        if input['Task'].lower() == 'negf':
            self.Processors      = input['Processors']
            self.DeviceRegion    = input['DeviceRegion']
            self.Contacts        = input['Contacts']
            self.ContactsRegions = input['ContactsRegions']
            #self.ContactsPot     = input['ContactsPot']
            self.PrinLayNunit    = input['PrinLayNunit']
            self.SaveSurface     = input['SaveSurface']
            self.SaveSelfEnergy  = input['SaveSelfEnergy']
            self.CalDeviceDOS    = input['CalDeviceDOS']
            self.SaveTrans       = input['SaveTrans']

            self.ShowContactBand = input['ShowContactBand']
            self.ContactsOnly    = input['ContactsOnly']
            if self.ShowContactBand in self.Contacts:
                self.HighSymKps = input['HighSymKps']
                self.KPATHstr = input['KPATH']
                self.KPATH = []
                for ik in self.KPATHstr:
                    self.KPATH.append(self.HighSymKps[ik])
                self.NKpLine = input['NKpLine']
            self.ShowContactDOS = input['ShowContactDOS']
            if self.ShowContactDOS:
                self.EmaxDOS = input['EmaxDOS']
                self.EminDOS = input['EminDOS']
                self.NEDOS   = input['NEDOS']
                self.Sigma   = input['Sigma']
            self.DOSKMesh = input['DOSKMesh']
            self.nkpoints = [1,  1, 1],
            self.kpstart  = input['kpstart']
            self.kpvec1   = input['kpvec1']
            self.kpvec2   = input['kpvec2']
            self.kpvec3   = input['kpvec3']
            self.Emin     = input['Emin']
            self.Emax     = input['Emax']
            self.NumE     = input['NumE']
            self.Eta      = input['Eta']


