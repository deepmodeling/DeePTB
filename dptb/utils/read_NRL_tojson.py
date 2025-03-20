import re
import os
import numpy as np
import scipy.constants as con
import json
import logging
log = logging.getLogger(__name__)
from dptb.utils.constants import Bohr2Ang, Ryd2eV, Harte2eV

#bohr2ang = con.value('Bohr radius')*1e10
#harte2ev = con.value('Hartree energy in eV')
#ryd2ev = con.value('Rydberg constant times hc in eV')
bohr2ang = Bohr2Ang
ryd2ev = Ryd2eV
harte2ev = Harte2eV

def read_nrl_file(nrl_file):
    with open(nrl_file, 'r') as f:
        tbpar = f.readlines()
    overlap_type =  int(re.findall(r"\d+\.?\d*",tbpar[0].split()[0])[0])
    log.info(f"System: " + tbpar[1].strip('\n'))
    n_atom_type = int(tbpar[2].split()[0])
    log.info(f'num of atom type: {n_atom_type}')

    Rcut, Lscreen = [float(i) for i in tbpar[3].split()[:2]]
    norbs_peratom = int(tbpar[4].split()[0])
    atomic_weight = [float(i) for i in tbpar[5].split()[:n_atom_type]]
    ncc_s, ncc_p, ncc_d = [float(i) for i in tbpar[6].split()[:3]]
    
    # get onsite parameters:
    if re.findall(r"lambda",tbpar[7]):
        lambda_value = float(tbpar[7].split()[0])
    iline = 8
    E_s,E_p,E_d = [],[],[]
    E_t2g, E_eg = [],[]
    while iline < len(tbpar):
        if re.findall(r"_s",tbpar[iline]):
            _value = float(tbpar[iline].split()[0])
            E_s.append(_value)
        if re.findall(r"_p",tbpar[iline]):
            _value = float(tbpar[iline].split()[0])
            E_p.append(_value)
        if re.findall(r"_d",tbpar[iline]):
            _value = float(tbpar[iline].split()[0])
            E_d.append(_value)
        if re.findall(r"_t2g",tbpar[iline]):
            _value = float(tbpar[iline].split()[0])
            E_t2g.append(_value)
        if re.findall(r"_eg",tbpar[iline]):
            _value = float(tbpar[iline].split()[0])
            E_eg.append(_value)

        if re.findall(r"sigma",tbpar[iline]):
            break
        iline += 1
    E_s,E_p = np.asarray(E_s), np.asarray(E_p)
    E_d,E_t2g,E_eg = np.asarray(E_d), np.asarray(E_t2g), np.asarray(E_eg)
    if (np.abs(E_t2g - E_eg) < 1E-6).all() :
        E_d = E_t2g

    onsite_para = {}
    onsite_para['s'] = E_s.tolist()
    onsite_para['p'] = E_p.tolist()
    onsite_para['d'] = E_d.tolist()

    # get hopping and overlap parameters:
    # 0-sigma, 1-pi, 2-delta
    bond_type = ['ss_sigma','sp_sigma','sd_sigma','pp_sigma','pp_pi','pd_sigma','pd_pi','dd_sigma','dd_pi','dd_delta']
    hop_para = {}
    overlap_para = {}
    for ibond in bond_type:
        hop_para[ibond] = []
        overlap_para[ibond] = []

    for iline in range(1,len(tbpar)):
        if re.findall(r"Hamiltonian",tbpar[iline]) or re.findall(r"Ham.",tbpar[iline]):
            for ibond in bond_type:
                orbs, btype = ibond.split('_')
                if re.findall(orbs,tbpar[iline]) and re.findall(btype,tbpar[iline]):
                    _value = float(tbpar[iline].split()[0])
                    hop_para[ibond].append(_value)
                    break
                else:
                    continue
        elif re.findall(r"Overlap",tbpar[iline]) or re.findall(r"Ovr.",tbpar[iline]):
            for ibond in bond_type:
                orbs, btype = ibond.split('_')
                if re.findall(orbs,tbpar[iline]) and re.findall(btype,tbpar[iline]):
                    _value = float(tbpar[iline].split()[0])
                    overlap_para[ibond].append(_value)
                    break
                else:
                    continue
        else:
            continue

    NRL_data = {}
    NRL_data['overlap_type'] = overlap_type
    NRL_data['lambda'] = lambda_value
    NRL_data['Rcut'] = Rcut
    NRL_data['Lscreen'] = Lscreen
    NRL_data['norbs_peratom'] = norbs_peratom
    NRL_data['n_atom_type'] = n_atom_type
    NRL_data['onsite_para'] = onsite_para
    NRL_data['hop_para'] = hop_para
    NRL_data['overlap_para'] = overlap_para

    return NRL_data



def transfer_bohr_Ang_Ry_eV(nrl_tb:dict, overlap_type:int):
    for key in nrl_tb['onsite']:
        for i in range(4):
            nrl_tb['onsite'][key][i] = nrl_tb['onsite'][key][i]*ryd2ev

    factor = [ryd2ev, ryd2ev/bohr2ang, ryd2ev/bohr2ang**2, 1.0/np.sqrt(bohr2ang)]
    for key in nrl_tb['hopping']:
        for i in range(4):
            nrl_tb['hopping'][key][i] = nrl_tb['hopping'][key][i]*factor[i]
    if overlap_type == 0:
        factor_s = [1.0, 1.0/bohr2ang, 1.0/bohr2ang**2, 1.0/np.sqrt(bohr2ang)]
    elif overlap_type ==1:
        factor_s = [1.0/bohr2ang, 1.0/bohr2ang**2, 1.0/bohr2ang**3, 1.0/np.sqrt(bohr2ang)]

    for key in nrl_tb['overlap']:
        for i in range(4):
            nrl_tb['overlap'][key][i] = nrl_tb['overlap'][key][i]*factor_s[i]
    return nrl_tb


def nrl2dptb(input, NRL_data):
    
    with open(input,'r') as f:
        input_nrl = json.load(f)
    
    atom_types = list(input_nrl['common_options']['basis'].keys())
    
    if  len(atom_types) != NRL_data['n_atom_type']:
        log.error('Error! the number of atom types in input file is not consistent with the NRL file.')
        exit(1)
    if len(atom_types) !=1:
        "now this module only support the elementary NRL para."
        log.error('Error! this module only support the elementary NRL para. for now, the number of atom types in input file should be 1.')
        exit(1)
    atom_symbol = atom_types[0]
    out_orb= input_nrl['common_options']['basis'][atom_symbol]

    if not input_nrl['common_options'].get('overlap',False):
        log.warning("Warning! the overlap in common_options  must be true, we update it to True.")
        input_nrl['common_options']['overlap'] = True

    orb_map = {}
    for iorb in out_orb:
        if re.findall(r"s",iorb):
            assert not 's' in orb_map.keys()
            orb_map['s'] = iorb
        elif re.findall(r"p",iorb):
            assert not 'p' in orb_map.keys()
            orb_map['p'] = iorb
        elif re.findall(r"d",iorb):
            assert not 'd' in orb_map.keys()
            orb_map['d'] = iorb
        else:
            log.error('Error!, the orbital type is not supported, please check the input file. only s, p, d are supported!')
            exit(1)
    # inverse key and value of orb_map:
    orb_map_r = {v: k for k, v in orb_map.items()}
    # # 0-sigma, 1-pi, 2-delta
    bond_map = {'sigma':0,'pi':1,'delta':2}

    nrl_out = {}

    onsite_para = NRL_data['onsite_para']
    hop_para = NRL_data['hop_para']
    overlap_para = NRL_data['overlap_para']
    overlap_type = NRL_data['overlap_type']
    lambda_value = NRL_data['lambda'] 
    Rcut = NRL_data['Rcut'] 
    Lscreen = NRL_data['Lscreen'] 
    if lambda_value < 0 :
        assert overlap_type ==1, 'For lambda, the overlap formula in  NRL should be new one.'
        lambda_value = abs(lambda_value)

    nrl_out['onsite'] = {}
    for iorb in out_orb:
        iiorb = orb_map_r[iorb]
        nrl_out['onsite'][f'{atom_symbol}-{iorb}-0'] = onsite_para[iiorb]

    nrl_out['hopping'] = {}
    for ikey in hop_para.keys():
        oioj, btype = ikey.split('_')
        oi, oj = oioj
        if oi in orb_map.keys() and oj in orb_map.keys():
            ooi, ooj = orb_map[oi],orb_map[oj]
            dptb_key = f"{atom_symbol}-{atom_symbol}-{ooi}-{ooj}-{bond_map[btype]}"
            nrl_out['hopping'][dptb_key] = hop_para[ikey]
        else:
            continue

    nrl_out['overlap'] = {}
    for ikey in overlap_para.keys():
        oioj, btype = ikey.split('_')
        oi, oj = oioj
        if oi in orb_map.keys() and oj in orb_map.keys():
            ooi, ooj = orb_map[oi],orb_map[oj]
            dptb_key = f"{atom_symbol}-{atom_symbol}-{ooi}-{ooj}-{bond_map[btype]}"
            nrl_out['overlap'][dptb_key] = overlap_para[ikey]
        else:
            continue

    nrl_tb = transfer_bohr_Ang_Ry_eV(nrl_out,overlap_type=overlap_type)
    
    if input_nrl.get('model_options',{}) == {}:
        input_nrl['model_options'] = {}
    if input_nrl['model_options'].get('nnsk',{}) != {}:
        log.info('Warning! the nnsk in model_options is not empty, will overwrite the nnsk.')
    else:
        input_nrl['model_options']['nnsk'] = {}

    freeze = input_nrl['model_options']['nnsk'].get('freeze',False)
    std = input_nrl['model_options']['nnsk'].get('std',0.01)
    push = input_nrl['model_options']['nnsk'].get('push',False)
    
    input_nrl['model_options']['nnsk'].update(
        {"onsite": {
            "method": "NRL",
            "rs": Rcut * bohr2ang,
            "w": Lscreen * bohr2ang,
            "lda": lambda_value/np.sqrt(bohr2ang)
        },
        "hopping": {
            "method": f'NRL{overlap_type}',
            "rs": Rcut * bohr2ang,
            "w": Lscreen * bohr2ang
        },
        "soc":{},
        "freeze": freeze,
        "std": std,
        "push": push}
    )

   ### models 
    model = {}
    model['version'] =2
    model['unit'] = 'eV'
    model['common_options'] ={
        "basis": {
            atom_symbol: out_orb
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": True    
        }
    model['model_options'] = {
        "nnsk": {
            "onsite": {
                "method": "NRL",
                "rs": Rcut * bohr2ang,
                "w": Lscreen * bohr2ang,
                "lda": lambda_value/np.sqrt(bohr2ang)
            },
            "hopping": {
                "method": f'NRL{overlap_type}',
                "rs": Rcut * bohr2ang,
                "w": Lscreen * bohr2ang
            },
            "soc":{},
            "freeze": freeze,
            "std": std,
            "push": push
        }
    }
    model['model_params'] = nrl_tb

    log.info(f"The NRL parameters are successfully transferred to DPTB.")
    log.info(f"Please check the input json file and modify the parameters if necessary.")
    log.info(f"!!!!!")
    log.info(f"Important: recommend to set the info.json r_max and oer_max >= {Rcut*bohr2ang} Ang")
    log.info(f"!!!!!")   

    return input_nrl, model


def save2json(input_dict:dict, nrl_tb_dict:dict, outdir='./out'):
    if os.path.exists(outdir):
        log.warning('Warning! the outdir exists, will overwrite the file.')
    else:
        os.makedirs(outdir)

    with open(f'{outdir}/input_nrl_auto.json','w') as f:
        json.dump(input_dict,f,indent=4)
    
    with open(f'{outdir}/nrl_ckpt.json','w') as f:
        json.dump(nrl_tb_dict,f,indent=4)