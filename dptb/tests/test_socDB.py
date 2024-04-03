from dptb.nn.sktb.socDB import soc_strength_database
import numpy as np
import re

def test_socDB():
    assert isinstance(soc_strength_database, dict)
    AtomSymbol=[
     'H',                                                                                                  'He', 
     'Li', 'Be',                                                             'B',  'C',  'N',  'O',  'F',  'Ne', 
     'Na', 'Mg',                                                             'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
     'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
     'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe', 
     'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
     ]

    for ia in soc_strength_database:
        assert ia in AtomSymbol, "the atom symbol is not in the periodic table"
        assert isinstance(soc_strength_database[ia], dict)

    for ia in AtomSymbol:
        if ia in ['Tc', 'La','Po', 'At']:
            continue
        else:
            assert ia in soc_strength_database, f"the atom {ia} should be in the onsite_energy_database"
    
    for ia in np.random.choice(list(soc_strength_database.keys()),5):
        orbs = list(soc_strength_database[ia].keys())
        for iorb in orbs:
            isinstance(iorb, str)
            assert len(re.findall(r'[0-9]+',iorb)) in [0,1]
            assert len(re.findall(r'[a-z]+',iorb)) in [1]
            assert len(re.findall(r'[A-Z]+',iorb)) in [0]

            if len(re.findall(r'[0-9]+',iorb)) == 1:
                nshell_str = re.findall(r'[0-9]+',iorb)[0]
                nlorb_str = re.findall(r'[a-z]+',iorb)[0]
                assert iorb == nshell_str + nlorb_str
            else:
                nlorb_str = re.findall(r'[a-z]+',iorb)[0]
                assert iorb ==  nlorb_str + '*'



