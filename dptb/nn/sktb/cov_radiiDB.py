# Covalent radii in pm from analysis of the Cambridge Structural Database, which contains about 1,030,000 crystal structures[4]
# The data is from the https://en.wikipedia.org/wiki/Covalent_radius
# For homonuclear A–A bonds, Linus Pauling took the covalent radius to be half the single-bond length in the element, e.g. R(H–H, in H2) = 74.14 pm so rcov(H) = 37.07 pm: 
# in practice, it is usual to obtain an average value from a variety of covalent compounds, although the difference is usually small.
# Therefore the bandlength A-B we can use the sum of the covalent radii of A and B.

# unit pm.
Covalent_radii_pm = {
    'H': 31,                                                                                                                                                                                                                                                    'He': 28, 
    'Li': 128,    'Be': 96,                                                                                                                                                            'B': 84,       'C': 76,       'N': 71,       'O': 66,       'F': 57,     'Ne': 58, 
    'Na': 166,    'Mg': 141,                                                                                                                                                          'Al': 121,     'Si': 111,      'P': 107,      'S': 105,     'Cl': 102,    'Ar': 106,
     'K': 203,    'Ca': 176,     'Sc': 170,     'Ti': 160,      'V': 153,    'Cr': 139,     'Mn': 139,     'Fe': 132,     'Co': 126,     'Ni': 124,     'Cu': 132,     'Zn': 122,     'Ga': 122,     'Ge': 120,     'As': 119,     'Se': 120,     'Br': 120,    'Kr': 116,
    'Rb': 220,    'Sr': 195,      'Y': 190,     'Zr': 175,     'Nb': 164,    'Mo': 154,     'Tc': 147,     'Ru': 146,     'Rh': 142,     'Pd': 139,     'Ag': 145,     'Cd': 144,     'In': 142,     'Sn': 139,     'Sb': 139,     'Te': 138,      'I': 139,    'Xe': 140,
    'Cs': 244,    'Ba': 215,     'Lu': 187,     'Hf': 175,     'Ta': 170,     'W': 162,     'Re': 151,     'Os': 144,     'Ir': 141,     'Pt': 136,     'Au': 136,     'Hg': 132,     'Tl': 145,     'Pb': 146,     'Bi': 148,     'Po': 140,     'At': 150,    'Rn': 150,
    'Fr': 260,    'Ra': 221,
    'La': 207,    'Ce': 204,    'Pr': 203,    'Nd': 201,    'Pm': 199,    'Sm': 198,    'Eu': 198,    'Gd': 196,    'Tb': 194,    'Dy': 192,    'Ho': 192,    'Er': 189,    'Tm': 190,    'Yb': 187,
    'Ac': 215,    'Th': 206,    'Pa': 200,     'U': 196,    'Np': 190,    'Pu': 187,    'Am': 180,    'Cm': 169
}

# unit AA
Covalent_radii = {}
for k, v in Covalent_radii_pm.items():
    if v is not None:
        Covalent_radii[k] = v * 0.01
    else:
        Covalent_radii[k] = v

# To constract the bond length, we can use the sum of the covalent radii of A and B.


from dptb.utils.constants import atomic_num_dict
import torch

# unit. \AA. 
R_cov_list = torch.zeros(int(max(atomic_num_dict.values()))) - 100
for k, v in Covalent_radii_pm.items():
    R_cov_list[atomic_num_dict[k]-1] = v * 0.01