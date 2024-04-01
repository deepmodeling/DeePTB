# Onsite energies database, loaded from GAPW lda potentials. stored as 
# A dictionary of dictionaries. The first dictionary is the element name, and the
# second dictionary is the orbital name. The orbital name is the key, and the value is the onsite  energy.


#
# Contains the elements as follows:

#    AtomSymbol=[
#     'H',                                                                                                  'He', 
#     'Li', 'Be',                                                             'B',  'C',  'N',  'O',  'F',  'Ne', 
#     'Na', 'Mg',                                                             'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
#     'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
#     'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo',     , 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe', 
#     'Cs', 'Ba',       'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',             'Rn'
#     ]

element = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
 "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
 "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Lu", "Hf", "Ta",
 "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Ra", "Th"]

r0 = [1.6,1.4,5.0,3.4,3.0,3.2,3.4,3.1,2.7,3.2,5.9,5.0,5.9,4.4,4.0,3.9,
 3.8,4.5,6.5,4.9,5.1,4.2,4.3,4.7,3.6,3.7,3.3,3.7,5.2,4.6,5.9,4.5,4.4,
 4.5,4.3,4.8,9.1,6.9,5.7,5.2,5.2,4.3,4.1,4.1,4.0,4.4,6.5,5.4,4.8,4.7,
 5.2,5.2,6.2,5.2,10.6,7.7,7.4,5.9,5.2,4.8,4.2,4.2,4.0,3.9,3.8,4.8,6.7,
 7.3,5.7,5.8,5.5,7.0,6.2]

bond_length = {
    'H': 1.6, 'He': 1.4, 'Li': 5.0, 'Be': 3.4, 'B': 3.0, 'C': 3.2, 'N': 3.4, 'O': 3.1, 'F': 2.7, 'Ne': 3.2, 'Na': 5.9, 'Mg': 5.0, 
    'Al': 5.9, 'Si': 4.4, 'P': 4.0, 'S': 3.9, 'Cl': 3.8, 'Ar': 4.5, 'K': 6.5, 'Ca': 4.9, 'Sc': 5.1, 'Ti': 4.2, 'V': 4.3, 'Cr': 4.7, 
    'Mn': 3.6, 'Fe': 3.7, 'Co': 3.3, 'Ni': 3.7, 'Cu': 5.2, 'Zn': 4.6, 'Ga': 5.9, 'Ge': 4.5, 'As': 4.4, 'Se': 4.5, 'Br': 4.3, 'Kr': 4.8, 
    'Rb': 9.1, 'Sr': 6.9, 'Y': 5.7, 'Zr': 5.2, 'Nb': 5.2, 'Mo': 4.3, 'Tc': 4.1, 'Ru': 4.1, 'Rh': 4.0, 'Pd': 4.4, 'Ag': 6.5, 'Cd': 5.4, 
    'In': 4.8, 'Sn': 4.7, 'Sb': 5.2, 'Te': 5.2, 'I': 6.2, 'Xe': 5.2, 'Cs': 10.6, 'Ba': 7.7, 'La': 7.4, 'Lu': 5.9, 'Hf': 5.2, 'Ta': 4.8, 
    'W': 4.2, 'Re': 4.2, 'Os': 4.0, 'Ir': 3.9, 'Pt': 3.8, 'Au': 4.8, 'Hg': 6.7, 'Tl': 7.3, 'Pb': 5.7, 'Bi': 5.8, 'Po': 5.5, 'Ra': 7.0, 
    'Th': 6.2}
