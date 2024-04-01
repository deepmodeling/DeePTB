import os
import pytest
import ase
from ase import Atoms
import numpy as np
from dptb.utils.tools import get_neighbours
from dptb.entrypoints.bond import bond
import io


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)

#outdir = f"{root_directory}/dptb/tests/data/out"
#if not os.path.exists(outdir):
#   os.makedirs(outdir)

def test_get_neighbours():
    # Create a test atom
    atom = ase.Atom("Cu", position=[0, 0, 0])
    atoms = ase.Atoms([atom])
    # Test case 1: No neighbors within cutoff distance
    cutoff = 1.0
    expected_result = {}
    assert get_neighbours(atoms, cutoff) == expected_result

    # Test case 2: Single neighbor within cutoff distance
    atom2 = ase.Atom("Cu", position=[0, 0, 1])
    atom3 = ase.Atom("Cu", position=[0, 1, 0])
    atom4 = ase.Atom("Cu", position=[1, 0, 0])
    atoms = ase.Atoms([atom, atom2, atom3, atom4])
    cutoff = 2.0
    expected_result = {"Cu-Cu": [1.0,np.sqrt(2)]}
    assert get_neighbours(atoms, cutoff) == expected_result

    # Test case 3: Multiple neighbors with different bond lengths
    atom5 = ase.Atom("Cu", position=[1, 1, 1])
    atoms.append(atom5)
    cutoff = 2.0
    expected_result = {"Cu-Cu": [1.0, np.sqrt(2), np.sqrt(3)]}
    assert get_neighbours(atoms, cutoff) == expected_result

    # Test case 4: Threshold for bond length difference
    cutoff = 2.0
    thr = 1.0
    expected_result = {"Cu-Cu": [1.0]}
    assert get_neighbours(atoms, cutoff, thr) == expected_result

# def test_bond_cmd(root_directory):

def test_bond_empty_structure(root_directory):
    outdir = f"{root_directory}/dptb/tests/data/out"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    struct = f"{outdir}/empty.xyz"
    accuracy = 0.01
    cutoff = 1.0
    log_level = 1
    log_path = None
    with pytest.raises(FileNotFoundError):
        bond(struct, accuracy, cutoff, log_level, log_path)

def test_bond_single_atom(root_directory):
    outdir = f"{root_directory}/dptb/tests/data/out"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    atoms = Atoms("Cu", positions=[(0, 0, 0)])
    struct = f"{outdir}/single_atom.xyz"
    accuracy = 0.01
    cutoff = 1.0
    log_level = 1
    log_path = None

    atoms.write(struct)
    out = bond(struct, accuracy, cutoff, log_level, log_path)
    assert out == ' Bond Type\n------------\n'
 
def test_bond_multiple_atoms(root_directory):
    outdir = f"{root_directory}/dptb/tests/data/out"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    atoms = Atoms("Cu4", positions=[(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)])
    struct = f"{outdir}/multiple_atoms.xyz"
    accuracy = 0.01
    cutoff = 2.0
    log_level = 1
    log_path = None

    atoms.write(struct)
    out = bond(struct, accuracy, cutoff, log_level, log_path)
    assert out == ' Bond Type         1         2\n------------------------------------\n     Cu-Cu      1.00      1.41\n'


def test_bond_hBN(root_directory):
    struct = (f"{root_directory}/dptb/tests/data/hBN/hBN.vasp")
    accuracy = 0.01
    cutoff = 5.0
    log_level = 1
    log_path = None

    out = bond(struct, accuracy, cutoff, log_level, log_path)
    assert out == ' Bond Type         1         2         3\n------------------------------------------------\n       N-N      2.50      4.34\n       N-B      1.45      2.89      3.82\n       B-B      2.50      4.34\n'
