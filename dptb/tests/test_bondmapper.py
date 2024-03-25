import pytest
import torch
import ase.data
from dptb.data.transforms import BondMapper
from dptb.data import AtomicData, AtomicDataDict

def test_bond_mapper_superinit():
    # 测试使用化学符号初始化
    chemical_symbols = ['H', 'C', 'O']
    mapper = BondMapper(chemical_symbols=chemical_symbols)
    assert mapper.num_types == 3
    assert mapper.type_names == chemical_symbols
    assert mapper.has_bond
    assert mapper.has_chemical_symbols
    assert mapper.chemical_symbol_to_type == {'H': 0, 'C': 1, 'O': 2}

    chemical_symbols = ['C', 'H', 'O']
    mapper = BondMapper(chemical_symbols=chemical_symbols)
    assert mapper.num_types == 3
    assert mapper.type_names == ['H', 'C', 'O'] # 使用chemical_symbols初始化，按照元素周期表的顺序排序
    assert mapper.chemical_symbol_to_type == {'C': 1, 'H': 0, 'O': 2}
    
    # 测试使用 chemical_symbols_to_type 初始化
    chemical_symbols_to_type = {'H': 0, 'C': 1, 'O': 2}
    mapper = BondMapper(chemical_symbols_to_type=chemical_symbols_to_type)
    assert mapper.num_types == 3
    assert mapper.has_bond
    assert mapper.has_chemical_symbols
    assert mapper.type_names == ['H', 'C', 'O']
    assert mapper.chemical_symbol_to_type == {'H': 0, 'C': 1, 'O': 2}

    chemical_symbols_to_type = {'H': 1, 'C': 2, 'O': 0}
    mapper = BondMapper(chemical_symbols_to_type=chemical_symbols_to_type)
    # 使用 chemical_symbols_to_type 初始化，因为已经给了type的编号，因此不会再次进行排序
    assert mapper.chemical_symbol_to_type == {'H': 1, 'C': 2, 'O': 0}
    # 使用 chemical_symbols_to_type 初始化，type_names会chemical_symbols_to_type的value排序
    assert mapper.type_names == ['O', 'H', 'C']

    # 测试无效输入 -1
    chemical_symbols = ['H', 'C', 'O']
    chemical_symbol_to_type = {'H': 0, 'C': 1, 'O': 2}
    with pytest.raises(ValueError):
        BondMapper(chemical_symbols=chemical_symbols,chemical_symbols_to_type=chemical_symbol_to_type)

    # 测试无效输入 -2
    with pytest.raises(ValueError):
        BondMapper()

def test_bond_mapper_init():
    # 测试使用化学符号初始化
    chemical_symbols = ['H', 'C', 'O']
    mapper = BondMapper(chemical_symbols=chemical_symbols)
    assert mapper.has_bond
    assert mapper.bond_types == ['H-H', 'H-C', 'H-O', 'C-H', 'C-C', 'C-O', 'O-H', 'O-C', 'O-O']
    assert mapper.reduced_bond_types == ['H-H', 'H-C', 'H-O', 'C-C', 'C-O', 'O-O']

    assert mapper.bond_to_type == {'H-H': 0,
                                        'H-C': 1,
                                        'H-O': 2,
                                        'C-H': 3,
                                        'C-C': 4,
                                        'C-O': 5,
                                        'O-H': 6,
                                        'O-C': 7,
                                        'O-O': 8}
    
    assert mapper.reduced_bond_to_type == {'H-H': 0, 
                                           'H-C': 1, 
                                           'H-O': 2, 
                                           'C-C': 3, 
                                           'C-O': 4, 
                                           'O-O': 5}
    
    assert mapper.type_to_reduced_bond == {0: 'H-H', 1: 'H-C', 2: 'H-O', 3: 'C-C', 4: 'C-O', 5: 'O-O'}

def test_bond_mapper_transform_atom():
    chemical_symbols = ['H', 'He', 'C']
    mapper = BondMapper(chemical_symbols=chemical_symbols)

    atomic_numbers = torch.tensor([1, 2, 6])  # H, He, Li
    types = mapper.transform_atom(atomic_numbers)
    assert torch.all(types == torch.tensor([0, 1, 2]))

    # 测试无效原子编号 - 1
    invalid_atomic_numbers = torch.tensor([1, 2, 5])
    with pytest.raises(ValueError):
        mapper.transform_atom(invalid_atomic_numbers)

    # 测试无效原子编号 - 2
    invalid_atomic_numbers = torch.tensor([1, 2, 7])
    with pytest.raises(ValueError):
        mapper.transform_atom(invalid_atomic_numbers)

def test_bond_mapper_untransform_atom():
    chemical_symbols = ['H', 'He', 'C']
    mapper = BondMapper(chemical_symbols=chemical_symbols)

    types = torch.tensor([0, 1, 2])
    atomic_numbers = mapper.untransform_atom(types)
    assert torch.all(atomic_numbers == torch.tensor([1, 2, 6]))

def test_bond_mapper_transform_bond():
    atomic_num_dict= {1: 'H', 6: 'C', 8: 'O'}
    chemical_symbols = ['H', 'C', 'O']
    mapper = BondMapper(chemical_symbols=chemical_symbols)
    ilist = []
    jlist = []
    bondtypelist = []
    for ii in [1, 6, 8]:
        for jj in [1, 6, 8]:
            ilist.append(ii)
            jlist.append(jj)
            bond = f'{atomic_num_dict[ii]}-{atomic_num_dict[jj]}'
            bondtypelist.append(mapper.bond_to_type[bond])
            bondtype = mapper.transform_bond(torch.tensor([ii]), torch.tensor([jj]))
            assert bondtype ==  mapper.bond_to_type[bond]
    
    bondtypes = mapper.transform_bond(torch.tensor(ilist), torch.tensor(jlist))
    assert torch.all(bondtypes == torch.tensor(bondtypelist))

    # 测试无效原子编号
    invalid_atomic_numbers = torch.tensor([1, 6, 9])

    with pytest.raises(ValueError):
        mapper.transform_bond(invalid_atomic_numbers, invalid_atomic_numbers)

    invalid_atomic_numbers = torch.tensor([1, 6, 7])
    with pytest.raises(ValueError):
        mapper.transform_bond(invalid_atomic_numbers[:2], invalid_atomic_numbers[1:])

def test_bond_mapper_untransform_bond():
    atomic_num_dict= {1: 'H', 6: 'C', 8: 'O'}
    chemical_symbols = ['H', 'C', 'O']
    mapper = BondMapper(chemical_symbols=chemical_symbols)

    bond_pairs = []
    bondtypelist = []
    for ii in [1, 6, 8]:
        for jj in [1, 6, 8]:
            bond_pairs.append([ii,jj])
            bond = f'{atomic_num_dict[ii]}-{atomic_num_dict[jj]}'
            bondtypelist.append(mapper.bond_to_type[bond])
            bondtype = torch.tensor(mapper.bond_to_type[bond])
            atomic_numbers_pairs = mapper.untransform_bond(bondtype)
            assert (atomic_numbers_pairs ==  torch.tensor([[ii, jj]])).all()

    
    bondtypes = mapper.untransform_bond(torch.tensor(bondtypelist))
    assert (bondtypes == torch.tensor(bond_pairs)).all()


def test_bond_mapper_transform_reduced_bond():
    atomic_num_dict= {1: 'H', 6: 'C', 8: 'O'}
    chemical_symbols = ['H', 'C', 'O']
    mapper = BondMapper(chemical_symbols=chemical_symbols)
    ilist = []
    jlist = []
    bondtypelist = []
    for ii in [1, 6, 8]:
        for jj in [1, 6, 8]:
            if ii > jj:
                continue
            ilist.append(ii)
            jlist.append(jj)
            bond = f'{atomic_num_dict[ii]}-{atomic_num_dict[jj]}'
            bondtypelist.append(mapper.reduced_bond_to_type[bond])
            bondtype = mapper.transform_reduced_bond(torch.tensor([ii]), torch.tensor([jj]))
            assert bondtype ==  mapper.reduced_bond_to_type[bond]
    
    bondtypes = mapper.transform_reduced_bond(torch.tensor(ilist), torch.tensor(jlist))
    assert torch.all(bondtypes == torch.tensor(bondtypelist))

    with pytest.raises(ValueError) as excinfo:
        mapper.transform_reduced_bond(torch.tensor([1, 9, 6]), torch.tensor([1, 6, 9]))
    assert "iatomic_numbers[i] should <= jatomic_numbers[i]" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        mapper.transform_reduced_bond(torch.tensor([1, 6]), torch.tensor([1, 9]))
    assert "Data included atomic numbers" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        mapper.transform_reduced_bond(torch.tensor([1, 5]), torch.tensor([1, 6]))
    assert "Data included atomic numbers" in str(excinfo.value)


def test_bond_mapper_untransform_reduced_bond():
    atomic_num_dict= {1: 'H', 6: 'C', 8: 'O'}
    chemical_symbols = ['H', 'C', 'O']
    mapper = BondMapper(chemical_symbols=chemical_symbols)

    bond_pairs = []
    bondtypelist = []
    for ii in [1, 6, 8]:
        for jj in [1, 6, 8]:
            if ii > jj:
                continue
            bond_pairs.append([ii,jj])
            bond = f'{atomic_num_dict[ii]}-{atomic_num_dict[jj]}'
            bondtypelist.append(mapper.reduced_bond_to_type[bond])
            bondtype = torch.tensor(mapper.reduced_bond_to_type[bond])
            atomic_numbers_pairs = mapper.untransform_reduced_bond(bondtype)
            assert (atomic_numbers_pairs ==  torch.tensor([[ii, jj]])).all()

    
    bondtypes = mapper.untransform_reduced_bond(torch.tensor(bondtypelist))
    assert (bondtypes == torch.tensor(bond_pairs)).all()


def test_bond_mapper_call():
    chemical_symbols = ['H', 'C', 'O']
    mapper = BondMapper(chemical_symbols=chemical_symbols)

    atomic_numbers = torch.tensor([1, 6, 8])
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 2]])
    data = {
        AtomicDataDict.ATOMIC_NUMBERS_KEY: atomic_numbers,
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
    }

    transformed_data = mapper(data)
    assert AtomicDataDict.ATOM_TYPE_KEY in transformed_data
    assert AtomicDataDict.EDGE_TYPE_KEY in transformed_data

    atom_types = transformed_data[AtomicDataDict.ATOM_TYPE_KEY]
    edge_types = transformed_data[AtomicDataDict.EDGE_TYPE_KEY]

    expected_atom_types = torch.tensor([0, 1, 2])
    expected_edge_types = torch.tensor([1, 5, 5])

    assert torch.all(atom_types == expected_atom_types)
    assert torch.all(edge_types == expected_edge_types)