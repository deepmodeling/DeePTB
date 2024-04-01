import pytest
import torch
import ase.data
from dptb.data.transforms import TypeMapper
from dptb.data import AtomicData, AtomicDataDict

def test_type_mapper_init():
    # 测试使用化学符号初始化
    chemical_symbols = ['H', 'He', 'Li']
    mapper = TypeMapper(chemical_symbols=chemical_symbols)
    assert mapper.num_types == 3
    assert mapper.type_names == chemical_symbols
    assert mapper.has_chemical_symbols
    assert mapper.chemical_symbol_to_type == {'H': 0, 'He': 1, 'Li': 2}

    chemical_symbols = ['C', 'He', 'B']
    mapper = TypeMapper(chemical_symbols=chemical_symbols)
    assert mapper.num_types == 3
    assert mapper.type_names == ['He', 'B', 'C'] # 使用chemical_symbols初始化，按照元素周期表的顺序排序
    assert mapper.chemical_symbol_to_type == {'C': 2, 'He': 0, 'B': 1}


    # 测试使用 chemical_symbol_to_type 初始化
    chemical_symbol_to_type = {'H': 0, 'He': 1, 'Li': 2}
    mapper = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)
    assert mapper.num_types == 3
    assert mapper.has_chemical_symbols

    chemical_symbol_to_type = {'H': 1, 'He': 2, 'Li': 0}
    mapper = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)
    # 使用 chemical_symbol_to_type 初始化，因为已经给了type的编号，因此不会再次进行排序
    assert mapper.chemical_symbol_to_type == {'H': 1, 'He': 2, 'Li': 0} 
    # 使用 chemical_symbol_to_type 初始化，type_names会chemical_symbol_to_type的value排序
    assert mapper.type_names == ['Li', 'H', 'He'] 

    # 测试使用 type_names 初始化
    type_names = ['Type1', 'Type2', 'Type3']
    mapper = TypeMapper(type_names=type_names)
    assert mapper.num_types == 3
    assert mapper.type_names == type_names
    assert not mapper.has_chemical_symbols

    # 测试无效输入
    with pytest.raises(ValueError):
        TypeMapper()

def test_type_mapper_transform():
    chemical_symbols = ['H', 'He', 'C']
    mapper = TypeMapper(chemical_symbols=chemical_symbols)

    atomic_numbers = torch.tensor([1, 2, 6])  # H, He, Li
    types = mapper.transform(atomic_numbers)
    assert torch.all(types == torch.tensor([0, 1, 2]))

    # 测试无效原子编号 - 1
    invalid_atomic_numbers = torch.tensor([1, 2, 5])
    with pytest.raises(ValueError):
        mapper.transform(invalid_atomic_numbers)

    # 测试无效原子编号 - 2
    invalid_atomic_numbers = torch.tensor([1, 2, 7])
    with pytest.raises(ValueError):
        mapper.transform(invalid_atomic_numbers)

def test_type_mapper_untransform():
    chemical_symbols = ['H', 'He', 'C']
    mapper = TypeMapper(chemical_symbols=chemical_symbols)

    types = torch.tensor([0, 1, 2])
    atomic_numbers = mapper.untransform(types)
    assert torch.all(atomic_numbers == torch.tensor([1, 2, 6]))

def test_call():
    chemical_symbols = ['H', 'He', 'C']
    mapper = TypeMapper(chemical_symbols=chemical_symbols)
    atomic_numbers = torch.tensor([1, 2, 6])
    data = {  AtomicDataDict.ATOMIC_NUMBERS_KEY: atomic_numbers    }
    transformed_data = mapper(data)

    assert not AtomicDataDict.ATOMIC_NUMBERS_KEY  in transformed_data
    assert  AtomicDataDict.ATOM_TYPE_KEY  in transformed_data
    assert torch.all(transformed_data[AtomicDataDict.ATOM_TYPE_KEY] == torch.tensor([0, 1, 2]))


def test_type_mapper_format():
    type_names = ['Type1', 'Type2', 'Type3']
    data = [1.23456, 2.34567, 3.45678]

    formatted_str = TypeMapper.format(data, type_names)
    expected_str = "[Type1: 1.234560, Type2: 2.345670, Type3: 3.456780]"
    assert formatted_str == expected_str

    formatted_str = TypeMapper.format(data, type_names, element_formatter=".2f")
    expected_str = "[Type1: 1.23, Type2: 2.35, Type3: 3.46]"
    assert formatted_str == expected_str

    formatted_str = TypeMapper.format(None, type_names)
    expected_str = "[Type1, Type2, Type3: None]"
    assert formatted_str == expected_str