
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from dptb.postprocess.unified.system import TBSystem
from dptb.data import AtomicData, AtomicDataDict
from dptb.postprocess.unified.properties.export import ExportAccessor

# Mock Model and Data
@pytest.fixture
def mock_system():
    # Mock Calculator/Model
    model = MagicMock(spec=torch.nn.Module)
    model.overlap = False
    # Mock IDP (Inverse Data Processor) needed for some exports
    model.idp = MagicMock()
    model.idp.type_names = {0: 'Si'}
    model.idp.basis = {'Si': ['s', 'px', 'py', 'pz']}
    model.idp.basis_to_full_basis = {'Si': {'s': 's', 'px': 'px', 'py': 'py', 'pz': 'pz'}}
    # Mock Calculator wrapper
    calculator = MagicMock()
    calculator.model = model
    calculator.device = 'cpu'
    calculator.cutoffs = {}
    
    # Mock System
    # We need to bypass __init__ complex logic or mock it
    # Easier to instantiate and then patch
    
    # Create valid AtomicData
    pos = torch.zeros((2, 3))
    cell = torch.eye(3)
    atom_types = torch.zeros(2, dtype=torch.long)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = AtomicData(pos=pos, cell=cell, atom_types=atom_types, edge_index=edge_index)
    
    with patch('dptb.postprocess.unified.system.build_model'), \
         patch('dptb.postprocess.unified.system.DeePTBAdapter') as MockAdapter, \
         patch.object(TBSystem, 'set_atoms') as mock_set_atoms:
         
        # Configure Adapter Mock
        adapter_instance = MockAdapter.return_value
        adapter_instance.model = model
        adapter_instance.device = 'cpu'
        adapter_instance.cutoffs = {}
         
        # Mock set_atoms to return valid dictionary without processing
        # Need to provide keys expected by exports
        data_dict = {
            AtomicDataDict.POSITIONS_KEY: pos,
            AtomicDataDict.CELL_KEY: cell,
            AtomicDataDict.ATOM_TYPE_KEY: atom_types,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.NODE_FEATURES_KEY: torch.randn(2, 4) # Dummy node features
        }
        mock_set_atoms.return_value = data_dict
         
        tbsys = TBSystem(calculator=model, data=data)
        tbsys._atomic_data = data_dict
        
        return tbsys

def test_export_accessor_init(mock_system):
    assert isinstance(mock_system.export, ExportAccessor)

@patch('dptb.postprocess.unified.properties.export.ToPythTB')
def test_to_pythtb(mock_to_pythtb, mock_system):
    mock_exporter = mock_to_pythtb.return_value
    mock_exporter.get_model.return_value = "pythtb_model"
    
    res = mock_system.export.to_pythtb()
    
    assert res == "pythtb_model"
    mock_to_pythtb.assert_called_with(mock_system.model, device='cpu')
    mock_exporter.get_model.assert_called_with(mock_system._atomic_data)

@patch('dptb.postprocess.unified.properties.export.ToPybinding')
def test_to_pybinding(mock_to_pybinding, mock_system):
    mock_exporter = mock_to_pybinding.return_value
    mock_exporter.get_lattice.return_value = "pybinding_lattice"
    
    res = mock_system.export.to_pybinding(results_path="test")
    
    assert res == "pybinding_lattice"
    mock_to_pybinding.assert_called_with(
        model=mock_system.model, 
        results_path="test",
        overlap=False,
        device='cpu'
    )
    mock_exporter.get_lattice.assert_called_with(mock_system._atomic_data)

@patch('dptb.postprocess.unified.properties.export.TBPLaS')
def test_to_tbplas(mock_to_tbplas, mock_system):
    mock_exporter = mock_to_tbplas.return_value
    mock_exporter.get_cell.return_value = ("tbplas_cell", 0.0)
    
    # Set Fermi level
    mock_system._efermi = 5.0
    
    res = mock_system.export.to_tbplas(results_path="test")
    
    assert res == ("tbplas_cell", 0.0)
    # Check if get_cell was called with correct ef
    mock_exporter.get_cell.assert_called_with(mock_system._atomic_data, e_fermi=5.0)

@patch('dptb.postprocess.unified.properties.export.ToWannier90')
def test_to_wannier90(mock_to_w90, mock_system):
    mock_exporter = mock_to_w90.return_value
    
    mock_system.export.to_wannier90(filename_prefix="test_w90")
    
    mock_to_w90.assert_called_with(mock_system.model, device='cpu')
    mock_exporter.write_hr.assert_called()
    mock_exporter.write_win.assert_called()
    mock_exporter.write_centres.assert_called()
    
    args, kwargs = mock_exporter.write_hr.call_args
    assert kwargs['filename'] == "test_w90_hr.dat"
