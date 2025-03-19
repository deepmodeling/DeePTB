import pytest
import os
import torch
from dptb.nn.nnsk import NNSK
from dptb.data.transforms import OrbitalMapper
from dptb.data.build import build_dataset
from pathlib import Path
from dptb.data import AtomicDataset, DataLoader, AtomicDataDict, AtomicData
import numpy as np
from dptb.utils.constants import atomic_num_dict_r
from dptb.tests.tstools import compare_tensors_as_sets_float

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")


class TestNNSK:

    common_options = {
        "basis": {
            "B": ["2s", "2p"],
            "N": ["2s", "2p"]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": False,
        "seed": 3982377700
    }
    model_options = {
    "nnsk": {
        "onsite": {
            "method": "uniform"
        },
        "hopping": {
            "method": "powerlaw",
            "rs": 2.6,
            "w": 0.35
        },
        "freeze": False,
        "std": 0.1,
        "push": None}
    }
    data_options = {
        "r_max": 2.6,
        "er_max": 2.6,
        "oer_max":1.6,
        "train": {
            "root": f"{rootdir}/hBN/dataset",
            "prefix": "kpath",
            "get_eigenvalues": True
        }
    }

    train_datasets = build_dataset(**data_options, **data_options["train"], **common_options)
    train_loader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=True)

    batch = next(iter(train_loader))
    batch = AtomicData.to_AtomicDataDict(batch)

    def test_nnsk_none_powerlaw(self):
        model_options = self.model_options
        model_options["nnsk"]["onsite"]["method"] = "none"
        model = NNSK(**model_options['nnsk'], **self.common_options,transform=False)

        data = model(self.batch)
        assert data[AtomicDataDict.NODE_FEATURES_KEY].shape == (2, 2)
        expected_onsiteE = torch.tensor([[-18.4200038910,  -7.2373123169], [ -9.3830089569,  -3.7138016224]])
        assert torch.allclose(data[AtomicDataDict.NODE_FEATURES_KEY], expected_onsiteE, atol=1e-10)

        # check params:
        reflective_bonds = np.array([model.idp_sk.bond_to_type["-".join(model.idp_sk.type_to_bond[i].split("-")[::-1])] for i  in range(len(model.idp_sk.bond_types))])
        assert (reflective_bonds == np.array([0, 2, 1, 3])).all()

        assert data[AtomicDataDict.EDGE_FEATURES_KEY].shape == torch.Size([18, 4])
        
        hopping_param = model.hopping_param
        assert model.hopping_param.shape == torch.Size([4, 4, 2])
        reflect_params = hopping_param[reflective_bonds]
        assert torch.all(reflect_params[0] == hopping_param[0])
        assert torch.all(reflect_params[3] == hopping_param[3])
        assert torch.all(reflect_params[1] == hopping_param[2])
        assert torch.all(reflect_params[1,model.idp_sk.orbpair_maps['1s-1s'],:] == hopping_param[1,model.idp_sk.orbpair_maps['1s-1s'],:])
        assert torch.all(reflect_params[1,model.idp_sk.orbpair_maps['1p-1p'],:] == hopping_param[1,model.idp_sk.orbpair_maps['1p-1p'],:])
        assert torch.all(torch.abs(reflect_params[1,model.idp_sk.orbpair_maps['1s-1p'],:] - hopping_param[1,model.idp_sk.orbpair_maps['1s-1p'],:])> 1e-6)

        model.hopping_param.data = torch.tensor([[[ 0.0310298353, -0.0458309017],
                                                  [-0.0073891575,  0.0130299432],
                                                  [-0.0648041517,  0.0022415700],
                                                  [ 0.0094638113,  0.0277521145]],

                                                 [[ 0.0545680001, -0.0022450921],
                                                  [-0.0633200631,  0.0212673787],
                                                  [-0.0274183583,  0.0192848369],
                                                  [-0.0155957118, -0.0132153183]],

                                                 [[ 0.0545680001, -0.0022450921],
                                                  [ 0.0559118390, -0.1622860432],
                                                  [-0.0274183583,  0.0192848369],
                                                  [-0.0155957118, -0.0132153183]],

                                                 [[ 0.0116694709,  0.1265508980],
                                                  [-0.1805780679, -0.0232779309],
                                                  [ 0.0463550761, -0.0525675006],
                                                  [-0.0973361954,  0.2217429876]]])
        
        data = model(self.batch)
        expected_hopskint = torch.tensor([[ 0.0045686616, -0.0731523260,  0.0185975507, -0.0369272530],
        [ 0.0616608486,  0.0647987276, -0.0310658477, -0.0176534709],
        [ 0.0045686616, -0.0731523260,  0.0185975507, -0.0369272530],
        [ 0.0616608560,  0.0647987351, -0.0310658514, -0.0176534727],
        [ 0.0045686616, -0.0731523260,  0.0185975507, -0.0369272530],
        [ 0.0616608560,  0.0647987351, -0.0310658514, -0.0176534727],
        [ 0.0109460149, -0.0026458376, -0.0233188029,  0.0033660505],
        [ 0.0109460149, -0.0026458376, -0.0233188029,  0.0033660505],
        [ 0.0109460149, -0.0026458376, -0.0233188029,  0.0033660505],
        [ 0.0045686616, -0.0731523260,  0.0185975507, -0.0369272530],
        [ 0.0616608486, -0.0717660785, -0.0310658477, -0.0176534709],
        [ 0.0045686616, -0.0731523260,  0.0185975507, -0.0369272530],
        [ 0.0616608560, -0.0717660859, -0.0310658514, -0.0176534727],
        [ 0.0045686616, -0.0731523260,  0.0185975507, -0.0369272530],
        [ 0.0616608560, -0.0717660859, -0.0310658514, -0.0176534727],
        [ 0.0109460149, -0.0026458376, -0.0233188029,  0.0033660505],
        [ 0.0109460149, -0.0026458376, -0.0233188029,  0.0033660505],
        [ 0.0109460149, -0.0026458376, -0.0233188029,  0.0033660505]])

        assert compare_tensors_as_sets_float(data[AtomicDataDict.EDGE_FEATURES_KEY], expected_hopskint, precision=6)
        # assert torch.allclose(data[AtomicDataDict.EDGE_FEATURES_KEY], expected_hopskint, atol=1e-10)

    def test_nnsk_uniform_varTang96(self):
        model_options = self.model_options
        model_options["nnsk"]["onsite"]["method"] = "uniform"
        model_options["nnsk"]["hopping"]["method"] = "varTang96"
        model = NNSK(**model_options['nnsk'], **self.common_options,transform=False)

        data = model(self.batch)
        assert data[AtomicDataDict.NODE_FEATURES_KEY].shape == (2, 2)

        assert model.onsite_param.shape == torch.Size([2, 2, 1])
        model.onsite_param.data = torch.tensor([[[ 0.0029474557],[-0.0363740884]],[[ 0.0442877077],[ 0.0745238438]]])
        data = model(self.batch)
        
        expected_onsiteE=torch.tensor([[-18.3757152557,  -7.1627883911],[ -9.3800611496,  -3.7501757145]])
        assert torch.allclose(data[AtomicDataDict.NODE_FEATURES_KEY], expected_onsiteE, atol=1e-8)
       
        # check params:
        reflective_bonds = np.array([model.idp_sk.bond_to_type["-".join(model.idp_sk.type_to_bond[i].split("-")[::-1])] for i  in range(len(model.idp_sk.bond_types))])
        assert (reflective_bonds == np.array([0, 2, 1, 3])).all()

        assert data[AtomicDataDict.EDGE_FEATURES_KEY].shape == torch.Size([18, 4])
        
        hopping_param = model.hopping_param
        assert model.hopping_param.shape == torch.Size([4, 4, 4])
        reflect_params = hopping_param[reflective_bonds]
        assert torch.all(reflect_params[0] == hopping_param[0])
        assert torch.all(reflect_params[3] == hopping_param[3])
        assert torch.all(reflect_params[1] == hopping_param[2])
        assert torch.all(reflect_params[1,model.idp_sk.orbpair_maps['1s-1s'],:] == hopping_param[1,model.idp_sk.orbpair_maps['1s-1s'],:])
        assert torch.all(reflect_params[1,model.idp_sk.orbpair_maps['1p-1p'],:] == hopping_param[1,model.idp_sk.orbpair_maps['1p-1p'],:])
        assert torch.all(torch.abs(reflect_params[1,model.idp_sk.orbpair_maps['1s-1p'],:] - hopping_param[1,model.idp_sk.orbpair_maps['1s-1p'],:])> 1e-6)

        model.hopping_param.data = torch.tensor([[[-0.0101760523, -0.1316463947, -0.1417859644, -0.1646364480],
         [ 0.0672337338, -0.1618858278,  0.2167103738, -0.0933398455],
         [-0.1156680509,  0.0753609762, -0.2145037651, -0.0209143031],
         [ 0.1458042562,  0.0542972945,  0.0912779644, -0.0455546267]],

        [[ 0.0951524228,  0.0572014563, -0.0067625209, -0.0385399312],
         [ 0.1281270236,  0.0849377736, -0.1423050016, -0.0156896617],
         [ 0.0883640796, -0.0021369997,  0.1038915813,  0.0695260465],
         [-0.0600154772, -0.0456165560, -0.0095871855, -0.0923910812]],

        [[ 0.0951524228,  0.0572014563, -0.0067625209, -0.0385399312],
         [ 0.0050242306,  0.0204085857,  0.0618826859, -0.0706135929],
         [ 0.0883640796, -0.0021369997,  0.1038915813,  0.0695260465],
         [-0.0600154772, -0.0456165560, -0.0095871855, -0.0923910812]],

        [[-0.0263542570, -0.1339780539, -0.0824253783, -0.0105612548],
         [ 0.0708842948,  0.1104989797, -0.0314897709,  0.0681586191],
         [ 0.0840835944,  0.0245513991, -0.0251300018,  0.0976505056],
         [-0.0091807013, -0.1158562526,  0.0938140526,  0.1089882031]]])
        
        data = model(self.batch)
        expected_hopskint = torch.tensor([[-0.0121830469,  0.0351885632,  0.0454408005, -0.0042278627],
        [ 0.0892327577,  0.0045129298,  0.0765390098, -0.0563499145],
        [-0.0121830469,  0.0351885632,  0.0454408005, -0.0042278627],
        [ 0.0892327577,  0.0045129298,  0.0765390098, -0.0563499145],
        [-0.0121830469,  0.0351885632,  0.0454408005, -0.0042278627],
        [ 0.0892327577,  0.0045129298,  0.0765390098, -0.0563499145],
        [-0.0043444759,  0.0260002706, -0.0492796339,  0.0716556087],
        [-0.0043444759,  0.0260002706, -0.0492796339,  0.0716556087],
        [-0.0043444759,  0.0260002706, -0.0492796339,  0.0716556087],
        [-0.0121830469,  0.0351885632,  0.0454408005, -0.0042278627],
        [ 0.0892327577,  0.1037823707,  0.0765390098, -0.0563499145],
        [-0.0121830469,  0.0351885632,  0.0454408005, -0.0042278627],
        [ 0.0892327577,  0.1037823707,  0.0765390098, -0.0563499145],
        [-0.0121830469,  0.0351885632,  0.0454408005, -0.0042278627],
        [ 0.0892327577,  0.1037823707,  0.0765390098, -0.0563499145],
        [-0.0043444759,  0.0260002706, -0.0492796339,  0.0716556087],
        [-0.0043444759,  0.0260002706, -0.0492796339,  0.0716556087],
        [-0.0043444759,  0.0260002706, -0.0492796339,  0.0716556087]])

        assert compare_tensors_as_sets_float(data[AtomicDataDict.EDGE_FEATURES_KEY], expected_hopskint, precision=6)
        # assert torch.allclose(data[AtomicDataDict.EDGE_FEATURES_KEY], expected_hopskint, atol=1e-10)

    def test_nnsk_onsite_strain(self):
        model_options = self.model_options
        model_options["nnsk"]["onsite"] =  {"method": "strain", "rs":2.6, "w":0.35}
        model_options["nnsk"]["hopping"]["method"] = "powerlaw"
        model = NNSK(**model_options['nnsk'], **self.common_options,transform=False)

        assert model.onsite_param is None
        assert hasattr(model, "strain_param")
        assert model.strain_param.shape == torch.Size([4, 4, 2])

        # check params:
        reflective_bonds = np.array([model.idp_sk.bond_to_type["-".join(model.idp_sk.type_to_bond[i].split("-")[::-1])] for i  in range(len(model.idp_sk.bond_types))])
        assert (reflective_bonds == np.array([0, 2, 1, 3])).all()
        
        strain_param = model.strain_param
        reflect_params = strain_param[reflective_bonds]
        assert torch.all(reflect_params[0] == strain_param[0])
        assert torch.all(reflect_params[3] == strain_param[3])
        assert torch.all(reflect_params[1] == strain_param[2])
        assert torch.all(torch.abs(reflect_params[1] - strain_param[1])>1e-5) # not equal.
        # assert torch.all(reflect_params[1,model.idp_sk.orbpair_maps['1s-1s'],:] == strain_param[1,model.idp_sk.orbpair_maps['1s-1s'],:])
        # assert torch.all(reflect_params[1,model.idp_sk.orbpair_maps['1p-1p'],:] == strain_param[1,model.idp_sk.orbpair_maps['1p-1p'],:])
        # assert torch.all(torch.abs(reflect_params[1,model.idp_sk.orbpair_maps['1s-1p'],:] - strain_param[1,model.idp_sk.orbpair_maps['1s-1p'],:])> 1e-6)

        data = model(self.batch)
        assert data[AtomicDataDict.NODE_FEATURES_KEY].shape == (2, 2)
        expected_onsiteE = torch.tensor([[-18.4200038910,  -7.2373123169], [ -9.3830089569,  -3.7138016224]])
        assert torch.allclose(data[AtomicDataDict.NODE_FEATURES_KEY], expected_onsiteE, atol=1e-10)

        model.strain_param.data = torch.tensor([[[-0.1173106357,  0.0362264216],
                                                 [-0.1356740892,  0.0245320536],
                                                 [ 0.2883755565, -0.1153828502],
                                                 [-0.0108789466, -0.0314204618]],

                                                [[-0.0038241993,  0.2925966978],
                                                 [ 0.0366993770,  0.0363534875],
                                                 [ 0.0616651438, -0.0194761902],
                                                 [ 0.0362601653, -0.0779399052]],

                                                [[ 0.1159430519, -0.0517875031],
                                                 [-0.1467775553,  0.1143923178],
                                                 [-0.0056324191,  0.1099175364],
                                                 [ 0.0007288644, -0.0285613686]],

                                                [[-0.0847647935,  0.0672122389],
                                                 [-0.1561884433, -0.1205002591],
                                                 [-0.0723597482, -0.1240826175],
                                                 [-0.1364385933, -0.1005168483]]])
        data = model(self.batch)

        expected_onsiteskints = torch.tensor([[ 0.1320440024, -0.1688235849, -0.0064738272,  0.0008270382],
        [ 0.1320440173, -0.1688235998, -0.0064738286,  0.0008270383],
        [ 0.1320440173, -0.1688235998, -0.0064738286,  0.0008270383],
        [-0.0045243129,  0.0416939147,  0.0698706210,  0.0414667279],
        [-0.0045243134,  0.0416939184,  0.0698706284,  0.0414667316],
        [-0.0045243134,  0.0416939184,  0.0698706284,  0.0414667316]])

        assert torch.allclose(data[AtomicDataDict.ONSITENV_FEATURES_KEY], expected_onsiteskints, atol=1e-10)

class TestNNSK_rmax_dict:
    common_options = {
        "basis": {
            "B": ["2s", "2p"],
            "N": ["2s", "2p"]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": False,
        "seed": 3982377700
    }
    model_options = {
    "nnsk": {
        "onsite": {
            "method": "uniform"
        },
        "hopping": {
            "method": "powerlaw",
            "rs": 2.6,
            "w": 0.35
        },
        "freeze": False,
        "std": 0.1,
        "push": None}
    }
    data_options = {
        "r_max": 2.6,
        "er_max": 2.6,
        "oer_max":1.6,
        "train": {
            "root": f"{rootdir}/hBN/dataset",
            "prefix": "kpath",
            "get_eigenvalues": True
        }
    }
    train_datasets = build_dataset(**data_options, **data_options["train"], **common_options)
    train_loader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=True)

    batch = next(iter(train_loader))
    batch = AtomicData.to_AtomicDataDict(batch)

    hopping_formula = ['varTang96', 'powerlaw','poly1pow','poly2pow','poly3pow','poly2exp']

    
    def test_nnsk_rmax_dict_samevalue(self):
        model_options = self.model_options.copy()
        rs_old = 2.6
        rs_new_dict = {'B':2.6, 'N':2.6}
        for formula in self.hopping_formula:
            model_options["nnsk"]["hopping"]["method"] = formula
            model_options["nnsk"]["hopping"]["rs"] = rs_old
            model = NNSK(**model_options['nnsk'], **self.common_options,transform=False)
            data = model(self.batch)
            hopping_old = data[AtomicDataDict.EDGE_FEATURES_KEY].clone()
            
            model_options["nnsk"]["hopping"]["rs"] = rs_new_dict
            model2 = NNSK(**model_options['nnsk'], **self.common_options,transform=False)
            model2.hopping_param = model.hopping_param
            data = model2(self.batch)
            hopping_new = data[AtomicDataDict.EDGE_FEATURES_KEY].clone()
            assert torch.allclose(hopping_old, hopping_new, atol=1e-5)


    def test_nnsk_rmax_dict_diffvalue(self):
        model_options = self.model_options.copy()
        rs_old = 2.6
        rs_new_dict = {'B':3.6, 'N':2.0}
        for formula in self.hopping_formula:
            model_options["nnsk"]["hopping"]["method"] = formula
            model_options["nnsk"]["hopping"]["rs"] = 2.6
            model = NNSK(**model_options['nnsk'], **self.common_options,transform=False)
            data = model(self.batch)
            hopping_old = data[AtomicDataDict.EDGE_FEATURES_KEY].clone()

            model_options["nnsk"]["hopping"]["rs"] = rs_new_dict
            model3 = NNSK(**model_options['nnsk'], **self.common_options,transform=False)
            model3.hopping_param = model.hopping_param
            data = model3(self.batch)
            hopping_new = data[AtomicDataDict.EDGE_FEATURES_KEY].clone()
            assert not torch.allclose(hopping_old, hopping_new, atol=1e-5)

            edge_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten() # it is bond_type index, transform it to reduced bond index
            edge_number = model.idp_sk.untransform_bond(edge_index).T

            assert edge_number.shape[1] == hopping_old.shape[0]

            for i in range(edge_number.shape[1]):
                isymbol = atomic_num_dict_r[int(edge_number[0,i])]
                jsymbol = atomic_num_dict_r[int(edge_number[1,i])]
                rs =  0.5 * (rs_new_dict[isymbol] + rs_new_dict[jsymbol])
                rij = data[AtomicDataDict.EDGE_LENGTH_KEY][i]
                w = model_options["nnsk"]["hopping"]["w"]
                if formula in ['varTang96', 'powerlaw']:
                    fij_old = 1/(1+torch.exp((rij-rs_old)/w))
                    fij_new = 1/(1+torch.exp((rij-rs)/w))

                else:
                    fij_old = 1/(1+torch.exp((rij-rs_old+5*w)/w))
                    fij_new = 1/(1+torch.exp((rij-rs+5*w)/w))
                
                assert torch.allclose(hopping_new[i] / fij_new, hopping_old[i] / fij_old, atol=1e-5)    
    
    def test_nnsk_rmax_bondwise_dict_samevalue(self):
        model_options = self.model_options.copy()
        rs_old = 2.6
        rs_new_dict = {'B-B':2.6, 'N-N':2.6,'B-N':2.6}
        for formula in self.hopping_formula:
            model_options["nnsk"]["hopping"]["method"] = formula
            model_options["nnsk"]["hopping"]["rs"] = rs_old
            model = NNSK(**model_options['nnsk'], **self.common_options,transform=False)
            data = model(self.batch)
            hopping_old = data[AtomicDataDict.EDGE_FEATURES_KEY].clone()
            
            model_options["nnsk"]["hopping"]["rs"] = rs_new_dict
            model2 = NNSK(**model_options['nnsk'], **self.common_options,transform=False)
            model2.hopping_param = model.hopping_param
            data = model2(self.batch)
            hopping_new = data[AtomicDataDict.EDGE_FEATURES_KEY].clone()
            assert torch.allclose(hopping_old, hopping_new, atol=1e-5)

    def test_nnsk_rmax_bondwise_dict_diffvalue(self):
        model_options = self.model_options.copy()
        rs_old = 2.6
        rs_new_dict = {'B-B':3.6, 'B-N':2.8, 'N-B':2.8, 'N-N':2.0}
        for formula in self.hopping_formula:
            model_options["nnsk"]["hopping"]["method"] = formula
            model_options["nnsk"]["hopping"]["rs"] = 2.6
            model = NNSK(**model_options['nnsk'], **self.common_options,transform=False)
            data = model(self.batch)
            hopping_old = data[AtomicDataDict.EDGE_FEATURES_KEY].clone()

            model_options["nnsk"]["hopping"]["rs"] = rs_new_dict
            model3 = NNSK(**model_options['nnsk'], **self.common_options,transform=False)
            model3.hopping_param = model.hopping_param
            data = model3(self.batch)
            hopping_new = data[AtomicDataDict.EDGE_FEATURES_KEY].clone()
            assert not torch.allclose(hopping_old, hopping_new, atol=1e-5)

            edge_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten() # it is bond_type index, transform it to reduced bond index
            edge_number = model.idp_sk.untransform_bond(edge_index).T

            assert edge_number.shape[1] == hopping_old.shape[0]

            for i in range(edge_number.shape[1]):
                isymbol = atomic_num_dict_r[int(edge_number[0,i])]
                jsymbol = atomic_num_dict_r[int(edge_number[1,i])]
                # rs =  0.5 * (rs_new_dict[isymbol] + rs_new_dict[jsymbol])
                rs = rs_new_dict[f"{isymbol}-{jsymbol}"]
                rij = data[AtomicDataDict.EDGE_LENGTH_KEY][i]
                w = model_options["nnsk"]["hopping"]["w"]
                if formula in ['varTang96', 'powerlaw']:
                    fij_old = 1/(1+torch.exp((rij-rs_old)/w))
                    fij_new = 1/(1+torch.exp((rij-rs)/w))

                else:
                    fij_old = 1/(1+torch.exp((rij-rs_old+5*w)/w))
                    fij_new = 1/(1+torch.exp((rij-rs+5*w)/w))
                
                assert torch.allclose(hopping_new[i] / fij_new, hopping_old[i] / fij_old, atol=1e-5)   