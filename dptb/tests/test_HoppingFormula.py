import pytest
from dptb.nn.sktb.hopping import HoppingFormula
from dptb.data.transforms import OrbitalMapper
import numpy as np
import torch
import torch.nn as nn

class TestHoppingFormula:
    basis = {
        "B": ["2s", "2p"],
        "N": ["2s", "2p"]
    }

    idp_sk = OrbitalMapper(basis=basis, method="sktb")
    idp_sk.get_orbpair_maps()
    idp_sk.get_skonsite_maps()

    reflective_bonds = []
    for ii in range(len(idp_sk.bond_types)):
        bond_name = "-".join(idp_sk.type_to_bond[ii].split("-")[::-1])
        reflective_bonds.append(idp_sk.bond_to_type[bond_name])
    reflective_bonds = np.array(reflective_bonds)   


    def test_reflective_bonds(self):
        assert (self.reflective_bonds == np.array([0, 2, 1, 3])).all()

    def test_hopping_param(self):
        hopping_param = torch.empty([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, 2])
        nn.init.normal_(hopping_param, mean=0.0, std=1)
        params = hopping_param
        reflect_params = params[self.reflective_bonds]
        for k in self.idp_sk.orbpair_maps.keys():
            iorb, jorb = k.split("-")
            if iorb == jorb:
                hopping_param[:,self.idp_sk.orbpair_maps[k],:] = 0.5 * (params[:,self.idp_sk.orbpair_maps[k],:] + reflect_params[:,self.idp_sk.orbpair_maps[k],:])

        reflect_params = hopping_param[self.reflective_bonds]
        assert torch.all(reflect_params[0] == hopping_param[0])
        assert torch.all(reflect_params[3] == hopping_param[3])
        assert torch.all(reflect_params[1] == hopping_param[2])
        assert torch.all(reflect_params[1,self.idp_sk.orbpair_maps['1s-1s'],:] == hopping_param[1,self.idp_sk.orbpair_maps['1s-1s'],:])
        assert torch.all(reflect_params[1,self.idp_sk.orbpair_maps['1p-1p'],:] == hopping_param[1,self.idp_sk.orbpair_maps['1p-1p'],:])
        assert torch.all(torch.abs(reflect_params[1,self.idp_sk.orbpair_maps['1s-1p'],:] - hopping_param[1,self.idp_sk.orbpair_maps['1s-1p'],:])> 1e-6)

    def test_hopping_fn_powerlaw(self):
        hop = HoppingFormula(functype='powerlaw')
        assert hop.num_paras == 2
        assert hop.functype == 'powerlaw'
        assert hop.overlap == False

        hopping_options = {'method': 'powerlaw', 'rs': 2.6, 'w': 0.35}
        hopping_params = torch.tensor([[[-0.1411516815,  0.1884556115],
         [ 0.0025317010,  0.0356881730],
         [-0.0211066809,  0.0549525395],
         [-0.1178076789, -0.1954882890]],

        [[ 0.0285179354,  0.0109017706],
         [ 0.0913131461,  0.0101144081],
         [-0.0309928786,  0.0247989111],
         [ 0.0271558911,  0.0849839747]],

        [[ 0.0285179354,  0.0109017706],
         [-0.0827299282,  0.1176596507],
         [-0.0309928786,  0.0247989111],
         [ 0.0271558911,  0.0849839747]],

        [[-0.0864071846,  0.0077147759],
         [ 0.0967203006,  0.1524405628],
         [-0.0170247164, -0.0191086754],
         [ 0.0211060215,  0.0705082119]]])

        rij = torch.tensor([2.5039999485, 1.4456850290, 2.5039999485, 2.8913702965, 4.3370552063,
        5.2124915123, 5.0079998970, 2.8913698196, 3.8249230385, 1.4456851482])
        r0 = torch.tensor([3.4000000954, 3.2000000477, 3.4000000954, 3.2000000477, 3.4000000954,
        3.2000000477, 3.4000000954, 3.2000000477, 3.2000000477, 3.2000000477])/1.8897259886
        edge_index = torch.tensor([3, 2, 3, 2, 3, 2, 3, 2, 2, 2])

        skints=hop.get_skhij(
            rij=rij,
            r0=r0,
            paraArray=hopping_params[edge_index],
            **hopping_options
        )
        except_skints = torch.tensor([[-3.5184152424e-02,  3.7543859333e-02, -6.9062374532e-03,
          8.4176203236e-03],
        [ 3.2268896699e-02, -9.5205180347e-02, -3.5146523267e-02,
          3.1089795753e-02],
        [-3.5184152424e-02,  3.7543859333e-02, -6.9062374532e-03,
          8.4176203236e-03],
        [ 5.0332243554e-03, -1.3790668920e-02, -5.4295156151e-03,
          4.6065850183e-03],
        [-2.4720147485e-04,  2.4362222757e-04, -4.8220012104e-05,
          5.7136447140e-05],
        [ 5.2428072195e-06, -1.3488981494e-05, -5.6094700085e-06,
          4.5934179980e-06],
        [-3.1633222534e-05,  3.0532923120e-05, -6.1603855102e-06,
          7.2457428359e-06],
        [ 5.0332304090e-03, -1.3790686615e-02, -5.4295226000e-03,
          4.6065901406e-03],
        [ 3.6688512773e-04, -9.7565469332e-04, -3.9423588896e-04,
          3.2889746944e-04],
        [ 3.2268892974e-02, -9.5205165446e-02, -3.5146519542e-02,
          3.1089795753e-02]])
        assert torch.allclose(skints, except_skints, atol=1e-6)
        
        # assert hop.powerlaw(1) == 1
                # check symm:
        skints=hop.get_skhij(
            rij=torch.tensor([1.4456850290, 1.4456850290]),
            r0=torch.tensor([3.2000000477, 3.2000000477]),
            paraArray=hopping_params[[2,1]],
            **hopping_options
        )
        assert skints[1,0] == skints[0,0]
        assert skints[1,1] != skints[0,1]
        assert skints[1,2] == skints[0,2]
        assert skints[1,3] == skints[0,3]

    def test_hopping_fn_varTang96(self):
        hop = HoppingFormula(functype='varTang96')
        assert hop.num_paras == 4
        assert hop.functype == 'varTang96'
        assert hop.overlap == False
        hopping_options = {'method': 'varTang96', 'rs': 2.6, 'w': 0.35}
        hopping_params = torch.tensor([[[-0.2412008047, -0.1411010474, -0.1898509115, -0.2404690534],
         [ 0.0917368904, -0.0668532252,  0.0636554882, -0.0292482674],
         [ 0.0535876118,  0.2154486924,  0.0440765359, -0.0101145217],
         [ 0.0335784964, -0.0215393025,  0.0570218973, -0.1350438148]],

        [[-0.1048618853,  0.0438515618,  0.0116684288,  0.0638046786],
         [ 0.0438838117,  0.0521011166,  0.1005084664, -0.0761227310],
         [ 0.1796368062, -0.0403945968, -0.0714195892,  0.0010075578],
         [ 0.0439102203, -0.0517578460,  0.1063545197,  0.0659844577]],

        [[-0.1048618853,  0.0438515618,  0.0116684288,  0.0638046786],
         [ 0.0282669961,  0.0175350886,  0.1031130701, -0.1271371096],
         [ 0.1796368062, -0.0403945968, -0.0714195892,  0.0010075578],
         [ 0.0439102203, -0.0517578460,  0.1063545197,  0.0659844577]],

        [[ 0.0185389221, -0.1147238389,  0.1271312535,  0.0258933604],
         [ 0.0641190037, -0.1572379917,  0.0771165416, -0.0175815504],
         [-0.1860750169, -0.1060247421,  0.0354737304,  0.1129984856],
         [ 0.1231309325,  0.0492822640, -0.0014328632,  0.1273492575]]])
        
        rij = torch.tensor([2.5039999485, 1.4456850290, 2.5039999485, 2.8913702965, 4.3370552063,
        5.2124915123, 5.0079998970, 2.8913698196, 3.8249230385, 1.4456851482])
        r0 = torch.tensor([3.4000000954, 3.2000000477, 3.4000000954, 3.2000000477, 3.4000000954,
        3.2000000477, 3.4000000954, 3.2000000477, 3.2000000477, 3.2000000477])
        edge_index = torch.tensor([3, 2, 3, 2, 3, 2, 3, 2, 2, 2])
        skints=hop.get_skhij(
            rij=rij,
            r0=r0,
            paraArray=hopping_params[edge_index],
            **hopping_options
        )
        expected_skints = torch.tensor([[ 8.3228154108e-03,  2.9156081378e-02, -9.2212997377e-02,
          6.6754586995e-02],
        [-9.8321832716e-02,  2.4309875444e-02,  1.5890605748e-01,
          3.7255816162e-02],
        [ 8.3228154108e-03,  2.9156081378e-02, -9.2212997377e-02,
          6.6754586995e-02],
        [-2.9963230714e-02,  7.4740285054e-03,  4.8566047102e-02,
          1.1240162887e-02],
        [ 9.5322779089e-05,  3.2658257987e-04, -1.0604971321e-03,
          7.9392339103e-04],
        [-5.5157794122e-05,  1.3852513803e-05,  8.9624154498e-05,
          2.0512125047e-05],
        [ 1.3863075765e-05,  4.7220117267e-05, -1.5439449635e-04,
          1.1660838209e-04],
        [-2.9963262379e-02,  7.4740359560e-03,  4.8566095531e-02,
          1.1240174063e-02],
        [-2.8621328529e-03,  7.1630079765e-04,  4.6445415355e-03,
          1.0692701908e-03],
        [-9.8321832716e-02,  2.4309875444e-02,  1.5890605748e-01,
          3.7255816162e-02]])
        assert torch.allclose(skints, expected_skints, atol=1e-6)
        
        # check symm:
        skints=hop.get_skhij(
            rij=torch.tensor([1.4456850290, 1.4456850290]),
            r0=torch.tensor([3.2000000477, 3.2000000477]),
            paraArray=hopping_params[[2,1]],
            **hopping_options
        )
        expected_skints = torch.tensor([[-0.0983218327,  0.0243098754,  0.1589060575,  0.0372558162],
                                        [-0.0983218327,  0.0374379344,  0.1589060575,  0.0372558162]])
        assert torch.allclose(skints, expected_skints, atol=1e-6)
        assert skints[1,0] == skints[0,0]
        assert skints[1,1] != skints[0,1]
        assert skints[1,2] == skints[0,2]
        assert skints[1,3] == skints[0,3]

