import pytest
import torch
from dptb.nn.dftb.sk_param import SKParam
import os
from pathlib import Path


rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "../../examples")


class TestSKParam:
    skdatapath = f"{rootdir}/hBN_dftb/slakos"

    def check_skdict(self, skparams):
        skdict = skparams.skdict
        assert isinstance(skdict, dict)
        assert "Distance" in skdict
        assert "Hopping" in skdict
        assert "Overlap" in skdict
        assert "OnsiteE" in skdict

        assert isinstance(skdict["Distance"],torch.Tensor)
        assert isinstance(skdict['Hopping'], torch.Tensor)
        assert isinstance(skdict['Overlap'], torch.Tensor)
        assert isinstance(skdict['OnsiteE'], torch.Tensor)

        assert len(skdict["Distance"].shape) == 1

        assert len(skdict['Hopping'].shape) == len(skdict['Overlap'].shape) == len(skdict['OnsiteE'].shape)== 3
        
        assert skdict['Hopping'].shape[0] == skdict['Overlap'].shape[0] == len(skparams.idp_sk.bond_types)
        assert skdict['Hopping'].shape[1] == skdict['Overlap'].shape[1] == skparams.idp_sk.reduced_matrix_element
        assert skdict['Hopping'].shape[2] == skdict['Overlap'].shape[2] == len(skdict["Distance"])
        
        assert skdict['OnsiteE'].shape[0] == skparams.idp_sk.num_types
        assert skdict['OnsiteE'].shape[1] == skparams.idp_sk.n_onsite_Es
        assert skdict['OnsiteE'].shape[2] == 1

    def test_init_path(self):
        with pytest.raises(IndexError):
            SKParam(basis={"C": ["s", "p"], "H": ["s"]}, skdata=self.skdatapath)
        with pytest.raises(FileNotFoundError):
            skparams = SKParam(basis={"C": ["2s", "2p"], "O": ["2s"]}, skdata=self.skdatapath)
        
        skparams = SKParam(basis={"C": ["2s", "2p"], "H": ["1s"]}, skdata=self.skdatapath)
        
        self.check_skdict(skparams)

    def test_read_files_and_init_from_dict_pth(self):
        skfiles ={
            'H-H': f'{self.skdatapath}/H-H.skf',
            'H-C': f'{self.skdatapath}/H-C.skf',
            'C-H': f'{self.skdatapath}/C-H.skf',
            'C-C': f'{self.skdatapath}/C-C.skf'}
        
        skdict = SKParam.read_skfiles(skfiles)

        assert isinstance(skdict, dict)
        assert "Distance" in skdict
        assert "Hopping" in skdict
        assert "Overlap" in skdict
        assert "OnsiteE" in skdict
        assert "HubdU" in skdict
        assert "Occu" in skdict

        assert isinstance(skdict["Distance"],dict)
        assert isinstance(skdict['Hopping'], dict)
        assert isinstance(skdict['Overlap'], dict)
        assert isinstance(skdict['OnsiteE'], dict)
        assert isinstance(skdict['HubdU'], dict)
        assert isinstance(skdict['Occu'], dict)


        assert len(skdict["Distance"]) == len(skfiles)
        assert len(skdict['Hopping']) == len(skfiles)
        assert len(skdict['Overlap']) == len(skfiles)
        assert len(skdict['OnsiteE']) == len(skfiles)/2

        for key in skfiles:
            assert key in skdict["Distance"]
            assert key in skdict['Hopping']
            assert key in skdict['Overlap']
            if key.split('-')[0] == key.split('-')[1]:
                iia = key.split('-')[0]
                assert iia in skdict['OnsiteE']
                assert iia in skdict['HubdU']
                assert iia in skdict['Occu']

                assert isinstance(skdict['OnsiteE'][iia], torch.Tensor)
                assert isinstance(skdict['HubdU'][iia], torch.Tensor)
                assert isinstance(skdict['Occu'][iia], torch.Tensor)
            else:
                assert key not in skdict['OnsiteE']
                assert key not in skdict['HubdU']
                assert key not in skdict['Occu']

            assert isinstance(skdict["Distance"][key], torch.Tensor)
            assert isinstance(skdict['Hopping'][key], torch.Tensor)
            assert isinstance(skdict['Overlap'][key], torch.Tensor)
            

            assert len(skdict["Distance"][key].shape) == 1
            nxx = skdict["Distance"]['C-C'].shape[0]
            assert skdict['Hopping'][key].shape == skdict['Overlap'][key].shape == torch.Size([10, nxx])

            if key.split('-')[0] == key.split('-')[1]:
                iia  =  key.split('-')[0]
                assert skdict['OnsiteE'][iia].shape == torch.Size([3])
                assert skdict['HubdU'][iia].shape == torch.Size([3])
                assert skdict['Occu'][iia].shape == torch.Size([3])
        
        ourdir = f"{rootdir}/../dptb/tests/data/out"
        torch.save(skdict, f"{ourdir}/skdict.pth")

        skparams = SKParam(basis={"C": ["2s", "2p"], "H": ["1s"]}, skdata=skdict)
        
        self.check_skdict(skparams)

        skparams = SKParam(basis={"C": ["2s", "2p"], "H": ["1s"]}, skdata=f"{ourdir}/skdict.pth")
        self.check_skdict(skparams)


