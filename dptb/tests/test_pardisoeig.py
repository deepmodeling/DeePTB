from dptb.postprocess.bandstructure.band import Band
from dptb.utils.tools import j_loader
from dptb.nn.build import build_model
from dptb.data import build_dataset, AtomicData, AtomicDataDict
from dptb.nn.hr2hR import Hr2HR
from dptb.nn.energy import PardisoEig
import torch
import pytest


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)

def test_hr2hr(root_directory):
    # build the trained e3_band hamiltonian and overlap model
    model = build_model(checkpoint=f"{root_directory}/dptb/tests/data/e3_band/ref_model/nnenv.ep1474.pth") 

    dataset = build_dataset.from_model(
        model=model,
        root=f"{root_directory}/dptb/tests/data/e3_band/data/",
        prefix="Si64"
    )

    adata = dataset[0]
    adata = AtomicData.to_AtomicDataDict(adata)

    hr2hr = Hr2HR(
        idp=model.idp,
        edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_field=AtomicDataDict.NODE_FEATURES_KEY,
        overlap=False,
        dtype=torch.float32, 
        device=torch.device("cpu")
    )

    sr2sr = Hr2HR(
        idp=model.idp,
        edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_field=AtomicDataDict.NODE_FEATURES_KEY,
        overlap=False,
        dtype=torch.float32, 
        device=torch.device("cpu")
    )

    adata = model(adata)
    image_h = hr2hr(adata)
    image_s = sr2sr(adata)

    h = image_h.sample_k([0,0,0], symm=True).to_scipy(format="csr")

    peig = PardisoEig(
        sigma=0.0,
        neig=10,
        mode='normal'
    )
    
    
    peig.solve(image_h, image_s, [[0,0,0],[1,1,1],[0.5,0.5,0.5]])
    
    # Test eigenvector return
    evals, evecs = peig.solve(image_h, image_s, [[0,0,0]], return_eigenvectors=True)
    assert len(evals) == 1
    assert len(evecs) == 1
    
    # Check shape
    N = image_h.sample_k([0,0,0], symm=True).to_scipy(format="csr").shape[0]
    assert evecs[0].shape == (N, 10)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])