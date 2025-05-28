from dptb.postprocess.bandstructure.band import Band
from dptb.utils.tools import j_loader
from dptb.nn.build import build_model
from dptb.data import build_dataset

# build the trained e3_band hamiltonian and overlap model
model = build_model(checkpoint="./ref_model/nnenv.ep1474.pth")

# build the dataset from the model
dataset = build_dataset.from_model(
    model=model,
    root="./data",
    prefix="Si64"
)  

jdata = j_loader("band.json")
kpath_kwargs = jdata["task_options"]

bcal = Band(model=model,
            use_gui=False,
            results_path="../../dptb/tests/data/e3_band/",
            device=model.device)
bcal.get_bands(data=dataset[0], 
               kpath_kwargs=kpath_kwargs) # compute band structure
bcal.band_plot(
        E_fermi = kpath_kwargs["E_fermi"],
        emin = kpath_kwargs["emin"],
        emax = kpath_kwargs["emax"]) # plot the band structure