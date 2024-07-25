from dptb.postprocess.bandstructure.band import Band
from dptb.utils.tools import j_loader
from dptb.nn.build import build_model
from dptb.data import build_dataset

model = build_model(checkpoint="./e3_silicon/checkpoint/nnenv.best.pth") # buld the trained e3 hamiltonian and overlap model

dataset = build_dataset(
    root="./data",
    prefix="Si64"
) # build the dataset

jdata = j_loader("./band.json")
kpath_kwargs = jdata["task_options"] # read the kpath information from the json file
# "task": "band",
# "kline_type":"abacus",
# "kpath":[[0.0000000000,   0.0000000000,   0.0000000000,   20],   
#         [0.5000000000,   0.0000000000,   0.0000000000,   1],               
#         [0.0000000000,   0.5000000000,   0.0000000000,   20],    
#         [0.0000000000,   0.0000000000,   0.0000000000,   20],     
#         [0.0000000000,   0.0000000000,   0.5000000000,   1],    
#         [-0.5000000000,  -0.5000000000,   0.5000000000,   20],                
#         [0.0000000000,   0.0000000000,   0.0000000000,   20],               
#         [0.0000000000,   -0.5000000000,   0.5000000000,   1 ],
#         [-0.5000000000,   0.0000000000,   0.5000000000, 20],
#         [0.0000000000, 0.0000000000, 0.0000000000, 20],
#         [0.5000000000, -0.500000000, 0.0000000000, 1]
#         ],
# "klabels":["G","X","Y","G","Z","R_2","G","T_2","U_2","G","V_2"],
# "nel_atom":{"Si":4},
# "E_fermi":0.0,
# "emin":-7,
# "emax":18

bcal = Band(model=model, 
            use_gui=False,
            results_path="./", 
            device=model.device)
bcal.get_bands(data=dataset[0], 
               kpath_kwargs=kpath_kwargs) # compute band structure

bcal.band_plot(
        E_fermi = kpath_kwargs["E_fermi"],
        emin = kpath_kwargs["emin"],
        emax = kpath_kwargs["emax"]
               ) # plot the band structure
