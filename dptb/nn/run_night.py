from dptb.data import ABACUSDataset
from dptb.data.transforms import OrbitalMapper
from dptb.nn.build import build_model

# "embedding": {
#         "method": "baseline",
#         "rc": 5.0,
#         "p": 4,
#         "n_axis": 10,
#         "n_basis": 20,
#         "n_radial": 45,
#         "n_layer": 4,
#         "radial_embedding": {
#             "neurons": [20,20,30],
#             "activation": "tanh",
#             "if_batch_normalized": False,
#         },
#     },
dptb_model_options = {
    "embedding": {
        "method": "deeph-e3",
        "irreps_embed": "64x0e", 
        "lmax": 5, 
        "irreps_mid": "64x0e+32x1o+16x2e+8x3o+8x4e+4x5o", 
        "n_layer": 3, 
        "r_max": 7.0, 
        "n_basis": 128, 
        "use_sc": True, 
        "no_parity": False, 
        "use_sbf": True,
    },
    # "embedding": {
    #     "method": "se2",
    #     "rc": 7.0,
    #     "rs": 2.0,
    #     "n_axis": 10,
    #     "radial_embedding": {
    #         "neurons": [120,120,130],
    #         "activation": "tanh",
    #         "if_batch_normalized": False,
    #     },
    # },
    # "embedding": {
        # "method": "baseline",
        # "rc": 4.0,
        # "p": 4,
        # "n_axis": 20,
        # "n_basis": 35,
    #     "n_radial": 300,
    #     "n_sqrt_radial": 20,
    #     "n_layer": 6,
    #     "radial_net": {
    #         "neurons": [1024, 1024],
    #         "activation": "tanh",
    #         "if_batch_normalized": False,
    #     },
    #     "hidden_net": {
    #         "neurons": [1024, 1024],
    #         "activation": "tanh",
    #         "if_batch_normalized": False,
    #     },
    # },
    # "embedding":{
    #     "method": "mpnn",
    #     "r_max": 7,
    #     "p": 4,
    #     "n_basis": 100,
    #     "n_node": 500,
    #     "n_edge": 500,
    #     "n_layer": 6,
    #     "if_exp": True,
    #     "node_net": {
    #         "neurons": [1024, 512],
    #         "activation": "silu",
    #         "if_batch_normalized": False,
    #     },
    #     "edge_net": {
    #         "neurons": [1024, 512],
    #         "activation": "silu",
    #         "if_batch_normalized": False,
    #     },
    # },
    # "prediction":{
    #     "method": "nn",
    #     "neurons": [512,1024,512],
    #     "activation": "silu",
    #     "if_batch_normalized": False,
    #     "quantities": ["hamiltonian"],
    #     "hamiltonian":{
    #         "method": "e3tb",
    #         "precision": 1e-5,               # use to check if rmax is large enough
    #         "overlap": False,
    #     }
    # }
    "prediction":{
        "method": "none",
        "quantities": ["hamiltonian"],
        "hamiltonian":{
            "method": "e3tb",
            "precision": 1e-5,               # use to check if rmax is large enough
            "overlap": False,
        }
    }
}

common_options = {
    "basis": {
        "Al": "4s4p1d",
        "As": "2s2p1d",
    },
    "device": "cpu",
    "dtype": "float32"
}

run_opt = {

}

# train_dataset = ABACUSDataset(
#     root="/share/semicond/lmp_abacus/abacus_hse_data/AlAs/result-prod-alas/prod-alas/AlAs/sys-000/",
#     preprocess_path="/share/semicond/lmp_abacus/abacus_hse_data/AlAs/result-prod-alas/prod-alas/AlAs/sys-000/T100/atomic_data/",
#     h5file_names=[
#         "frame-18/AtomicData.h5", 
#         "frame-29/AtomicData.h5",
#         "frame-64/AtomicData.h5", 
#         "frame-72/AtomicData.h5",
#         "frame-88/AtomicData.h5", 
#         "frame-98/AtomicData.h5",
#         ],
#     AtomicData_options={
#         "r_max": 5.0,
#         "er_max": 5.0,
#         "pbc": True,
#     },
#     type_mapper=OrbitalMapper(basis=common_options["basis"]),
# )

train_dataset = ABACUSDataset(
    root="/share/semicond/lmp_abacus/abacus_hse_data/AlAs/result-prod-alas/prod-alas/AlAs/sys-000/",
    preprocess_path="/share/semicond/lmp_abacus/abacus_hse_data/AlAs/result-prod-alas/prod-alas/AlAs/sys-000/atomic_data/",
    h5file_names=[
        "T100/frame-29/AtomicData.h5",
        "T500/frame-100/AtomicData.h5",
        "T500/frame-44/AtomicData.h5",
        "T1000/frame-27/AtomicData.h5",
        "T1000/frame-52/AtomicData.h5",
        "T1500/frame-35/AtomicData.h5",
        "T1500/frame-89/AtomicData.h5",
        ],
    AtomicData_options={
        "r_max": 7.0,
        "er_max": 7.0,
        "pbc": True,
    },
    type_mapper=OrbitalMapper(basis=common_options["basis"]),
)

# validation_dataset = ABACUSDataset(
#     root="/share/semicond/lmp_abacus/abacus_hse_data/AlAs/result-prod-alas/prod-alas/AlAs/sys-000/",
#     preprocess_path="/share/semicond/lmp_abacus/abacus_hse_data/AlAs/result-prod-alas/prod-alas/AlAs/sys-000/atomic_data/",
#     h5file_names=[
#         "T100/frame-88/AtomicData.h5",
#         "T600/frame-100/AtomicData.h5",
#         "T1000/frame-67/AtomicData.h5",
#         "T1500/frame-52/AtomicData.h5",
#         ],
#     AtomicData_options={
#         "r_max": 7.0,
#         "er_max": 7.0,
#         "pbc": True,
#     },
#     type_mapper=OrbitalMapper(basis=common_options["basis"]),
# )

# initilize trainer
from dptb.nnops.trainer import Trainer
from dptb.plugins.monitor import TrainLossMonitor, LearningRateMonitor, Validationer
from dptb.plugins.train_logger import Logger
import heapq
import logging
from dptb.utils.loggers import set_log_handles

train_options = {
    "seed": 12070,
    "num_epoch": 4000,
    "batch_size": 1,
    "optimizer": {
        "lr": 0.001,
        "type": "Adam",
    },
    "lr_scheduler": {
        "type": "exp",
        "gamma": 0.99
    },
    "loss_options":{
        "train":{"method": "hamil", "overlap": False},
        "validation":{"method": "hamil", "overlap": False},
    },
    "save_freq": 10,
    "validation_freq": 10,
    "display_freq": 1
}



dptb = build_model(run_options=run_opt, model_options=dptb_model_options, common_options=common_options)

trainer = Trainer(
    train_options = train_options,
    common_options = common_options,
    model = dptb,
    train_datasets = train_dataset,
)


trainer.register_plugin(TrainLossMonitor())
# trainer.register_plugin(Validationer())
trainer.register_plugin(LearningRateMonitor())
trainer.register_plugin(Logger(["train_loss", "lr"], 
    interval=[(1, 'iteration'), (1, 'epoch')]))
set_log_handles(getattr(logging, "INFO"))
for q in trainer.plugin_queues.values():
    heapq.heapify(q)


trainer.run(1500)