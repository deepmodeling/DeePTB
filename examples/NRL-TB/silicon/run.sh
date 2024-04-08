
# load json model to plot band structure
dptb run in_band.json -i  ./outv2spd/nrl_ckpt.json -o band


# load  trained pth model to plot band structure
# dptb run in_band.json -i  ./nrlspd_frzoverlap/checkpoint/nnsk.best.pth -o band

# load json model to further train model:
# dptb train input_spd_fz.json -i  ./outv2spd/nrl_ckpt.json -o ckpt

