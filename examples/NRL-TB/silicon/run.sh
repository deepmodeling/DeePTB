
# load json model to plot band structure
dptb run band_jsonckpt.json  -sk -i  ./ckpt/nrl_ckpt.json -o band

# load json model to further train model:
#dptb train input_nrl.json -sk -i ./ckpt/nrl_ckpt.json -o ckpt

# load the trained model to plot band structure
#dptb run band_pthckpt.json -sk -i ./ckpt/nrl_ckpt.pth -o bandpth
