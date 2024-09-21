#  NRL-TB paramererization

The DeePTB package support TB parameters with NRL-TB format, and provide a useful tools to transcript the NRL-TB format to our DeePTB loadable and readable json checkpoint. Moreover, utilizing the efficient differentiable optimization, we can further finetune the transcripted parameters in the DeePTB training framework, to achieve ab initial accuracy without training from scratch.

## NRL-TB Introduction:
The NRL Tight-binding method provides an efficient method for calculating properties of materials. The advantage of the NRL-TB method over classical potential simulations is that it explicitly incorporates the real electronic structure and bonding of the material, obtained by an interpolation from a database of first-principles results. 

For full theory of NRL-TB method, we refer to the official website http://esd.cos.gmu.edu/tb/.

## Transcript NRL-TB parameters to DeePTB
Here we show a few lines of a templete NRL-TB parameter file, with bulk silicon system using s, p orbital set. This file is located at DeePTB/examples/NRL-TB/silicon/Si-sp.par:
```
NN00001                           (New style Overlap Parameters)
Silicon (Si) -- sp parametrization
1                                 (One atom type in this file)
12.5   0.5                        (RCUT and SCREENL for 1-1 interactions)
4                                 (Orbitals for atom 1)
28.086                            (Atomic Weight of Atom 1)
 2.0  2.0  0.0                    (formal spd valence occupancy for atom 1)
   .110356625153E+01   0   1     lambda  (equation 7)
  -.532334619024E-01   0   2     a_s     (equation 9)
  -.907642743186E+00   0   3     b_s     (equation 9)
  -.883084913674E+01   0   4     c_s     (equation 9)
   .565661321469E+02   0   5     d_s     (equation 9)
   .357859715265E+00   0   6     a_p     (equation 9)
   .303647693101E+00   0   7     b_p     (equation 9)
   .709222903560E+01   0   8     c_p     (equation 9)
  -.774785508399E+02   0   9     d_p     (equation 9)
   .100000000000E+02   1  10     a_t2g   (equation 9)
   .000000000000E+00   1  11     b_t2g   (equation 9)
   .000000000000E+00   1  12     c_t2g   (equation 9)
   .000000000000E+00   1  13     d_t2g   (equation 9)
   .100000000000E+02   1  14     a_eg    (equation 9)
   .000000000000E+00   1  15     b_eg    (equation 9)
   .000000000000E+00   1  16     c_eg    (equation 9)
   .000000000000E+00   1  17     d_eg    (equation 9)
   .219560813651E+03   0  18     e_{ss sigma}    (equation 11) (Hamiltonian)
  -.162132459618E+02   0  19     f_{ss sigma}    (equation 11) (Hamiltonian)
  -.155048968097E+02   0  20     fbar_{ss sigma} (equation 11) (Hamiltonian)
   .126439940008E+01   0  78     g_{ss sigma}    (equation 11) (Hamiltonian)
   .101276876206E+02   0  21     e_{sp sigma}    (equation 11) (Hamiltonian)
```
You can download more parameters from internet and use them in our code. To transcript the file, we can use the command:
```bash
usage: dptb n2j [-h] [-v {DEBUG,3,INFO,2,WARNING,1,ERROR,0}] [-l LOG_PATH]
                [-nrl NRL_FILE] [-o OUTDIR]
                INPUT

positional arguments:
  INPUT                 the input parameter file in json or yaml format

optional arguments:
  -h, --help            show this help message and exit
  -v {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                        set verbosity level by string or number, 0=ERROR,
                        1=WARNING, 2=INFO and 3=DEBUG (default: INFO)
  -l LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk, if not
                        specified, the logs will only be output to console
                        (default: None)
  -nrl NRL_FILE, --nrl_file NRL_FILE
                        The NRL file name (default: None)
  -o OUTDIR, --outdir OUTDIR
                        The output files to save the transfered model and
                        updated input. (default: ./)
```
For example:
```bash
dptb n2j DeePTB/examples/NRL-TB/silicon/input_n2j.json -nrl DeePTB/examples/NRL-TB/silicon/Si_sp.par -o nrl_out
```
For doing this, we need to have a config file contains the common_options as the `input_n2j.json`:
```
{
    "common_options": {
        "basis": {
            "Si": [
                "3s",
                "3p",
                "d*"
            ]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": true,
        "seed": 3982377700
    }
}
```

Then, you can find the transcripted DeePTB json model in `./nrl_out/nrl_ckpt.json`:
```
{
    "version": 2,
    "unit": "eV",
    "common_options": {
        "basis": {
            "Si": [
                "3s",
                "3p"
            ]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": true
    },
    "model_options": {
        "nnsk": {
            "onsite": {
                "method": "NRL",
                "rc": 6.6147151362875,
                "w": 0.2645886054515,
                "lda": 1.517042837140912
            },
            "hopping": {
                "method": "NRL1",
                "rc": 6.6147151362875,
                "w": 0.2645886054515
            }
        }
    },
    "model_params": {
        "onsite": {
            "Si-3s-0": [
                -0.7242781465186467,
                -12.349108629101169,
                -120.1498233699409,
                769.6214351454472
            ],
            "Si-3p-0": [
                4.868929466977601,
                4.131337329837268,
                96.49469181636128,
                -1054.1493863419682
            ]
        },
        "hopping": {
            "Si-Si-3s-3s-0": [
                2987.2770523703775,
                -416.85931392897385,
                -753.3335086381961,
                1.738135839617497
            ],
            "Si-Si-3s-3p-0": [
                137.79420981142889,
                -113.22319395024688,
                11.013502291392534,
                1.2683722943630147
            ],
            "Si-Si-3p-3p-0": [
                -312.3734908328387,
                44.24279542724431,
                68.95104019693315,
                1.4177954447585202
            ],
            "Si-Si-3p-3p-1": [
                139.6685524393218,
                120.11742815429497,
                -107.67596847852917,
                1.5277405887687854
            ]
        },
        "overlap": {
            "Si-Si-3s-3s-0": [
                9.746427246194099,
                2.3569360238929495,
                -0.5502870702732275,
                1.5233364156496074
            ],
            "Si-Si-3s-3p-0": [
                16.768761923322074,
                -57.99684419121014,
                34.971872964934846,
                1.7054914546860056
            ],
            "Si-Si-3p-3p-0": [
                21.260342995500338,
                -4.178618273015186,
                -7.1474883718956805,
                1.5638674454024022
            ],
            "Si-Si-3p-3p-1": [
                -1308.0386246487092,
                1414.6889330892875,
                -93.24315909334297,
                2.1616536435817872
            ]
        }
    }
}
```
and a recommend model options for continuous training using this checkpoint:
```
{
    "common_options": {
        "basis": {
            "Si": [
                "3s",
                "3p"
            ]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": true,
        "seed": 3982377700
    },
    "model_options": {
        "nnsk": {
            "onsite": {
                "method": "NRL",
                "rc": 6.6147151362875,
                "w": 0.2645886054515,
                "lda": 1.517042837140912
            },
            "hopping": {
                "method": "NRL1",
                "rc": 6.6147151362875,
                "w": 0.2645886054515
            },
            "freeze": false,
            "std": 0.01,
            "push": false
        }
    }
}
```

You can use the transcripted model as any other DeePTB model.