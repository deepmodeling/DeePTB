# A quick Example

## h-BN model

DeePTB is a package that utilizes machine-learning method to train TB models for target systems with the DFT training data. Here, h-BN monolayer  has been chosen as a quick start example. 

hBN is a binary compound made of equal numbers of boron (B) and nitrogen (N), we present this as a quick hands-on example. The prepared files are located in:
```
deeptb/examples/hBN/
-- data/kpath.0/
-- -- bandinfo.json
-- -- xdat.traj
-- -- kpoints.npy
-- -- eigs.npy
-- run/
-- input_short.json
```
The ```input_short.json``` file contains the least number of parameters that are required to start training the **DeePTB** model. ```data``` folder contains the bandstructure data ```kpath.0```, where another important configuration file ```bandinfo.json``` is located.

```bash
cd deeptb/examples/hBN/data
# to see the bond length
dptb bond struct.vasp  
# output:
Bond Type         1         2         3         4         5
------------------------------------------------------------------------
       N-N      2.50      4.34      5.01
       N-B      1.45      2.89      3.82      5.21      5.78
       B-B      2.50      4.34      5.01
```
Having the data file and input parameter, we can start training our first **DeePTB** model, the first step using the parameters defined in ```input_short.json```:
Here list some important parameters:
```json
    "common_options": {
        "onsitemode": "none",
        "bond_cutoff": 1.6,
    }
    
    "proj_atom_anglr_m": {
            "N": [
                "2s",
                "2p"
            ],
            "B": [
                "2s",
                "2p"
            ]
        }
```
The ```onsitemode``` is set to ```none``` which means we do not use onsite correction. The ```bond_cutoff``` is set to ```1.6``` which means we use the 1st nearest neighbour for bonding. The ```proj_atom_anglr_m``` is set to ```2s``` and ```2p``` which means we use $s$ and $p$ orbitals as basis. 

using the command to train the first model:
```bash
cd deeptb/examples/hBN
dptb train -sk input_short.json -o ./first
```
Here ``-sk`` indicate to fit the sk parameters, and ``-o`` indicate the output directory. During the fitting procedure, we can see the loss curve of hBN is decrease consistently. When finished, we get the fitting results in folders ```first```:
```shell
first/
|-- checkpoint
|   |-- best_nnsk_b1.600_c1.600_w0.300.json
|   |-- best_nnsk_b1.600_c1.600_w0.300.pth
|   |-- latest_nnsk_b1.600_c1.600_w0.300.json
|   `-- latest_nnsk_b1.600_c1.600_w0.300.pth
|-- input_short.json
|-- log
|   `-- log.txt
`-- train_config.json
```
Here checkpoint saves our fitting files, which best indicate the one in the fitting procedure which has the lowest validation loss. The latest is the most recent results.
we can plot the fitting bandstructure as:
```bash
dptb run -sk band.json  -i ./first/checkpoint/best_nnsk_b1.600_c1.600_w0.300.pth -o ./band
```
``-i`` states initialize the model from the checkpoint file `./first/checkpoint/best_nnsk_b1.600_c1.600_w0.300.pth`. results will be saved in the directory `band`:
```
band/
-- log/
-- -- log.txt
-- results/
-- -- band.png
-- -- bandstructure.npy
```
Where `band.png` is the band structure of the trained model. Which looks like this:

<div align=center>
<img src="https://raw.githubusercontent.com/deepmodeling/DeePTB/main/docs/img/band_0.png" width = "60%" height = "60%" alt="hBN Bands" align=center />
</div>


It shows that the fitting has learned the rough shape of the bandstructure, but not very accurate. We can further improve the accuracy by incorporating more features of our code, for example, the onsite correction. There are two kinds of onsite correction supported: `uniform` or `strain`. We use `strain` for now to see the effect. Now change the `input_short.json` by the parameters:
```json
    "common_options": {
        "onsitemode": "strain",
        "bond_cutoff": 1.6,
    }
    "train_options": {
        "num_epoch": 800,
    }
```
After the training is finished, you can get the strain folder with:
```shell
strain
|-- checkpoint
|   |-- best_nnsk_b1.600_c1.600_w0.300.json
|   |-- best_nnsk_b1.600_c1.600_w0.300.pth
|   |-- latest_nnsk_b1.600_c1.600_w0.300.json
|   `-- latest_nnsk_b1.600_c1.600_w0.300.pth
|-- log
|   `-- log.txt
`-- train_config.json
```
plot the result again:
```bash
dptb run -sk band.json  -i ./strain/checkpoint/best_nnsk_b1.600_c1.600_w0.300.pth -o ./band
```
<div align=center>
<img src="./examples/hBN/reference/2.strain/results/band.png" width = "60%" height = "60%" alt="hBN Bands" align=center />
</div>
It looks ok, we can further improve the accuracy by adding more neighbours, and training for a longer time. We can gradually increase the `sk_cutoff` from 1st to 3rd neighbour. change the `input_short.json` by the parameters:
```json
    "common_options": {
        "onsitemode": "strain",
        "bond_cutoff": 3.6,
    }
    "train_options": {
        "num_epoch": 2000,
    }
    "model_options": {
        "skfunction": {
            "sk_cutoff": [1.6,3.6],
            "sk_decay_w": 0.3,
        }
    }
```
This means that we use up to 3rd nearest neighbour for bonding, and we train for 2000 epochs. see the input file `hBN/reference/3.varycutoff/input_short.json` for detail. Then we can run the training again:
```bash
dptb train -sk input_short.json -o ./varycutoff -i ./strain/checkpoint/best_nnsk_b1.600_c1.600_w0.300.pth
```

After the training is finished, you can get the strain folder with:
```shell
varycutoff
|-- checkpoint
|   |-- best_nnsk_b3.600_c1.600_w0.300.json
|   |-- best_nnsk_b3.600_c1.600_w0.300.pth
|   |-- ... ...
|   |-- latest_nnsk_b3.600_c3.599_w0.300.json
|   `-- latest_nnsk_b3.600_c3.599_w0.300.pth
|-- log
|   `-- log.txt
`-- train_config.json
```
We finally get the `latest_nnsk_b3.600_c3.599_w0.300.pth` with more neighbours.
plot the result again:
```bash
dptb run -sk band.json  -i ./varycutoff/checkpoint/latest_nnsk_b3.600_c3.599_w0.300.pth -o ./band
```
<div align=center>
<img src="./examples/hBN/reference/3.varycutoff/results/band.png" width = "60%" height = "60%" alt="hBN Bands" align=center />
</div>

We can again increase more training epochs, using the larger cutoff checkpoint, and change the input using 
```json
    "train_options": {
        "num_epoch": 10000,
        "optimizer": {"lr":1e-3},
        "lr_scheduler": {
            "type": "exp",
            "gamma": 0.9995
        }
    }
    "model_options": {
        "skfunction": {
            "sk_cutoff": 3.6,
            "sk_decay_w": 0.3
        }
    }
```
We can get a better fitting result:
<div align=center>
<img src="./examples/hBN/reference/4.longtrain/results/band.png" width = "60%" height = "60%" alt="hBN Bands" align=center />
</div>

Now you have learnt the basis use of **DeePTB**, however, the advanced functions still need to be explored for accurate and flexible electron structure representation, such as:
- atomic orbitals
- environmental correction
- spin-orbit coupling (SOC)
- ...

Altogether, we can simulate the electronic structure of a crystal system in a dynamic trajectory. **DeePTB** is capable of handling atom movement, volume change under stress, SOC effect and can use DFT eigenvalues with different orbitals and xc functionals as training targets.