# readme

This guide provides instructions on how to set up the environment, train the SOC (Spin-Orbit Coupling) model, and visualize the band structures using the specific `soc-0310` branch.

## 1. Environment Setup

You can either clone an existing `dptb` environment or follow the [DeePTB Official Documentation](https://github.com/deepmodeling/deeptb) for a fresh installation.

## 2. Installation

Clone the specific SOC version which has been optimized to remove complex dependencies like `torch` during the initial setup.

```bash
# Clone the specific branch
git clone https://github.com/Franklalalala/DeePTB.git -b soc-0310
cd DeePTB

# Activate your environment and install in editable mode for debugging
python -m pip install -e .
```

## 3. Training the SOC Model

The training case uses a single-frame GaAs dataset located in the `delta_H_lmdb` directory.

1. Navigate to the training directory:
   ```bash
   cd example/soc_case/0310_soc_train
   ```
2. Start training using the provided configuration:
   ```bash
   # Replace with the standard DeePTB training command (e.g., python -m deeptb train input.json)
   deeptb train input.json
   ```

## 4. Band Structure Plotting

After training, you can plot the band structure to verify the results against ABACUS labels.

### Prerequisites
To run the plotting script, you must install the following packages:
* **abacus-test**: Use the `dev` branch.
  ```bash
  git clone https://github.com/Franklalalala/abacus-test.git -b dev
  cd abacus-test && pip install .
  ```
* **dprep**: Use the `main` branch.
  ```bash
  git clone https://github.com/DeePTB-Lab/dprep.git
  cd dprep && pip install .
  ```

### Execution
1. Navigate to the SOC example directory:
   ```bash
   cd example/soc_case/
   ```
2. Run the plotting script:
   ```bash
   python band_plot_0310.py
   ```
   The script will automatically load the best checkpoint file and generate comparison plots between DeePTB and ABACUS. The output figures will be saved in the `band_plots` directory.