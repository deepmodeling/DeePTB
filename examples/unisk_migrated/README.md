# UniSK Migrated Examples

These examples were copied from UniSK to help inspect and manually validate the DFTB/SCC migration in DeePTB.

Most imports were mechanically changed from `dpusk` to `dptb`, and obvious absolute/local SK paths were adjusted to use data inside this directory.

Recommended notebooks for the migrated DeePTB functionality:

- `benchmark_eos_dftbp.ipynb`: SCC EOS comparison against DFTB+ reference data in `dptb/tests/data/dftb`.
- `hBN_scc/test_scc_hBN.ipynb`: hBN SCC workflow and band calculation.
- `hBN_scc/test_ewald.ipynb`: Ewald/Coulomb matrix checks.
- `hBN_dftb/test_total_energy.ipynb`: DFTB total energy workflow.
- `hBN_dftb/test_gamma.ipynb` and `hBN_dftb/test_pot.ipynb`: lower-level gamma/potential inspection.

Known caveat:

- `hBN_dftb/hotapi.ipynb` contains atomic-DFT SK-table generation via `dptb.nn.dftb.atomicdft.DFT2SKTable`. That atomic-DFT feature was intentionally not part of the current DeePTB migration, so the notebook is preserved as source context rather than guaranteed to run end to end.
