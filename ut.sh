#!/bin/sh
conda init
source activate deeptb
pip install .
pip install pytest
pytest ./dptb/tests/