#!/bin/sh
conda activate deeptb
pip install .
pip install pytest
pytest ./dptb/tests/