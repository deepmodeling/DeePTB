#!/bin/bash
conda activate deeptb
pip install .
pip install pytest
pytest ./dptb/tests/