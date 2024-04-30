#!/bin/sh
conda init
source activate deeptb
chown -R $(whoami) .git
chmod -R u+rwX .git
pip install .
pip install pytest
pytest ./dptb/tests/