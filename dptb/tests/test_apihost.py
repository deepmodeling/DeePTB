import torch
import numpy as np
from dptb.plugins.init_nnsk import InitSKModel
from dptb.postprocess.NN2HRK import NN2HRK
from dptb.nnops.apihost import NNSKHost
from ase.io import read,write
from dptb.structure.structure import BaseStruct
import matplotlib.pyplot as plt

import pytest
import os
from dptb.nnops.nnapi import NNSK, DeePTB
from ase.io import read,write
from dptb.structure.structure import BaseStruct

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)



def test_apihost(root_directory):
    checkfile = f'{root_directory}/dptb/tests/data/hBN/checkpoint/best_nnsk.pth'
    nnskapi = NNSKHost(checkpoint=checkfile)
    nnskapi.register_plugin(InitSKModel())
    nnskapi.build()


def test_api_2HRK(root_directory):
    checkfile = f'{root_directory}/dptb/tests/data/hBN/checkpoint/best_nnsk.pth'
    nnskapi = NNSKHost(checkpoint=checkfile)
    nnskapi.register_plugin(InitSKModel())
    nnskapi.build()

    nnHrk = NN2HRK(apihost=nnskapi, mode='nnsk')
    