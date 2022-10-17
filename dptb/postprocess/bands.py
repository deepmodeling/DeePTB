from dptb.utils.argcheck import normalize
from dptb.utils.tools import j_loader
from dptb.nnops.apihost import NNSKHost
from dptb.postprocess.NN2HRK import NN2HRK

def bandcalc(mode, INPUT, structure):
    jdata = j_loader(INPUT)
    jdata = normalize(jdata)

    if mode == 'nnsk':
        apihost = NNSKHost(jdata)
    nn2hrk = NN2HRK(apihost=apihost,mode=mode)
    nn2hrk.update_struct(structure=structure)
    

