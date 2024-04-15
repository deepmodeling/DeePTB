from dptb.utils.argcheck import common_options, train_options, data_options, model_options
from dptb.utils.argcheck import run_options, set_info_options
from dargs.dargs import Argument



def gen_doc_list(arglist, make_anchor=False, make_link=False):
    
    if isinstance(arglist, Argument):
        arglist = [arglist]
    assert isinstance(arglist, list)

    ptr = []
    for arg in arglist:
        ptr.append(arg.gen_doc(make_anchor=make_anchor, make_link=make_link))

    return  "\n\n".join(ptr)

path = "./docs/input_params/"

with open(f"{path}/index.rst", "w") as of:
    print('========================================\nFull Input Parameters\n========================================', file=of)
    print('', file=of)
    print('.. toctree::', file=of)
    print('   :maxdepth: 2', file=of)
    print('', file=of)
    print('   common_options', file=of)
    print('   train_options', file=of)
    print('   model_options', file=of)
    print('   data_options', file=of)
    print('   run_options', file=of)
    print('   set_info', file=of)


ops = common_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open(f"{path}/common_options.rst", "w") as of:
    print('========================================\nCommon Options\n========================================', file=of)
    print(docstr, file=of)


ops = train_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open(f"{path}/train_options.rst", "w") as of:
    print('========================================\nTrain Options\n========================================', file=of)
    print(docstr, file=of)

ops = model_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open(f"{path}/model_options.rst", "w") as of:
    print('========================================\nModel Options\n========================================', file=of)
    print(docstr, file=of)

ops = data_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open(f"{path}/data_options.rst", "w") as of:
    print('========================================\nData Options\n========================================', file=of)
    print(docstr, file=of)


ops = run_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open(f"{path}/run_options.rst", "w") as of:
    print('========================================\nRun Options\n========================================', file=of)
    print(docstr, file=of)


ops = set_info_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open(f"{path}/set_info.rst", "w") as of:
    print('========================================\nSet Info\n========================================', file=of)
    print(docstr, file=of)