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


# 1. common_options
ops = common_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open("./input_params/common_options.rst", "w") as of:
    print('========================================\nCommon Options\n========================================', file=of)
    print(docstr, file=of)
print("1. updated common_options.rst")

# 2. train_options
ops = train_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open("./input_params/train_options.rst", "w") as of:
    print('========================================\nTrain Options\n========================================', file=of)
    print(docstr, file=of)
print("2. updated train_options.rst")

# 3. model_options
ops = model_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open("./input_params/model_options.rst", "w") as of:
    print('========================================\nModel Options\n========================================', file=of)
    print(docstr, file=of)
print("3. updated model_options.rst")

# 4. data_options
ops = data_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open("./input_params/data_options.rst", "w") as of:
    print('========================================\nData Options\n========================================', file=of)
    print(docstr, file=of)
print("4. updated data_options.rst")

# 5. run_options
ops = run_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open("./input_params/run_options.rst", "w") as of:
    print('========================================\nRun Options\n========================================', file=of)
    print(docstr, file=of)
print("5. updated run_options.rst")

# 6. set_info_options
ops = set_info_options()
docstr = gen_doc_list(ops, make_anchor=True, make_link=True)
with open("./input_params/set_info.rst", "w") as of:
    print('========================================\nSet Info\n========================================', file=of)
    print(docstr, file=of)
print("6. updated set_info.rst")
