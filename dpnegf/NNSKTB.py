import os
import re
import argparse
from shutil import copyfile

#from nnet.ParametersNN import Paras
from dpnegf.Parameters import Paras
from dpnegf.nnet.Model import Model



def deepTB(args:argparse.Namespace):
    if bool(re.match('tr',args.mode)):
        mode = 'train'
    elif bool(re.match('pr',args.mode)):
        mode = 'predict'
    elif bool(re.match('ts',args.mode)) or bool(re.match('test',args.mode)):
        mode = 'test'
    else:
        print('The mode is not correct. use --help for details')

    input_file = args.input_file
    fp = open(input_file)
    paras = Paras(fp)

    if mode == 'train':
        rsf = bool(re.match('f',args.run_status))
        rsr = bool(re.match('r',args.run_status))
        # ensure the task is either f or r, not both or neither.
        assert rsf ^ rsr, 'Error, wrong run status command.'

        paras.istrain = True
        paras.istest = False
        paras.ispredict = False

        if rsf:
            paras.trainmode = 'from_scratch'
            print('# run job from_scratch.')
            if os.path.exists(paras.save_checkpoint):
                print('Warning! checkpoint file already exits. But by running the job from_scratch, the checkpoint will be overwritten!')
                print('copy the existing checkpoint to ' + paras.save_checkpoint,paras.save_checkpoint + '_bak')
                copyfile(paras.save_checkpoint,paras.save_checkpoint + '_bak')

        else:
            paras.trainmode = 'restart'
            print('# restart to run the job.')

        mdl = Model(paras)
        mdl.train()

    elif mode == 'test':
        paras.istrain = False
        paras.istest = True
        paras.ispredict = False

        mdl = Model(paras)
        mdl.test()

def dpTBtrain(args:argparse.Namespace):
    """Train the NN based TB model.

    Args:
        args (argparse.Namespace): _description_
    """
    input_file = args.input_file
    fp = open(input_file)
    paras = Paras(fp,args.command)
    
    paras.istrain = True
    paras.istest = False
    paras.ispredict = False
    paras.TBmodel = 'nntb'


    if args.restart:
        paras.trainmode = 'restart'
        print('# restart to run the training job.')
        assert os.path.exists(paras.save_checkpoint), 'Error, the checkpoint file does not exist.'
    else:
        paras.trainmode = 'from_scratch'
        print('# run job from_scratch.')
        if os.path.exists(paras.save_checkpoint):
            print('Warning! checkpoint file already exits. But by running the job from_scratch, the checkpoint will be overwritten!')
            print('copy the existing checkpoint to ' + paras.save_checkpoint,paras.save_checkpoint + '_bak')
            copyfile(paras.save_checkpoint,paras.save_checkpoint + '_bak')        

    mdl = Model(paras)
    mdl.train()

def dpTBtest(args:argparse.Namespace):
    """Test the NN based TB model.

    Args:
        args (argparse.Namespace): args from command line.
    """
    input_file = args.input_file
    fp = open(input_file)
    paras = Paras(fp,args.command)
    
    paras.istrain = False
    paras.istest = True
    paras.ispredict = False
    paras.TBmodel = 'nntb'

    mdl = Model(paras)
    mdl.test()
