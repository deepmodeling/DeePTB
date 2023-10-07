from typing import List, Callable
from dargs import dargs, Argument, Variant, ArgumentEncoder


nnsk_model_config_checklist = ['unit','skfunction-skformula']
nnsk_model_config_updatelist = ['sknetwork-sk_hop_nhidden', 'sknetwork-sk_onsite_nhidden', 'sknetwork-sk_soc_nhidden']
dptb_model_config_checklist = ['dptb-if_batch_normalized', 'dptb-hopping_net_type', 'dptb-soc_net_type', 'dptb-env_net_type', 'dptb-onsite_net_type', 'dptb-hopping_net_activation', 'dptb-soc_net_activation', 'dptb-env_net_activation', 'dptb-onsite_net_activation', 
                        'dptb-hopping_net_neuron', 'dptb-env_net_neuron', 'dptb-soc_net_neuron', 'dptb-onsite_net_neuron', 'dptb-axis_neuron', 'skfunction-skformula', 'sknetwork-sk_onsite_nhidden', 
                        'sknetwork-sk_hop_nhidden']


def init_model():
    doc_path = "The model checkpoint path"
    doc_interpolate = r"Use linear interpolation of hoppings between single atom types to init the inter type atom's hopping; Default: `False`"

    args = [
        Argument("path", [list, str, None], optional = True, default=None, doc = doc_path),
        Argument("interpolate", bool, optional = True, default=False, doc = doc_interpolate)
    ]
    doc_init_model = ""
    return Argument("init_model", dict, optional = True, default={}, sub_fields=args, doc = doc_init_model)

def common_options():
    doc_device = "The device to run the calculation, choose among `cpu` and `cuda[:int]`, Default: `cpu`"
    doc_dtype = "The digital number's precison, choose among: \n\n\
            - `float32`: indicating torch.float32\n\n\
            - `float64`: indicating torch.float64\n\n\
            Default: `float32`\n\n"
    doc_onsitemode = r"The onsite correction mode, the onsite energy is expressed as the energy of isolated atoms plus the model correction, the correction mode are:\n\n\
            - `strain`: The strain mode correct the onsite matrix densly by $$H_{i,i}^{lm,l^\prime m^\prime} = \epsilon_l^0 \delta_{ll^\prime}\delta_{mm^\prime} + \sum_p \sum_{\zeta} \Big[ \mathcal{U}_{\zeta}(\hat{\br}_{ip}) \ \epsilon_{ll^\prime \zeta} \Big]_{mm^\prime}$$ which is also parameterized as a set of Slater-Koster like integrals.\n\n\
            - `uniform`: The correction is a energy shift respect of orbital of each atom. Which is formally written as: \n\n\
                  $$H_{i,i}^{lm,l^\prime m^\prime} = (\epsilon_l^0+\epsilon_l^\prime) \delta_{ll^\prime}\delta_{mm^\prime}$$ Where $\epsilon_l^0$ is the isolated energy level from the DeePTB onsite database, and $\epsilon_l^\prime$ is the parameters to fit. E.p. \n\n\
            - `split`: (not recommanded) The split onsite mode correct onsite hamiltonian with a magnetic quantum number dependent form, which violate the rotation equivariace, but some times can be effective. The formula is: \
                $$H_{i,i}^{lm,l^\prime m^\prime} = (\epsilon_l^0+\epsilon_{lm}^\prime) \delta_{ll^\prime}\delta_{mm^\prime}$$ \n\n\
            Default: `none`"
    doc_onsite_cutoff = "The cutoff-range considered when using strain mode correction. Out of which the atom are assume to have no effect on current atom's onsite energy."
    doc_bond_cutoff = "The cutoff-range of bond hoppings, beyond which it assume the atom pairs have 0 hopping integrals."
    doc_env_cutoff = "The cutoff-range of DeePTB environmental correction, recommand range is: (0.5*bond_cutoff, bond_cutoff)"
    doc_sk_file_path = ""
    doc_proj_atom_neles = "Number of electron considered atoms of the system."
    doc_proj_atom_anglr_m = "The atomic orbitals used to construct the basis. E.p. {'A':'2s','2p','s*','B':'3s','3p' }"
    doc_atomtype = "The list of atom type consist in the system."
    doc_time_symm = "Determine whether time symmetry is conserved, if set to be True, the eigenvalues on -k and k point is considered equal. Default: `True`"
    doc_soc = "Determine whether soc effect is modeled. If True, the soc network setting in model options need to be setted. Default: `False`"
    doc_unit = "Determine the unit of Tight-Binding parameters learned in DeePTB. Can be `eV`, `Hartree` or `Rothberg`. It will not affect the eigenvalues output form DeePTB, which is always in the unit of eV. Default: `Hartree`"

    args = [
        Argument("onsite_cutoff", float, optional = False, doc = doc_onsite_cutoff),
        Argument("bond_cutoff", float, optional = False, doc = doc_bond_cutoff),
        Argument("env_cutoff", float, optional = False, doc = doc_env_cutoff),
        Argument("atomtype", list, optional = False, doc = doc_atomtype),
        Argument("proj_atom_neles", dict, optional = False, doc = doc_proj_atom_neles),
        Argument("proj_atom_anglr_m", dict, optional = False, doc = doc_proj_atom_anglr_m),
        Argument("device", str, optional = True, default="cpu", doc = doc_device),
        Argument("dtype", str, optional = True, default="float32", doc = doc_dtype),
        Argument("onsitemode", str, optional = True, default = "none", doc = doc_onsitemode),
        Argument("sk_file_path", str, optional = True, default="./", doc = doc_sk_file_path),
        Argument("time_symm", bool, optional = True, default=True, doc = doc_time_symm),
        Argument("soc", bool, optional=True, default=False, doc=doc_soc),
        Argument("unit", str, optional=True, default="Hartree", doc=doc_unit)
    ]

    doc_common_options = ""

    return Argument("common_options", dict, optional=False, sub_fields=args, sub_variants=[], doc=doc_common_options)


def train_options():
    doc_num_epoch = "Total number of training epochs. It is worth noted, if the model is reloaded with `-r` or `--restart` option, epoch which have been trained will counted from the time that the checkpoint is saved."
    doc_seed = "The random seed used to initialize the parameters and determine the shuffling order of datasets. Default: `3982377700`"
    doc_save_freq = "Frequency, or every how many iteration to saved the current model into checkpoints, The name of checkpoint is formulated as `latest|best_dptb|nnsk_b<bond_cutoff>_c<sk_cutoff>_w<sk_decay_w>`. Default: `10`"
    doc_validation_freq = "Frequency or every how many iteration to do model validation on validation datasets. Default: `10`"
    doc_display_freq = "Frequency, or every how many iteration to display the training log to screem. Default: `1`"
    doc_optimizer = "\
        The optimizer setting for selecting the gradient optimizer of model training. Optimizer supported includes `Adam`, `SGD` and `LBFGS` \n\n\
        For more information about these optmization algorithm, we refer to:\n\n\
        - `Adam`: [Adam: A Method for Stochastic Optimization.](https://arxiv.org/abs/1412.6980)\n\n\
        - `SGD`: [Stochastic Gradient Descent.](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)\n\n\
        - `LBFGS`: [On the limited memory BFGS method for large scale optimization.](http://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf) \n\n\
    "
    doc_lr_scheduler = "The learning rate scheduler tools settings, the lr scheduler is used to scales down the learning rate during the training process. Proper setting can make the training more stable and efficient. The supported lr schedular includes: `Exponential Decaying (exp)`, `Linear multiplication (linear)`"

    args = [
        Argument("num_epoch", int, optional=False, doc=doc_num_epoch),
        Argument("seed", int, optional=True, default=3982377700, doc=doc_seed),
        Argument("optimizer", dict, sub_fields=[], optional=True, default={}, sub_variants=[optimizer()], doc = doc_optimizer),
        Argument("lr_scheduler", dict, sub_fields=[], optional=True, default={}, sub_variants=[lr_scheduler()], doc = doc_lr_scheduler),
        Argument("save_freq", int, optional=True, default=10, doc=doc_save_freq),
        Argument("validation_freq", int, optional=True, default=10, doc=doc_validation_freq),
        Argument("display_freq", int, optional=True, default=1, doc=doc_display_freq)
    ]

    doc_train_options = "Options that defines the training behaviour of DeePTB."

    return Argument("train_options", dict, sub_fields=args, sub_variants=[], optional=False, doc=doc_train_options)


def Adam():
    doc_lr = "learning rate. Default: 1e-3"
    doc_betas = "coefficients used for computing running averages of gradient and its square Default: (0.9, 0.999)"
    doc_eps = "term added to the denominator to improve numerical stability, Default: 1e-8"
    doc_weight_decay = "weight decay (L2 penalty), Default: 0"
    doc_amsgrad = "whether to use the AMSGrad variant of this algorithm from the paper On the [Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) ,Default: False"

    return [
        Argument("lr", float, optional=True, default=1e-3, doc=doc_lr),
        Argument("betas", list, optional=True, default=[0.9, 0.999], doc=doc_betas),
        Argument("eps", float, optional=True, default=1e-8, doc=doc_eps),
        Argument("weight_decay", float, optional=True, default=0, doc=doc_weight_decay),
        Argument("amsgrad", bool, optional=True, default=False, doc=doc_amsgrad)
    ]

def SGD():
    doc_lr = "learning rate. Default: 1e-3"
    doc_weight_decay = "weight decay (L2 penalty), Default: 0"
    doc_momentum = "momentum factor Default: 0"
    doc_dampening = "dampening for momentum, Default: 0"
    doc_nesterov = "enables Nesterov momentum, Default: False"

    return [
        Argument("lr", float, optional=True, default=1e-3, doc=doc_lr),
        Argument("momentum", float, optional=True, default=0., doc=doc_momentum),
        Argument("weight_decay", float, optional=True, default=0., doc=doc_weight_decay),
        Argument("dampening", float, optional=True, default=0., doc=doc_dampening),
        Argument("nesterov", bool, optional=True, default=False, doc=doc_nesterov)
    ]

def optimizer():
    doc_type = "select type of optimizer, support type includes: `Adam`, `SGD` and `LBFGS`. Default: `Adam`"

    return Variant("type", [
            Argument("Adam", dict, Adam()),
            Argument("SGD", dict, SGD())
        ],optional=True, default_tag="Adam", doc=doc_type)

def ExponentialLR():
    doc_gamma = "Multiplicative factor of learning rate decay."

    return [
        Argument("gamma", float, optional=True, default=0.999, doc=doc_gamma)
    ]

def LinearLR():
    doc_start_factor = "The number we multiply learning rate in the first epoch. \
        The multiplication factor changes towards end_factor in the following epochs. Default: 1./3."
    doc_end_factor = "The number we multiply learning rate in the first epoch. \
    The multiplication factor changes towards end_factor in the following epochs. Default: 1./3."
    doc_total_iters = "The number of iterations that multiplicative factor reaches to 1. Default: 5."

    return [
        Argument("start_factor", float, optional=True, default=0.3333333, doc=doc_start_factor),
        Argument("end_factor", float, optional=True, default=0.3333333, doc=doc_end_factor),
        Argument("total_iters", int, optional=True, default=5, doc=doc_total_iters)
    ]

def lr_scheduler():
    doc_type = "select type of lr_scheduler, support type includes `exp`, `linear`"

    return Variant("type", [
            Argument("exp", dict, ExponentialLR()),
            Argument("linear", dict, LinearLR())
        ],optional=True, default_tag="exp", doc=doc_type)


def train_data_sub():
    doc_batch_size = "number of configurations used to update the model parameters in a step of optmization."
    doc_path = "the path of dataset folders"
    doc_prefix = "the prefix of dataset folder's name. The dataset is recommended to named as <prefix>.<suffix>, data with the same prefix will be loaded as the datasets."
    
    args = [
        Argument("batch_size", int, optional=False, doc=doc_batch_size),
        Argument("path", str, optional=False, doc=doc_path),
        Argument("prefix", str, optional=False, doc=doc_prefix)
    ]

    doc_train = ""

    return Argument("train", dict, optional=False, sub_fields=args, sub_variants=[], doc=doc_train)

def validation_data_sub():
    doc_batch_size = "number of configurations used to update the model parameters in a step of optmization."
    doc_path = "the path of dataset folders"
    doc_prefix = "the prefix of dataset folder's name. The dataset is recommended to named as <prefix>.<suffix>, data with the same prefix will be loaded as the datasets."
    
    args = [
        Argument("batch_size", int, optional=False, doc=doc_batch_size),
        Argument("path", str, optional=False, doc=doc_path),
        Argument("prefix", str, optional=False, doc=doc_prefix)
    ]

    doc_validation = ""

    return Argument("validation", dict, optional=False, sub_fields=args, sub_variants=[], doc=doc_validation)

def reference_data_sub():
    doc_batch_size = "number of configurations used to update the model parameters in a step of optmization."
    doc_path = "the path of dataset folders"
    doc_prefix = "the prefix of dataset folder's name. The dataset is recommended to named as <prefix>.<suffix>, data with the same prefix will be loaded as the datasets."

    args = [
        Argument("batch_size", int, optional=False, doc=doc_batch_size),
        Argument("path", str, optional=False, doc=doc_path),
        Argument("prefix", str, optional=False, doc=doc_prefix)
    ]

    doc_reference = ""
    
    return Argument("reference", dict, optional=False, sub_fields=args, sub_variants=[], doc=doc_reference)

def test_data_sub():
    doc_batch_size = "number of configurations used to update the model parameters in a step of optmization."
    doc_path = "the path of dataset folders"
    doc_prefix = "the prefix of dataset folder's name. The dataset is recommended to named as <prefix>.<suffix>, data with the same prefix will be loaded as the datasets."

    args = [
        Argument("batch_size", int, optional=False, doc=doc_batch_size),
        Argument("path", str, optional=False, doc=doc_path),
        Argument("prefix", str, optional=False, doc=doc_prefix)
    ]

    doc_reference = ""
    
    return Argument("test", dict, optional=False, sub_fields=args, default={}, sub_variants=[], doc=doc_reference)


def data_options():
    doc_use_reference = "Whether to use a reference dataset that jointly train the model. It acting as a constraint or normalization to make sure the model won't deviate too much from the reference data."

    args = [Argument("use_reference", bool, optional=False, doc=doc_use_reference),
        train_data_sub(),
        validation_data_sub(),
        reference_data_sub()
    ]

    doc_data_options = ""

    return Argument("data_options", dict, sub_fields=args, sub_variants=[], optional=False, doc=doc_data_options)

def test_data_options():

    args = [
        test_data_sub()
    ]

    doc_test_data_options = "parameters for dataset settings in testing"

    return Argument("data_options", dict, sub_fields=args, sub_variants=[], optional=False, doc=doc_test_data_options)


def sknetwork():
    doc_sk_hop_nhidden = "the size of nnsk's hidden neuron for each hopping parameter."
    doc_sk_onsite_nhidden = "the size of nnsk's hidden neuron for each onsite parameter."
    doc_sk_soc_nhidden = "the size of nnsk's hidden neuron for each soc parameters"

    args = [
        Argument("sk_hop_nhidden", int, optional=False, doc=doc_sk_hop_nhidden),
        Argument("sk_onsite_nhidden", int, optional=False, doc=doc_sk_onsite_nhidden),
        Argument("sk_soc_nhidden", [int, None], optional=True, default=None, doc=doc_sk_soc_nhidden)
    ]

    doc_sknetwork = ""

    return Argument("sknetwork", dict, optional=True, sub_fields=args, sub_variants=[], default={}, doc=doc_sknetwork)

def skfunction():
    doc_skformula = r"defines the analytic formula of hoppings like parameters, which depend explicitly on the length of the chemical bonds. Supported type includes: \n\n\
        - powerlaw: $V_{powerlaw}(r_{ij})=alpha_0 * (r_0/r_{ij})^{1+alpha_1}$\n\n\
        - varTang96: $V_{varTang96}(r_{ij})=alpha_0 * r_{ij}^(-\alpha_1) * exp(-\alpha_2 * (rij)^(\alpha_3))$\n\n\
        - custom: the user need to implement a custom formula in the dptb.nnsktb.SKFormula Class \n\n\
        Here $V_{powerlaw}(r) and V_{varTang96}(r)$ is the bond length $r_ij$ dependent formula of fitted parameters. $alpha_{0~3}$ are the parameters fitted by DeePTB. The attained $V(r)$ is then multiplied with a decaying function $f(r)=1+exp((r_{ij}-r_c)/\omega)$ defined by `sk_cutoff` an `sk_decay_w` parameters. Default: `powerlaw` \n\n\
            "
    doc_sk_cutoff = r"The decay param $r_c$ in $f(r)=1+exp((r_{ij}-r_c)/\omega)$, controls the range of the decay, support list input to move the boundary of devaying function from near to afar. Default: 6.0."
    doc_sk_decay_w = r"The decay param $\omega$ in $f(r)=1+exp((r_{ij}-r_c)/\omega)$, control how smooth the decay function is, support list input to move the decaying function from soft to hard. Default: 0.1."

    args = [
        Argument("skformula", str, optional=True, default="powerlaw", doc=doc_skformula),
        Argument("sk_cutoff", [float,int,list], optional=True, default=6.0, doc=doc_sk_cutoff),
        Argument("sk_decay_w", [float, list], optional=True, default=0.1, doc=doc_sk_decay_w)
    ]

    doc_skfunction = "The parameter to define the analytic function formula of the SK like integrals."

    return Argument("skfunction", dict, optional=True, sub_fields=args, sub_variants=[], default={}, doc=doc_skfunction)

def dptb():
    doc_soc_env = "button that allow environmental correction for soc parameters, used only when soc is open, Default: False"
    doc_axis_neuron = "The axis_neuron specifies the size of the submatrix of the embedding matrix, the axis matrix as explained in the [DeepPot-SE paper](https://arxiv.org/abs/1805.09003)."
    doc_onsite_net_neuron = r"The number of hidden neurons in the network for onsites $\langle i|H|i\rangle$ of `dptb` model. Default: `[128, 128, 256, 256]`"
    doc_env_net_neuron = "The number of hidden neurons in the environment embedding network of `dptb` model. Default: `[128, 128, 256, 256]`"
    doc_hopping_net_neuron = r"The number of hidden neurons in the network for hoppings $\langle i|H|j\rangle$ of `dptb` model. Default: `[128, 128, 256, 256]`"
    doc_onsite_net_activation = "The activation function for onsite networks. Default: `tanh`"
    doc_env_net_activation = "The activation function for environment embedding networks. Default: `tanh`"
    doc_hopping_net_activation = "The activation function for hopping networks. Default: `tanh`"
    doc_soc_net_activation = "The activation function for soc networks. Default: `tanh`"
    doc_soc_net_neuron = r"The number of hidden neurons in the network for soc $\lambda$ of `dptb` model. Default: `[128, 128, 256, 256]`"
    doc_soc_net_type = "The network type for soc, the value can be:\n\n\
        - `res`: for feedforward Network with residual connections, for more information about residual network, we refer to [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf). \n\n\
        - `ffn`: for feedforward Network."
    doc_onsite_net_type = "The network type for onsites."
    doc_env_net_type = "The network type for environment embeddings."
    doc_hopping_net_type = "The network type for hoppings."
    doc_if_batch_normalized = "Whether to use batch normalization after each layer in neural network. The batch normalization normalize the itermidiate values in neural network with the mean and variance estimated in the batch dimension. The batch here means the batch of onsite or hopping embeddings or position vectors that processed by neural network at one time computation, which is different from the batch defined in `data_options`. Default: False."

    args = [
        Argument("soc_env", bool, optional=True, default=False, doc=doc_soc_env),
        Argument("axis_neuron", int, optional=True, default=10, doc=doc_axis_neuron),
        Argument("onsite_net_neuron", list, optional=True, default=[128, 128, 256, 256], doc=doc_onsite_net_neuron),
        Argument("soc_net_neuron", list, optional=True, default=[128, 128, 256, 256], doc=doc_soc_net_neuron),
        Argument("env_net_neuron", list, optional=True, default=[128, 128, 256, 256], doc=doc_env_net_neuron),
        Argument("hopping_net_neuron", list, optional=True, default=[128, 128, 256, 256], doc=doc_hopping_net_neuron),
        Argument("onsite_net_activation", str, optional=True, default="tanh", doc=doc_onsite_net_activation),
        Argument("env_net_activation", str, optional=True, default="tanh", doc=doc_env_net_activation),
        Argument("hopping_net_activation", str, optional=True, default="tanh", doc=doc_hopping_net_activation),
        Argument("soc_net_activation", str, optional=True, default="tanh", doc=doc_soc_net_activation),
        Argument("onsite_net_type", str, optional=True, default="res", doc=doc_onsite_net_type),
        Argument("env_net_type", str, optional=True, default="res", doc=doc_env_net_type),
        Argument("hopping_net_type", str, optional=True, default="res", doc=doc_hopping_net_type),
        Argument("soc_net_type", str, optional=True, default="res", doc=doc_soc_net_type),
        Argument("if_batch_normalized", bool, optional=True, default=False, doc=doc_if_batch_normalized)
    ]

    doc_dptb = "The parameters for `dptb` model, which maps the environmental information to Tight-Binding parameters."

    return Argument("dptb", dict, optional=True, sub_fields=args, sub_variants=[], default={}, doc=doc_dptb)


def model_options():

    doc_model_options = "The parameters to define the `nnsk` and `dptb` model."

    return Argument("model_options", dict, sub_fields=[skfunction(), sknetwork(), dptb()], sub_variants=[], optional=False, doc=doc_model_options)


def loss_options():
    doc_losstype = "The loss function type, defined by a string like `<fitting target>_<loss type>`, Default: `eigs_l2dsf`. supported loss functions includes:\n\n\
    - `eig_l2`: The l2 norm of predicted and labeled eigenvalues.\n\n\
    - `eigs_l2d`: The l2 norm and the random differences of the predicted and labeled eigenvalues.\n\n\
    - `block_l2`: \n\n\
        Notice: The loss option define here only affect the training loss function, the loss for evaluation will always be `eig_l2`, as it compute the standard MSE of fitted eigenvalues."
    doc_sortstrength = ""
    doc_nkratio = "The ratio is `null` or a positive float value smaller than `1.0`. If equals some float type, DeePTB will randomly select 100*ratio % of eigenvalues to compute the error and backpropagate to train the models. Default: None."

    args = [
        Argument("losstype", str, optional=True, doc=doc_losstype, default='eigs_l2dsf'),
        Argument("sortstrength", list, optional=True, doc=doc_sortstrength,default=[0.01,0.01]),
        Argument("nkratio", [float,None], optional=True, doc=doc_nkratio, default=None)
    ]

    doc_loss_options = ""
    return Argument("loss_options", dict, sub_fields=args, sub_variants=[], optional=True, default={}, doc=doc_loss_options)


def normalize(data):

    ini = init_model()

    co = common_options()
    tr = train_options()
    da = data_options()
    mo = model_options()
    lo = loss_options()

    base = Argument("base", dict, [ini, co, tr, da, mo, lo])
    data = base.normalize_value(data)
    # data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=True)

    return data

def normalize_test(data):

    ini = init_model()

    co = common_options()
    da = test_data_options()
    mo = model_options()
    lo = loss_options()

    base = Argument("base", dict, [ini, co, da, mo, lo])
    data = base.normalize_value(data)
    # data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=True)

    return data

def task_options():
    doc_task = "The string define the task DeePTB conduct, includes:\n\n\
        - `band`: for band structure plotting. \n\n\
        - `dos`: for density of states plotting.\n\n\
        - `pdos`: for projected density of states plotting.\n\n\
        - `FS2D`: for 2D fermi-surface plotting.\n\n\
        - `FS3D`: for 3D fermi-surface plotting.\n\n\
        - `write_sk`: for transcript the nnsk model to standard sk parameter table\n\n\
        - `ifermi`: \n\n"
    return Variant("task", [
            Argument("band", dict, band()),
            Argument("dos", dict, dos()),
            Argument("pdos", dict, pdos()),
            Argument("FS2D", dict, FS2D()),
            Argument("FS3D", dict, FS3D()),
            Argument("write_sk", dict, write_sk()),
            Argument("ifermi", dict, ifermi()),
            Argument("negf", dict, negf())
        ],optional=True, default_tag="band", doc=doc_task)


def negf():
    doc_scf = ""
    doc_block_tridiagonal = ""
    doc_ele_T = ""
    doc_unit = ""
    doc_scf_options = ""
    doc_stru_options = ""
    doc_poisson_options = ""
    doc_sgf_solver = ""
    doc_espacing = ""
    doc_emin = ""
    doc_emax = ""
    doc_e_fermi = ""
    doc_eta_lead = ""
    doc_eta_device = ""
    doc_out_dos = ""
    doc_out_tc = ""
    doc_out_current = ""
    doc_out_current_nscf = ""
    doc_out_ldos = ""
    doc_out_density = ""
    doc_out_lcurrent = ""
    doc_density_options = ""
    doc_out_potential = ""

    return [
        Argument("scf", bool, optional=True, default=False, doc=doc_scf),
        Argument("block_tridiagonal", bool, optional=True, default=False, doc=doc_block_tridiagonal),
        Argument("ele_T", [float, int], optional=False, doc=doc_ele_T),
        Argument("unit", str, optional=True, default="Hartree", doc=doc_unit),
        Argument("scf_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[scf_options()], doc=doc_scf_options),
        Argument("stru_options", dict, optional=False, sub_fields=stru_options(), doc=doc_stru_options),
        Argument("poisson_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[poisson_options()], doc=doc_poisson_options),
        Argument("sgf_solver", str, optional=True, default="Sancho-Rubio", doc=doc_sgf_solver),
        Argument("espacing", [int, float], optional=False, doc=doc_espacing),
        Argument("emin", [int, float], optional=False, doc=doc_emin),
        Argument("emax", [int, float], optional=False, doc=doc_emax),
        Argument("e_fermi", [int, float], optional=False, doc=doc_e_fermi),
        Argument("density_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[density_options()], doc=doc_density_options),
        Argument("eta_lead", [int, float], optional=True, default=1e-5, doc=doc_eta_lead),
        Argument("eta_device", [int, float], optional=True, default=0., doc=doc_eta_device),
        Argument("out_dos", bool, optional=True, default=False, doc=doc_out_dos),
        Argument("out_tc", bool, optional=True, default=False, doc=doc_out_tc),
        Argument("out_density", bool, optional=True, default=False, doc=doc_out_density),
        Argument("out_potential", bool, optional=True, default=False, doc=doc_out_potential),
        Argument("out_current", bool, optional=True, default=False, doc=doc_out_current),
        Argument("out_current_nscf", bool, optional=True, default=False, doc=doc_out_current_nscf),
        Argument("out_ldos", bool, optional=True, default=False, doc=doc_out_ldos),
        Argument("out_lcurrent", bool, optional=True, default=False, doc=doc_out_lcurrent)
    ]

def stru_options():
    doc_kmesh = ""
    doc_pbc = ""
    doc_device = ""
    doc_lead_L = ""
    doc_lead_R = ""
    return [
        Argument("device", dict, optional=False, sub_fields=device(), doc=doc_device),
        Argument("lead_L", dict, optional=False, sub_fields=lead(), doc=doc_lead_L),
        Argument("lead_R", dict, optional=False, sub_fields=lead(), doc=doc_lead_R),
        Argument("kmesh", list, optional=True, default=[1,1,1], doc=doc_kmesh),
        Argument("pbc", list, optional=True, default=[False, False, False], doc=doc_pbc)
    ]

def device():
    doc_id=""
    doc_sort=""

    return [
        Argument("id", str, optional=False, doc=doc_id),
        Argument("sort", bool, optional=True, default=True, doc=doc_sort)
    ]

def lead():
    doc_id=""
    doc_voltage=""

    return [
        Argument("id", str, optional=False, doc=doc_id),
        Argument("voltage", [int, float], optional=False, doc=doc_voltage)
    ]

def scf_options():
    doc_mode = ""
    doc_PDIIS = ""

    return Variant("mode", [
        Argument("PDIIS", dict, PDIIS(), doc=doc_PDIIS)
        ], optional=True, default_tag="PDIIS", doc=doc_mode)

def PDIIS():
    doc_mixing_period = ""
    doc_step_size = ""
    doc_n_history = ""
    doc_abs_err = ""
    doc_rel_err = ""
    doc_max_iter = ""

    return [
        Argument("mixing_period", int, optional=True, default=3, doc=doc_mixing_period),
        Argument("step_size", [int, float], optional=True, default=0.05, doc=doc_step_size),
        Argument("n_history", int, optional=True, default=6, doc=doc_n_history),
        Argument("abs_err", [int, float], optional=True, default=1e-6, doc=doc_abs_err),
        Argument("rel_err", [int, float], optional=True, default=1e-4, doc=doc_rel_err),
        Argument("max_iter", int, optional=True, default=100, doc=doc_max_iter)
    ]

def poisson_options():
    doc_solver = ""
    doc_fmm = ""
    return Variant("solver", [
        Argument("fmm", dict, fmm(), doc=doc_fmm)
    ], optional=True, default_tag="fmm", doc=doc_solver)

def density_options():
    doc_method = ""
    doc_Ozaki = ""
    return Variant("method", [
        Argument("Ozaki", dict, Ozaki(), doc=doc_method)
    ], optional=True, default_tag="Ozaki", doc=doc_Ozaki)

def Ozaki():
    doc_M_cut = ""
    doc_R = ""
    doc_n_gauss = ""
    return [
        Argument("R", [int, float], optional=True, default=1e6, doc=doc_R),
        Argument("M_cut", int, optional=True, default=30, doc=doc_M_cut),
        Argument("n_gauss", int, optional=True, default=10, doc=doc_n_gauss),
    ]

def fmm():
    doc_err = ""

    return [
        Argument("err", [int, float], optional=True, default=1e-5, doc=doc_err)
    ]

def normalize_run(data):
    doc_property = ""
    doc_model_options = ""
    doc_device = ""
    doc_dtype = ""
    doc_onsitemode = ""
    doc_onsite_cutoff = ""
    doc_bond_cutoff = ""
    doc_env_cutoff = ""
    doc_sk_file_path = ""
    doc_proj_atom_neles = ""
    doc_proj_atom_anglr_m = ""
    doc_atomtype = ""
    doc_time_symm = ""
    doc_soc = ""
    doc_unit = ""
    doc_common_options = ""
    doc_structure = ""
    doc_use_correction = ""
 
    args = [
        Argument("onsite_cutoff", float, optional = False, doc = doc_onsite_cutoff),
        Argument("bond_cutoff", float, optional = False, doc = doc_bond_cutoff),
        Argument("env_cutoff", float, optional = False, doc = doc_env_cutoff),
        Argument("atomtype", list, optional = False, doc = doc_atomtype),
        Argument("proj_atom_neles", dict, optional = False, doc = doc_proj_atom_neles),
        Argument("proj_atom_anglr_m", dict, optional = False, doc = doc_proj_atom_anglr_m),
        Argument("device", str, optional = True, default="cpu", doc = doc_device),
        Argument("dtype", str, optional = True, default="float32", doc = doc_dtype),
        Argument("onsitemode", str, optional = True, default = "none", doc = doc_onsitemode),
        Argument("sk_file_path", str, optional = True, default="./", doc = doc_sk_file_path),
        Argument("time_symm", bool, optional = True, default=True, doc = doc_time_symm),
        Argument("soc", bool, optional=True, default=False, doc=doc_soc),
        Argument("unit", str, optional=True, default="Hartree", doc=doc_unit)
    ]

    co = Argument("common_options", dict, optional=True, sub_fields=args, sub_variants=[], doc=doc_common_options)
    ini = init_model()
    mo = Argument("model_options", dict, sub_fields=[skfunction(), sknetwork(), dptb()], sub_variants=[], optional=True, doc=doc_model_options)

    args = [
        ini,
        co,
        mo,
        Argument("structure", [str,None], optional=True, default=None, doc = doc_structure),
        Argument("use_correction", [str,None], optional=True, default=None, doc = doc_use_correction),
        Argument("task_options", dict, sub_fields=[], optional=True, sub_variants=[task_options()], doc = doc_property)
    ]

    base = Argument("base", dict, args)
    data = base.normalize_value(data)
    base.check_value(data, strict=True)
    
    return data
    

def band():
    doc_kline_type = ""
    doc_kpath = ""
    doc_klabels = ""
    doc_emin=""
    doc_emax=""
    doc_E_fermi = ""
    doc_ref_band = ""
    
    return [
        Argument("kline_type", str, optional=False, doc=doc_kline_type),
        Argument("kpath", [str,list], optional=False, doc=doc_kpath),
        Argument("klabels", list, optional=True, default=[''], doc=doc_klabels),
        Argument("E_fermi", [float, int, None], optional=True, doc=doc_E_fermi, default=None),
        Argument("emin", [float, int, None], optional=True, doc=doc_emin, default=None),
        Argument("emax", [float, int, None], optional=True, doc=doc_emax, default=None),
        Argument("nkpoints", int, optional=True, doc=doc_emax, default=0),
        Argument("ref_band", [str, None], optional=True, default=None, doc=doc_ref_band)
    ]


def dos():
    doc_mesh_grid = ""
    doc_gamma_center = ""
    doc_sigma = ""
    doc_npoints = ""
    doc_width = ""
    doc_E_fermi=""

    return [
        Argument("mesh_grid", list, optional=False, doc=doc_mesh_grid),
        Argument("sigma", float, optional=False, doc=doc_sigma),
        Argument("npoints", int, optional=False, doc=doc_npoints),
        Argument("width", list, optional=False, doc=doc_width),
        Argument("E_fermi", [float, int, None], optional=True, doc=doc_E_fermi, default=None),
        Argument("gamma_center", bool, optional=True, default=False, doc=doc_gamma_center)
    ]

def pdos():
    doc_mesh_grid = ""
    doc_gamma_center = ""
    doc_sigma = ""
    doc_npoints = ""
    doc_width = ""
    doc_E_fermi=""
    doc_atom_index = ""
    doc_orbital_index = ""

    return [
        Argument("mesh_grid", list, optional=False, doc=doc_mesh_grid),
        Argument("sigma", float, optional=False, doc=doc_sigma),
        Argument("npoints", int, optional=False, doc=doc_npoints),
        Argument("width", list, optional=False, doc=doc_width),
        Argument("E_fermi", [float, int, None], optional=True, doc=doc_E_fermi, default=None),
        Argument("atom_index", list, optional=False, doc=doc_atom_index),
        Argument("orbital_index", list, optional=False, doc=doc_orbital_index),
        Argument("gamma_center", bool, optional=True, default=False, doc=doc_gamma_center)
    ]

def FS2D():
    doc_mesh_grid = ""
    doc_E0 = ""
    doc_sigma = ""
    doc_intpfactor = ""

    return [
        Argument("mesh_grid", list, optional=False, doc=doc_mesh_grid),
        Argument("sigma", float, optional=False, doc=doc_sigma),
        Argument("E0", int, optional=False, doc=doc_E0),
        Argument("intpfactor", int, optional=False, doc=doc_intpfactor)
    ]

def FS3D():
    doc_mesh_grid = ""
    doc_E0 = ""
    doc_sigma = ""
    doc_intpfactor = ""

    return [
        Argument("mesh_grid", list, optional=False, doc=doc_mesh_grid),
        Argument("sigma", float, optional=False, doc=doc_sigma),
        Argument("E0", int, optional=False, doc=doc_E0),
        Argument("intpfactor", int, optional=False, doc=doc_intpfactor)
    ]


def ifermi():
    doc_fermi = ""
    doc_prop = ""
    doc_mesh_grid = ""
    doc_mu = ""
    doc_sigma = ""
    doc_intpfactor = ""
    doc_wigner_seitz = ""
    doc_nworkers = ""
    doc_plot_type = "plot_type: Method used for plotting. Valid options are: matplotlib, plotly, mayavi, crystal_toolkit."
    doc_use_gui=""
    doc_plot_fs_bands = ""
    doc_fs_plane = ""
    doc_fs_distanc= ""
    doc_color_properties ="""color_properties: Whether to use the properties to color the Fermi surface.
                If the properties is a vector then the norm of the properties will be
                used. Note, this will only take effect if the Fermi surface has
                properties. If set to True, the viridis colormap will be used.
                Alternative colormaps can be selected by setting ``color_properties``
                to a matplotlib colormap name. This setting will override the ``colors``
                option. For vector properties, the arrows are colored according to the
                norm of the properties by default. If used in combination with the
                ``projection_axis`` option, the color will be determined by the dot
                product of the properties with the projection axis."""
    doc_fs_plot_options=""
    doc_projection_axis = """projection_axis: Projection axis that can be used to calculate the color of
                vector properties. If None, the norm of the properties will be used,
                otherwise the color will be determined by the dot product of the
                properties with the projection axis. Only has an effect when used with
                the ``vector_properties`` option."""

    doc_velocity = ""
    doc_colormap = ""
    doc_prop_plane = ""
    doc_prop_distance=""
    doc_prop_plot_options=""
    doc_hide_surface = """hide_surface: Whether to hide the Fermi surface. Only recommended in combination with the ``vector_properties`` option."""
    doc_hide_labels ="""hide_labels: Whether to show the high-symmetry k-point labels."""
    doc_hide_cell = """hide_cell: Whether to show the reciprocal cell boundary."""
    doc_vector_spacing="""vector_spacing: The rough spacing between arrows. Uses a custom algorithm
                for resampling the Fermi surface to ensure that arrows are not too close
                together. Only has an effect when used with the ``vector_properties``
                option."""
    doc_azimuth="azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended by the position vector on a sphere projected on to the x-y plane."
    doc_elevation="The zenith angle of the viewpoint in degrees, i.e. the angle subtended by the position vector and the z-axis."
    doc_colors ="""The color specification for the iso-surfaces. Valid options are:
                - A single color to use for all Fermi surfaces, specified as a tuple of
                  rgb values from 0 to 1. E.g., red would be ``(1, 0, 0)``.
                - A list of colors, specified as above.
                - A dictionary of ``{Spin.up: color1, Spin.down: color2}``, where the
                  colors are specified as above.
                - A string specifying which matplotlib colormap to use. See
                  https://matplotlib.org/tutorials/colors/colormaps.html for more
                  information.
                - ``None``, in which case the default colors will be used.
                """

    """Defaults."""

    AZIMUTH = 45.0
    ELEVATION = 35.0
    VECTOR_SPACING = 0.2
    COLORMAP = "viridis"
    SYMPREC = 1e-3
    KTOL = 1e-5
    SCALE = 4

 
    plot_options=[
        Argument("colors", [str,dict,list,None], optional=True, default=None, doc=doc_colors),
        Argument("projection_axis", [list,None], optional=True, default=None, doc=doc_projection_axis),
        Argument("hide_surface", bool, optional=True, default=False, doc=doc_hide_surface),
        Argument("hide_labels", bool, optional=True, default=False, doc=doc_hide_labels),
        Argument("hide_cell", bool, optional=True, default=False, doc=doc_hide_cell),
        Argument("vector_spacing",float, optional=True, default=VECTOR_SPACING, doc=doc_vector_spacing),
        Argument("azimuth", float, optional=True, default=AZIMUTH, doc=doc_azimuth),
        Argument("elevation", float, optional=True, default=ELEVATION, doc=doc_elevation),
    ]


    plot_options_fs=[
        Argument("projection_axis", [list,None], optional=True, default=None, doc=doc_projection_axis)
    ]
    args_fermi = [
        Argument("mesh_grid", list, optional = False, default=[2,2,2], doc = doc_mesh_grid),
        Argument("mu", [float,int], optional = False, default=0.0, doc = doc_mu),
        Argument("sigma", float, optional = True, default=0.1, doc = doc_sigma),
        Argument("intpfactor", int, optional = False, default=1, doc = doc_intpfactor),
        Argument("wigner_seitz", bool, optional = True, default=True, doc = doc_wigner_seitz),
        Argument("nworkers", int, optional = True, default=-1, doc = doc_nworkers),
        Argument("plot_type", str, optional = True, default="plotly", doc = doc_plot_type),
        Argument("use_gui", bool, optional = True, default=False, doc = doc_use_gui),
        Argument("plot_fs_bands", bool, optional = True, default = False, doc = doc_plot_fs_bands),
        Argument("fs_plane", list, optional = True, default=[0,0,1], doc = doc_fs_plane),
        Argument("fs_distance", [int,float], optional = True, default=0, doc = doc_fs_distanc),
        Argument("plot_options", dict, optional=True, sub_fields=plot_options, sub_variants=[], default={}, doc=doc_fs_plot_options)
    ]


    args_prop = [
        Argument("velocity", bool, optional = True, default=False, doc = doc_velocity),
        Argument("color_properties", [str,bool], optional = True, default=False, doc = doc_color_properties),
        Argument("colormap", str, optional = True,default="viridis",doc = doc_colormap),
        Argument("prop_plane", list, optional = True, default=[0,0,1],doc = doc_prop_plane),
        Argument("prop_distance", [int,float], optional = True, default=0, doc = doc_prop_distance),
        Argument("plot_options", dict, optional = True, sub_fields=plot_options, sub_variants=[], default={}, doc = doc_prop_plot_options)
    ]

    fermiarg = Argument("fermisurface", dict, optional=False, sub_fields=args_fermi, sub_variants=[], default={}, doc=doc_fermi)
    prop = Argument("property", dict, optional=True, sub_fields=args_prop, sub_variants=[], default={}, doc=doc_prop)

    return [fermiarg, prop]

def write_sk():
    doc_thr = ""
    doc_format = ""

    return [
        Argument("format", str, optional=True, default="sktable",  doc=doc_format),
        Argument("thr", float, optional=True, default=1e-3, doc=doc_thr)
    ]


def host_normalize(data):

    ini = init_model()
    co = common_options()
    mo = model_options()

    base = Argument("base", dict, [ini, co, mo])
    data = base.normalize_value(data)
    # data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=False)

    return data


def normalize_bandinfo(data):
    doc_band_min = ""
    doc_band_max = ""
    doc_emin = ""
    doc_emax = ""
    doc_gap_penalty = ""
    doc_fermi_band = ""
    doc_loss_gap_eta = ""
    doc_eout_weight=""
    doc_weight = ""

    args = [
        Argument("band_min", int, optional=True, doc=doc_band_min, default=0),
        Argument("band_max", [int, None], optional=True, doc=doc_band_max, default=None),
        Argument("emin", [float, None], optional=True, doc=doc_emin,default=None),
        Argument("emax", [float, None], optional=True, doc=doc_emax,default=None),
        Argument("gap_penalty", bool, optional=True, doc=doc_gap_penalty, default=False),
        Argument("fermi_band", int, optional=True, doc=doc_fermi_band,default=0),
        Argument("loss_gap_eta", float, optional=True, doc=doc_loss_gap_eta, default=0.01),
        Argument("eout_weight", float, optional=True, doc=doc_eout_weight, default=0.00),
        Argument("weight", [int, float, list], optional=True, doc=doc_weight, default=1.)
    ]
    bandinfo = Argument("bandinfo", dict, sub_fields=args)
    data = bandinfo.normalize_value(data)
    bandinfo.check_value(data, strict=True)

    return data

