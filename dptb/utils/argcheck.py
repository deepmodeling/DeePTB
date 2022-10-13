import doctest
from tracemalloc import get_traceback_limit
from typing import List, Callable

from dargs import dargs, Argument, Variant, ArgumentEncoder
from dptb.plugins.base_plugin import Plugin
from numpy import floating


def common_options():
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
        Argument("time_symm", bool, optional = True, default=True, doc = doc_time_symm)
    ]

    doc_common_options = ""

    return Argument("common_options", dict, sub_fields=args, sub_variants=[], doc=doc_common_options)


def train_options():
    doc_num_epoch = ""
    doc_seed = ""
    doc_save_freq = ""
    doc_test_freq = ""
    doc_display_freq = ""
    doc_optimizer = ""
    doc_lr_scheduler = ""

    args = [
        Argument("num_epoch", int, optional=False, doc=doc_num_epoch),
        Argument("seed", int, optional=True, default=3982377700, doc=doc_seed),
        Argument("optimizer", dict, sub_fields=[], sub_variants=[optimizer()], doc = doc_optimizer),
        Argument("lr_scheduler", dict, sub_fields=[], sub_variants=[lr_scheduler()], doc = doc_lr_scheduler),
        Argument("save_freq", int, optional=True, default=10, doc=doc_save_freq),
        Argument("test_freq", int, optional=True, default=10, doc=doc_test_freq),
        Argument("display_freq", int, optional=True, default=1, doc=doc_display_freq)
    ]

    doc_train_options = ""

    return Argument("train_options", dict, sub_fields=args, sub_variants=[], optional=False, doc=doc_train_options)


def Adam():

    return [
        Argument("lr", float, optional=True, default=1e-3),
        Argument("betas", list, optional=True, default=[0.9, 0.999]),
        Argument("eps", float, optional=True, default=1e-8),
        Argument("weight_decay", float, optional=True, default=0),
        Argument("amsgrad", bool, optional=True, default=False),
        Argument("maximize", bool, optional=True, default=False)

    ]

def SGD():

    return [
        Argument("lr", float, optional=True, default=1e-3),
        Argument("momentum", float, optional=True, default=0.),
        Argument("weight_decay", float, optional=True, default=0.),
        Argument("dampening", float, optional=True, default=0.),
        Argument("nesterov", bool, optional=True, default=False),
        Argument("maximize", bool, optional=True, default=False)

    ]

def optimizer():
    doc_type = ""

    return Variant("type", [
            Argument("Adam", dict, Adam()),
            Argument("SGD", dict, SGD())
        ],optional=True, default_tag="Adam", doc=doc_type)

def ExponentialLR():

    return [
        Argument("gamma", float, optional=False)
    ]

def lr_scheduler():
    doc_type = ""

    return Variant("type", [
            Argument("Exp", dict, ExponentialLR())
        ],optional=True, default_tag="Exp", doc=doc_type)


def train_data_sub():
    doc_batch_size = ""
    doc_path = ""
    doc_prefix = ""
    
    args = [
        Argument("batch_size", int, optional=False, doc=doc_batch_size),
        Argument("path", str, optional=False, doc=doc_path),
        Argument("prefix", str, optional=False, doc=doc_prefix)
    ]

    doc_train = ""

    return Argument("train", dict, optional=False, sub_fields=args, sub_variants=[], doc=doc_train)

def validation_data_sub():
    doc_batch_size = ""
    doc_path = ""
    doc_prefix = ""
    
    args = [
        Argument("batch_size", int, optional=False, doc=doc_batch_size),
        Argument("path", str, optional=False, doc=doc_path),
        Argument("prefix", str, optional=False, doc=doc_prefix)
    ]

    doc_validation = ""

    return Argument("validation", dict, optional=False, sub_fields=args, sub_variants=[], doc=doc_validation)

def reference_data_sub():
    doc_batch_size = ""
    doc_path = ""
    doc_prefix = ""

    args = [
        Argument("batch_size", int, optional=False, doc=doc_batch_size),
        Argument("path", str, optional=False, doc=doc_path),
        Argument("prefix", str, optional=False, doc=doc_prefix)
    ]

    doc_reference = ""
    
    return Argument("reference", dict, optional=False, sub_fields=args, sub_variants=[], doc=doc_reference)

def data_options():
    doc_use_reference = ""

    args = [Argument("use_reference", bool, optional=False, doc=doc_use_reference),
        train_data_sub(),
        validation_data_sub(),
        reference_data_sub()
    ]

    doc_data_options = ""

    return Argument("data_options", dict, sub_fields=args, sub_variants=[], optional=False, doc=doc_data_options)
        

def sknetwork():
    doc_sk_hop_nhidden = ""
    doc_sk_onsite_nhidden = ""

    args = [
        Argument("sk_hop_nhidden", int, optional=False, doc=doc_sk_hop_nhidden),
        Argument("sk_onsite_nhidden", int, optional=False, doc=doc_sk_onsite_nhidden),
    ]

    doc_sknetwork = ""

    return Argument("sknetwork", dict, optional=True, sub_fields=args, sub_variants=[], default={}, doc=doc_sknetwork)

def skfunction():
    doc_skformula = ""
    doc_sk_cutoff = ""
    doc_sk_decay_w = ""

    args = [
        Argument("skformula", str, optional=True, default="varTang96", doc=doc_skformula),
        Argument("sk_cutoff", float, optional=True, default=6.0, doc=doc_sk_cutoff),
        Argument("sk_decay_w", float, optional=True, default=0.1, doc=doc_sk_decay_w)
    ]

    doc_skfunction = ""

    return Argument("skfunction", dict, optional=True, sub_fields=args, sub_variants=[], default={}, doc=doc_skfunction)

def dptb():

    pass

def model_options():

    doc_model_options = ""

    return Argument("model_options", dict, sub_fields=[skfunction(), sknetwork()], sub_variants=[], optional=False, doc=doc_model_options)


def loss_options():
    doc_band_min = ""
    doc_band_max = ""
    doc_ref_band_min = ""
    doc_ref_band_max = ""
    doc_emin = ""
    doc_emax = ""
    doc_sigma = ""
    doc_numomega = ""
    doc_sortstrength = ""
    doc_gap_penalty = ""
    doc_fermi_band = ""
    doc_loss_gap_eta = ""

    args = [
        Argument("band_min", int, optional=False, doc=doc_band_min),
        Argument("band_max", int, optional=False, doc=doc_band_max),
        Argument("ref_band_min", int, optional=False, doc=doc_ref_band_min),
        Argument("ref_band_max", int, optional=False, doc=doc_ref_band_max),
        Argument("emin", int, optional=False, doc=doc_emin),
        Argument("emax", int, optional=False, doc=doc_emax),
        Argument("sigma", int, optional=False, doc=doc_sigma),
        Argument("num_omega", int, optional=False, doc=doc_numomega),
        Argument("sortstrength", list, optional=False, doc=doc_sortstrength),
        Argument("gap_penalty", bool, optional=False, doc=doc_gap_penalty),
        Argument("fermi_band", int, optional=False, doc=doc_fermi_band),
        Argument("loss_gap_eta", float, optional=False, doc=doc_loss_gap_eta),
    ]

    doc_loss_options = ""
    return Argument("loss_options", dict, sub_fields=args, sub_variants=[], optional=False, doc=doc_loss_options)

def normalize(data):

    co = common_options()
    tr = train_options()
    da = data_options()
    mo = model_options()
    lo = loss_options()

    base = Argument("base", dict, [co, tr, da, mo, lo])
    data = base.normalize_value(data)
    # data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=True)

    return data

