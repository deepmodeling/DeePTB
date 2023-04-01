from typing import List, Callable
from dargs import dargs, Argument, Variant, ArgumentEncoder


nnsk_model_config_checklist = ['unit','skfunction-skformula']
nnsk_model_config_updatelist = ['sknetwork-sk_hop_nhidden', 'sknetwork-sk_onsite_nhidden', 'sknetwork-sk_soc_nhidden']
dptb_model_config_checklist = ['dptb-if_batch_normalized', 'dptb-bond_net_type', 'dptb-soc_net_type', 'dptb-env_net_type', 'dptb-onsite_net_type', 'dptb-bond_net_activation', 'dptb-soc_net_activation', 'dptb-env_net_activation', 'dptb-onsite_net_activation', 
                        'dptb-bond_net_neuron', 'dptb-env_net_neuron', 'dptb-soc_net_neuron', 'dptb-onsite_net_neuron', 'dptb-axis_neuron', 'skfunction-skformula', 'sknetwork-sk_onsite_nhidden', 
                        'sknetwork-sk_hop_nhidden']


def init_model():
    doc_path = ""
    doc_interpolate = ""

    args = [
        Argument("path", [list, str, None], optional = True, default=None, doc = doc_path),
        Argument("interpolate", bool, optional = True, default=False, doc = doc_interpolate)
    ]
    doc_init_model = ""
    return Argument("init_model", dict, optional = True, default={}, sub_fields=args, doc = doc_init_model)

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
    doc_soc = ""
    doc_unit = ""

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
    doc_num_epoch = ""
    doc_seed = ""
    doc_save_freq = ""
    doc_validation_freq = ""
    doc_display_freq = ""
    doc_optimizer = ""
    doc_lr_scheduler = ""

    args = [
        Argument("num_epoch", int, optional=False, doc=doc_num_epoch),
        Argument("seed", int, optional=True, default=3982377700, doc=doc_seed),
        Argument("optimizer", dict, sub_fields=[], optional=True, default={}, sub_variants=[optimizer()], doc = doc_optimizer),
        Argument("lr_scheduler", dict, sub_fields=[], optional=True, default={}, sub_variants=[lr_scheduler()], doc = doc_lr_scheduler),
        Argument("save_freq", int, optional=True, default=10, doc=doc_save_freq),
        Argument("validation_freq", int, optional=True, default=10, doc=doc_validation_freq),
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
        Argument("amsgrad", bool, optional=True, default=False)

    ]

def SGD():

    return [
        Argument("lr", float, optional=True, default=1e-3),
        Argument("momentum", float, optional=True, default=0.),
        Argument("weight_decay", float, optional=True, default=0.),
        Argument("dampening", float, optional=True, default=0.),
        Argument("nesterov", bool, optional=True, default=False)

    ]

def optimizer():
    doc_type = ""

    return Variant("type", [
            Argument("Adam", dict, Adam()),
            Argument("SGD", dict, SGD())
        ],optional=True, default_tag="Adam", doc=doc_type)

def ExponentialLR():

    return [
        Argument("gamma", float, optional=True, default=0.999)
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
    doc_sk_soc_nhidden = ""

    args = [
        Argument("sk_hop_nhidden", int, optional=False, doc=doc_sk_hop_nhidden),
        Argument("sk_onsite_nhidden", int, optional=False, doc=doc_sk_onsite_nhidden),
        Argument("sk_soc_nhidden", [int, None], optional=True, default=None, doc=doc_sk_soc_nhidden)
    ]

    doc_sknetwork = ""

    return Argument("sknetwork", dict, optional=True, sub_fields=args, sub_variants=[], default={}, doc=doc_sknetwork)

def skfunction():
    doc_skformula = ""
    doc_sk_cutoff = ""
    doc_sk_decay_w = ""

    args = [
        Argument("skformula", str, optional=True, default="varTang96", doc=doc_skformula),
        Argument("sk_cutoff", [float,int], optional=True, default=6.0, doc=doc_sk_cutoff),
        Argument("sk_decay_w", float, optional=True, default=0.1, doc=doc_sk_decay_w)
    ]

    doc_skfunction = ""

    return Argument("skfunction", dict, optional=True, sub_fields=args, sub_variants=[], default={}, doc=doc_skfunction)

def dptb():
    doc_soc_env = "button that allow environmental correction for soc parameters, used only when soc is open"
    doc_axis_neuron = ""
    doc_onsite_net_neuron = ""
    doc_env_net_neuron = ""
    doc_bond_net_neuron = ""
    doc_onsite_net_activation = ""
    doc_env_net_activation = ""
    doc_bond_net_activation = ""
    doc_soc_net_activation = ""
    doc_soc_net_neuron = ""
    doc_soc_net_type = ""
    doc_onsite_net_type = ""
    doc_env_net_type = ""
    doc_bond_net_type = ""
    doc_if_batch_normalized = ""

    args = [
        Argument("soc_env", bool, optional=True, default=False, doc=doc_soc_env),
        Argument("axis_neuron", int, optional=True, default=10, doc=doc_axis_neuron),
        Argument("onsite_net_neuron", list, optional=True, default=[128, 128, 256, 256], doc=doc_onsite_net_neuron),
        Argument("soc_net_neuron", list, optional=True, default=[128, 128, 256, 256], doc=doc_onsite_net_neuron),
        Argument("env_net_neuron", list, optional=True, default=[128, 128, 256, 256], doc=doc_env_net_neuron),
        Argument("bond_net_neuron", list, optional=True, default=[128, 128, 256, 256], doc=doc_bond_net_neuron),
        Argument("onsite_net_activation", str, optional=True, default="tanh", doc=doc_onsite_net_activation),
        Argument("env_net_activation", str, optional=True, default="tanh", doc=doc_env_net_activation),
        Argument("bond_net_activation", str, optional=True, default="tanh", doc=doc_bond_net_activation),
        Argument("soc_net_activation", str, optional=True, default="tanh", doc=doc_soc_net_activation),
        Argument("onsite_net_type", str, optional=True, default="res", doc=doc_onsite_net_type),
        Argument("env_net_type", str, optional=True, default="res", doc=doc_env_net_type),
        Argument("bond_net_type", str, optional=True, default="res", doc=doc_bond_net_type),
        Argument("soc_net_type", str, optional=True, default="res", doc=doc_soc_net_type),
        Argument("if_batch_normalized", bool, optional=True, default=False, doc=doc_if_batch_normalized)
    ]

    doc_dptb = ""

    return Argument("dptb", dict, optional=True, sub_fields=args, sub_variants=[], default={}, doc=doc_dptb)


def model_options():

    doc_model_options = ""

    return Argument("model_options", dict, sub_fields=[skfunction(), sknetwork(), dptb()], sub_variants=[], optional=False, doc=doc_model_options)


def loss_options():
    doc_losstype = ""
    doc_sortstrength = ""
    doc_nkratio = ""

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

def task_options():
    doc_task = ""
    return Variant("task", [
            Argument("band", dict, band()),
            Argument("dos", dict, dos()),
            Argument("pdos", dict, pdos()),
            Argument("FS2D", dict, FS2D()),
            Argument("FS3D", dict, FS3D()),
            Argument("write_sk", dict, write_sk()),
            Argument("ifermi", dict, ifermi())
        ],optional=True, default_tag="band", doc=doc_task)

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
    
    return [
        Argument("kline_type", str, optional=False, doc=doc_kline_type),
        Argument("kpath", list, optional=False, doc=doc_kpath),
        Argument("klabels", list, optional=True, default=[''], doc=doc_klabels),
        Argument("E_fermi", [float, int, None], optional=True, doc=doc_E_fermi, default=None),
        Argument("emin", [float, int, None], optional=True, doc=doc_emin, default=None),
        Argument("emax", [float, int, None], optional=True, doc=doc_emax, default=None)
    ]


def dos():
    doc_mesh_grid = ""
    doc_gamma_center = ""
    doc_sigma = ""
    doc_npoints = ""
    doc_width = ""

    return [
        Argument("mesh_grid", list, optional=False, doc=doc_mesh_grid),
        Argument("sigma", float, optional=False, doc=doc_sigma),
        Argument("npoints", int, optional=False, doc=doc_npoints),
        Argument("width", list, optional=False, doc=doc_width),
        Argument("gamma_center", bool, optional=True, default=False, doc=doc_gamma_center)
    ]

def pdos():
    doc_mesh_grid = ""
    doc_gamma_center = ""
    doc_sigma = ""
    doc_npoints = ""
    doc_width = ""
    doc_atom_index = ""
    doc_orbital_index = ""

    return [
        Argument("mesh_grid", list, optional=False, doc=doc_mesh_grid),
        Argument("sigma", float, optional=False, doc=doc_sigma),
        Argument("npoints", int, optional=False, doc=doc_npoints),
        Argument("width", list, optional=False, doc=doc_width),
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
        Argument("sigma", float, optional = False, default=0.1, doc = doc_sigma),
        Argument("intpfactor", int, optional = False, default=1, doc = doc_intpfactor),
        Argument("wigner_seitz", bool, optional = True, default=True, doc = doc_wigner_seitz),
        Argument("nworkers", int, optional = True, default=-1, doc = doc_nworkers),
        Argument("plot_type", str, optional = True, default="plotly", doc = doc_plot_type),
        Argument("use_gui", bool, optional = True, default=False, doc = doc_use_gui),
        Argument("plot_fs_bands", bool, optional = True, default = False, doc = doc_plot_fs_bands),
        Argument("fs_plane", list, optional = True, default=[0,0,1], doc = doc_fs_plane),
        Argument("fs_distance", [int,float], optional = True, default=0, doc = doc_fs_distanc),
        Argument("color_properties", bool, optional = True, default=False, doc = doc_color_properties),
        Argument("plot_options", dict, optional=True, sub_fields=plot_options, sub_variants=[], default={}, doc=doc_fs_plot_options)
    ]


    args_prop = [
        Argument("velocity", bool, optional = True, default=True, doc = doc_velocity),
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

