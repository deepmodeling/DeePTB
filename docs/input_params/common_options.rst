========================================
Common Options
========================================
.. _`common_options`: 

common_options: 
    | type: ``dict``
    | argument path: ``common_options``

    .. _`common_options/basis`: 

    basis: 
        | type: ``dict``
        | argument path: ``common_options/basis``

        The atomic orbitals used to construct the basis. e.p. {'A':['2s','2p','s*'],'B':'[3s','3p']}

    .. _`common_options/overlap`: 

    overlap: 
        | type: ``bool``, optional, default: ``False``
        | argument path: ``common_options/overlap``

        Whether to calculate the overlap matrix. Default: False

    .. _`common_options/device`: 

    device: 
        | type: ``str``, optional, default: ``cpu``
        | argument path: ``common_options/device``

        The device to run the calculation, choose among `cpu` and `cuda[:int]`, Default: `cpu`

    .. _`common_options/dtype`: 

    dtype: 
        | type: ``str``, optional, default: ``float32``
        | argument path: ``common_options/dtype``

        The digital number's precison, choose among: 
                            Default: `float32`
                                - `float32`: indicating torch.float32
                                - `float64`: indicating torch.float64
                

    .. _`common_options/seed`: 

    seed: 
        | type: ``int``, optional, default: ``3982377700``
        | argument path: ``common_options/seed``

        The random seed used to initialize the parameters and determine the shuffling order of datasets. Default: `3982377700`

