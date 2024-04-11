========================================
Data Options
========================================
.. _`data_options`: 

data_options: 
    | type: ``dict``
    | argument path: ``data_options``

    The options for dataset settings in training.

    .. _`data_options/train`: 

    train: 
        | type: ``dict``
        | argument path: ``data_options/train``

        The dataset settings for training.

        .. _`data_options/train/type`: 

        type: 
            | type: ``str``, optional, default: ``DefaultDataset``
            | argument path: ``data_options/train/type``

            The type of dataset.

        .. _`data_options/train/root`: 

        root: 
            | type: ``str``
            | argument path: ``data_options/train/root``

            This is where the dataset stores data files.

        .. _`data_options/train/prefix`: 

        prefix: 
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``data_options/train/prefix``

            The prefix of the folders under root, which will be loaded in dataset.

        .. _`data_options/train/separator`: 

        separator: 
            | type: ``str``, optional, default: ``.``
            | argument path: ``data_options/train/separator``

            the sepatator used to separate the prefix and suffix in the dataset directory. Default: '.'

        .. _`data_options/train/get_Hamiltonian`: 

        get_Hamiltonian: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_Hamiltonian``

            Choose whether the Hamiltonian blocks (and overlap blocks, if provided) are loaded when building dataset.

        .. _`data_options/train/get_overlap`: 

        get_overlap: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_overlap``

            Choose whether the overlap blocks are loaded when building dataset.

        .. _`data_options/train/get_DM`: 

        get_DM: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_DM``

            Choose whether the density matrix is loaded when building dataset.

        .. _`data_options/train/get_eigenvalues`: 

        get_eigenvalues: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_eigenvalues``

            Choose whether the eigenvalues and k-points are loaded when building dataset.

    .. _`data_options/validation`: 

    validation: 
        | type: ``dict``, optional
        | argument path: ``data_options/validation``

        The dataset settings for validation.

        .. _`data_options/validation/type`: 

        type: 
            | type: ``str``, optional, default: ``DefaultDataset``
            | argument path: ``data_options/validation/type``

            The type of dataset.

        .. _`data_options/validation/root`: 

        root: 
            | type: ``str``
            | argument path: ``data_options/validation/root``

            This is where the dataset stores data files.

        .. _`data_options/validation/prefix`: 

        prefix: 
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``data_options/validation/prefix``

            The prefix of the folders under root, which will be loaded in dataset.

        .. _`data_options/validation/separator`: 

        separator: 
            | type: ``str``, optional, default: ``.``
            | argument path: ``data_options/validation/separator``

            the sepatator used to separate the prefix and suffix in the dataset directory. Default: '.'

        .. _`data_options/validation/get_Hamiltonian`: 

        get_Hamiltonian: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_Hamiltonian``

            Choose whether the Hamiltonian blocks (and overlap blocks, if provided) are loaded when building dataset.

        .. _`data_options/validation/get_overlap`: 

        get_overlap: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_overlap``

            Choose whether the overlap blocks are loaded when building dataset.

        .. _`data_options/validation/get_DM`: 

        get_DM: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_DM``

            Choose whether the density matrix is loaded when building dataset.

        .. _`data_options/validation/get_eigenvalues`: 

        get_eigenvalues: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_eigenvalues``

            Choose whether the eigenvalues and k-points are loaded when building dataset.

    .. _`data_options/reference`: 

    reference: 
        | type: ``dict``, optional
        | argument path: ``data_options/reference``

        The dataset settings for reference.

        .. _`data_options/reference/type`: 

        type: 
            | type: ``str``, optional, default: ``DefaultDataset``
            | argument path: ``data_options/reference/type``

            The type of dataset.

        .. _`data_options/reference/root`: 

        root: 
            | type: ``str``
            | argument path: ``data_options/reference/root``

            This is where the dataset stores data files.

        .. _`data_options/reference/prefix`: 

        prefix: 
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``data_options/reference/prefix``

            The prefix of the folders under root, which will be loaded in dataset.

        .. _`data_options/reference/separator`: 

        separator: 
            | type: ``str``, optional, default: ``.``
            | argument path: ``data_options/reference/separator``

            the sepatator used to separate the prefix and suffix in the dataset directory. Default: '.'

        .. _`data_options/reference/get_Hamiltonian`: 

        get_Hamiltonian: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_Hamiltonian``

            Choose whether the Hamiltonian blocks (and overlap blocks, if provided) are loaded when building dataset.

        .. _`data_options/reference/get_overlap`: 

        get_overlap: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_overlap``

            Choose whether the overlap blocks are loaded when building dataset.

        .. _`data_options/reference/get_DM`: 

        get_DM: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_DM``

            Choose whether the density matrix is loaded when building dataset.

        .. _`data_options/reference/get_eigenvalues`: 

        get_eigenvalues: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_eigenvalues``

            Choose whether the eigenvalues and k-points are loaded when building dataset.

