========================================
Model Options
========================================
.. _`model_options`: 

model_options: 
    | type: ``dict``, optional
    | argument path: ``model_options``

    The parameters to define the `nnsk`,`mix` and `dptb` model.

    .. _`model_options/embedding`: 

    embedding: 
        | type: ``dict``, optional
        | argument path: ``model_options/embedding``

        The parameters to define the embedding model.


        Depending on the value of *method*, different sub args are accepted. 

        .. _`model_options/embedding/method`: 

        method:
            | type: ``str`` (flag key), default: ``se2``
            | argument path: ``model_options/embedding/method`` 
            | possible choices: |code:model_options/embedding[se2]|_, |code:model_options/embedding[baseline]|_, |code:model_options/embedding[deeph-e3]|_, |code:model_options/embedding[e3baseline_0]|_, |code:model_options/embedding[e3baseline_1]|_, |code:model_options/embedding[e3baseline_2]|_, |code:model_options/embedding[e3baseline_3]|_, |code:model_options/embedding[e3baseline_4]|_, |code:model_options/embedding[e3baseline_5]|_, |code:model_options/embedding[e3baseline_6]|_, |code:model_options/embedding[e3baseline_nonlocal]|_

            The parameters to define the embedding model.

            .. |code:model_options/embedding[se2]| replace:: ``se2``
            .. _`code:model_options/embedding[se2]`: `model_options/embedding[se2]`_
            .. |code:model_options/embedding[baseline]| replace:: ``baseline``
            .. _`code:model_options/embedding[baseline]`: `model_options/embedding[baseline]`_
            .. |code:model_options/embedding[deeph-e3]| replace:: ``deeph-e3``
            .. _`code:model_options/embedding[deeph-e3]`: `model_options/embedding[deeph-e3]`_
            .. |code:model_options/embedding[e3baseline_0]| replace:: ``e3baseline_0``
            .. _`code:model_options/embedding[e3baseline_0]`: `model_options/embedding[e3baseline_0]`_
            .. |code:model_options/embedding[e3baseline_1]| replace:: ``e3baseline_1``
            .. _`code:model_options/embedding[e3baseline_1]`: `model_options/embedding[e3baseline_1]`_
            .. |code:model_options/embedding[e3baseline_2]| replace:: ``e3baseline_2``
            .. _`code:model_options/embedding[e3baseline_2]`: `model_options/embedding[e3baseline_2]`_
            .. |code:model_options/embedding[e3baseline_3]| replace:: ``e3baseline_3``
            .. _`code:model_options/embedding[e3baseline_3]`: `model_options/embedding[e3baseline_3]`_
            .. |code:model_options/embedding[e3baseline_4]| replace:: ``e3baseline_4``
            .. _`code:model_options/embedding[e3baseline_4]`: `model_options/embedding[e3baseline_4]`_
            .. |code:model_options/embedding[e3baseline_5]| replace:: ``e3baseline_5``
            .. _`code:model_options/embedding[e3baseline_5]`: `model_options/embedding[e3baseline_5]`_
            .. |code:model_options/embedding[e3baseline_6]| replace:: ``e3baseline_6``
            .. _`code:model_options/embedding[e3baseline_6]`: `model_options/embedding[e3baseline_6]`_
            .. |code:model_options/embedding[e3baseline_nonlocal]| replace:: ``e3baseline_nonlocal``
            .. _`code:model_options/embedding[e3baseline_nonlocal]`: `model_options/embedding[e3baseline_nonlocal]`_

        .. |flag:model_options/embedding/method| replace:: *method*
        .. _`flag:model_options/embedding/method`: `model_options/embedding/method`_


        .. _`model_options/embedding[se2]`: 

        When |flag:model_options/embedding/method|_ is set to ``se2``: 

        .. _`model_options/embedding[se2]/rs`: 

        rs: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[se2]/rs``

            The soft cutoff where the smooth function starts.

        .. _`model_options/embedding[se2]/rc`: 

        rc: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[se2]/rc``

            The hard cutoff where the smooth function value ~0.0

        .. _`model_options/embedding[se2]/radial_net`: 

        radial_net: 
            | type: ``dict``
            | argument path: ``model_options/embedding[se2]/radial_net``

            network to build the descriptors.

            .. _`model_options/embedding[se2]/radial_net/neurons`: 

            neurons: 
                | type: ``list``
                | argument path: ``model_options/embedding[se2]/radial_net/neurons``

                the size of nn for descriptor

            .. _`model_options/embedding[se2]/radial_net/activation`: 

            activation: 
                | type: ``str``, optional, default: ``tanh``
                | argument path: ``model_options/embedding[se2]/radial_net/activation``

                activation

            .. _`model_options/embedding[se2]/radial_net/if_batch_normalized`: 

            if_batch_normalized: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``model_options/embedding[se2]/radial_net/if_batch_normalized``

                whether to turn on the batch normalization.

        .. _`model_options/embedding[se2]/n_axis`: 

        n_axis: 
            | type: ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[se2]/n_axis``

            the out axis shape of the deepmd-se2 descriptor.


        .. _`model_options/embedding[baseline]`: 

        When |flag:model_options/embedding/method|_ is set to ``baseline``: 

        .. _`model_options/embedding[baseline]/p`: 

        p: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[baseline]/p``

        .. _`model_options/embedding[baseline]/rc`: 

        rc: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[baseline]/rc``

        .. _`model_options/embedding[baseline]/n_basis`: 

        n_basis: 
            | type: ``int``
            | argument path: ``model_options/embedding[baseline]/n_basis``

        .. _`model_options/embedding[baseline]/n_radial`: 

        n_radial: 
            | type: ``int``
            | argument path: ``model_options/embedding[baseline]/n_radial``

        .. _`model_options/embedding[baseline]/n_sqrt_radial`: 

        n_sqrt_radial: 
            | type: ``int``
            | argument path: ``model_options/embedding[baseline]/n_sqrt_radial``

        .. _`model_options/embedding[baseline]/n_layer`: 

        n_layer: 
            | type: ``int``
            | argument path: ``model_options/embedding[baseline]/n_layer``

        .. _`model_options/embedding[baseline]/radial_net`: 

        radial_net: 
            | type: ``dict``
            | argument path: ``model_options/embedding[baseline]/radial_net``

            .. _`model_options/embedding[baseline]/radial_net/neurons`: 

            neurons: 
                | type: ``list``
                | argument path: ``model_options/embedding[baseline]/radial_net/neurons``

            .. _`model_options/embedding[baseline]/radial_net/activation`: 

            activation: 
                | type: ``str``, optional, default: ``tanh``
                | argument path: ``model_options/embedding[baseline]/radial_net/activation``

            .. _`model_options/embedding[baseline]/radial_net/if_batch_normalized`: 

            if_batch_normalized: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``model_options/embedding[baseline]/radial_net/if_batch_normalized``

        .. _`model_options/embedding[baseline]/hidden_net`: 

        hidden_net: 
            | type: ``dict``
            | argument path: ``model_options/embedding[baseline]/hidden_net``

            .. _`model_options/embedding[baseline]/hidden_net/neurons`: 

            neurons: 
                | type: ``list``
                | argument path: ``model_options/embedding[baseline]/hidden_net/neurons``

            .. _`model_options/embedding[baseline]/hidden_net/activation`: 

            activation: 
                | type: ``str``, optional, default: ``tanh``
                | argument path: ``model_options/embedding[baseline]/hidden_net/activation``

            .. _`model_options/embedding[baseline]/hidden_net/if_batch_normalized`: 

            if_batch_normalized: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``model_options/embedding[baseline]/hidden_net/if_batch_normalized``

        .. _`model_options/embedding[baseline]/n_axis`: 

        n_axis: 
            | type: ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[baseline]/n_axis``


        .. _`model_options/embedding[deeph-e3]`: 

        When |flag:model_options/embedding/method|_ is set to ``deeph-e3``: 

        .. _`model_options/embedding[deeph-e3]/irreps_embed`: 

        irreps_embed: 
            | type: ``str``, optional, default: ``64x0e``
            | argument path: ``model_options/embedding[deeph-e3]/irreps_embed``

        .. _`model_options/embedding[deeph-e3]/irreps_mid`: 

        irreps_mid: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[deeph-e3]/irreps_mid``

        .. _`model_options/embedding[deeph-e3]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[deeph-e3]/lmax``

        .. _`model_options/embedding[deeph-e3]/n_basis`: 

        n_basis: 
            | type: ``int``, optional, default: ``128``
            | argument path: ``model_options/embedding[deeph-e3]/n_basis``

        .. _`model_options/embedding[deeph-e3]/rc`: 

        rc: 
            | type: ``float``
            | argument path: ``model_options/embedding[deeph-e3]/rc``

        .. _`model_options/embedding[deeph-e3]/n_layer`: 

        n_layer: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[deeph-e3]/n_layer``


        .. _`model_options/embedding[e3baseline_0]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_0``: 

        .. _`model_options/embedding[e3baseline_0]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[e3baseline_0]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_0]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_0]/lmax``

        .. _`model_options/embedding[e3baseline_0]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``, optional, default: ``50``
            | argument path: ``model_options/embedding[e3baseline_0]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_0]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_0]/r_max``

        .. _`model_options/embedding[e3baseline_0]/n_layers`: 

        n_layers: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_0]/n_layers``

        .. _`model_options/embedding[e3baseline_0]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_0]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_0]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_0]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_0]/latent_kwargs`: 

        latent_kwargs: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[e3baseline_0]/latent_kwargs``

        .. _`model_options/embedding[e3baseline_0]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_0]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_0]/linear_after_env_embed`: 

        linear_after_env_embed: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_0]/linear_after_env_embed``

        .. _`model_options/embedding[e3baseline_0]/latent_resnet_update_ratios_learnable`: 

        latent_resnet_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_0]/latent_resnet_update_ratios_learnable``


        .. _`model_options/embedding[e3baseline_1]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_1``: 

        .. _`model_options/embedding[e3baseline_1]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[e3baseline_1]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_1]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_1]/lmax``

        .. _`model_options/embedding[e3baseline_1]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``, optional, default: ``50``
            | argument path: ``model_options/embedding[e3baseline_1]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_1]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_1]/r_max``

        .. _`model_options/embedding[e3baseline_1]/n_layers`: 

        n_layers: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_1]/n_layers``

        .. _`model_options/embedding[e3baseline_1]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_1]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_1]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_1]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_1]/latent_kwargs`: 

        latent_kwargs: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[e3baseline_1]/latent_kwargs``

        .. _`model_options/embedding[e3baseline_1]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_1]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_1]/linear_after_env_embed`: 

        linear_after_env_embed: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_1]/linear_after_env_embed``

        .. _`model_options/embedding[e3baseline_1]/latent_resnet_update_ratios_learnable`: 

        latent_resnet_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_1]/latent_resnet_update_ratios_learnable``


        .. _`model_options/embedding[e3baseline_2]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_2``: 

        .. _`model_options/embedding[e3baseline_2]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[e3baseline_2]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_2]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_2]/lmax``

        .. _`model_options/embedding[e3baseline_2]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``, optional, default: ``50``
            | argument path: ``model_options/embedding[e3baseline_2]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_2]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_2]/r_max``

        .. _`model_options/embedding[e3baseline_2]/n_layers`: 

        n_layers: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_2]/n_layers``

        .. _`model_options/embedding[e3baseline_2]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_2]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_2]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_2]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_2]/latent_kwargs`: 

        latent_kwargs: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[e3baseline_2]/latent_kwargs``

        .. _`model_options/embedding[e3baseline_2]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_2]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_2]/linear_after_env_embed`: 

        linear_after_env_embed: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_2]/linear_after_env_embed``

        .. _`model_options/embedding[e3baseline_2]/latent_resnet_update_ratios_learnable`: 

        latent_resnet_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_2]/latent_resnet_update_ratios_learnable``


        .. _`model_options/embedding[e3baseline_3]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_3``: 

        .. _`model_options/embedding[e3baseline_3]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[e3baseline_3]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_3]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_3]/lmax``

        .. _`model_options/embedding[e3baseline_3]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``, optional, default: ``50``
            | argument path: ``model_options/embedding[e3baseline_3]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_3]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_3]/r_max``

        .. _`model_options/embedding[e3baseline_3]/n_layers`: 

        n_layers: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_3]/n_layers``

        .. _`model_options/embedding[e3baseline_3]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_3]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_3]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_3]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_3]/latent_kwargs`: 

        latent_kwargs: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[e3baseline_3]/latent_kwargs``

        .. _`model_options/embedding[e3baseline_3]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_3]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_3]/linear_after_env_embed`: 

        linear_after_env_embed: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_3]/linear_after_env_embed``

        .. _`model_options/embedding[e3baseline_3]/latent_resnet_update_ratios_learnable`: 

        latent_resnet_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_3]/latent_resnet_update_ratios_learnable``


        .. _`model_options/embedding[e3baseline_4]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_4``: 

        .. _`model_options/embedding[e3baseline_4]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[e3baseline_4]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_4]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_4]/lmax``

        .. _`model_options/embedding[e3baseline_4]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``, optional, default: ``50``
            | argument path: ``model_options/embedding[e3baseline_4]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_4]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_4]/r_max``

        .. _`model_options/embedding[e3baseline_4]/n_layers`: 

        n_layers: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_4]/n_layers``

        .. _`model_options/embedding[e3baseline_4]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_4]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_4]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_4]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_4]/latent_kwargs`: 

        latent_kwargs: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[e3baseline_4]/latent_kwargs``

        .. _`model_options/embedding[e3baseline_4]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_4]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_4]/linear_after_env_embed`: 

        linear_after_env_embed: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_4]/linear_after_env_embed``

        .. _`model_options/embedding[e3baseline_4]/latent_resnet_update_ratios_learnable`: 

        latent_resnet_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_4]/latent_resnet_update_ratios_learnable``


        .. _`model_options/embedding[e3baseline_5]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_5``: 

        .. _`model_options/embedding[e3baseline_5]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``
            | argument path: ``model_options/embedding[e3baseline_5]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_5]/lmax`: 

        lmax: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_5]/lmax``

        .. _`model_options/embedding[e3baseline_5]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_5]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_5]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_5]/r_max``

        .. _`model_options/embedding[e3baseline_5]/n_layers`: 

        n_layers: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_5]/n_layers``

        .. _`model_options/embedding[e3baseline_5]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``10``
            | argument path: ``model_options/embedding[e3baseline_5]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_5]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_5]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_5]/cutoff_type`: 

        cutoff_type: 
            | type: ``str``, optional, default: ``polynomial``
            | argument path: ``model_options/embedding[e3baseline_5]/cutoff_type``

            The type of cutoff function. Default: polynomial

        .. _`model_options/embedding[e3baseline_5]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_5]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_5]/tp_radial_emb`: 

        tp_radial_emb: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_5]/tp_radial_emb``

            Whether to use tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_5]/tp_radial_channels`: 

        tp_radial_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_5]/tp_radial_channels``

            The number of channels in tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_5]/latent_channels`: 

        latent_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_5]/latent_channels``

            The number of channels in latent embedding.

        .. _`model_options/embedding[e3baseline_5]/latent_dim`: 

        latent_dim: 
            | type: ``int``, optional, default: ``256``
            | argument path: ``model_options/embedding[e3baseline_5]/latent_dim``

            The dimension of latent embedding.

        .. _`model_options/embedding[e3baseline_5]/res_update`: 

        res_update: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model_options/embedding[e3baseline_5]/res_update``

            Whether to use residual update.

        .. _`model_options/embedding[e3baseline_5]/res_update_ratios`: 

        res_update_ratios: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model_options/embedding[e3baseline_5]/res_update_ratios``

            The ratios of residual update, should in (0,1).

        .. _`model_options/embedding[e3baseline_5]/res_update_ratios_learnable`: 

        res_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_5]/res_update_ratios_learnable``

            Whether to make the ratios of residual update learnable.


        .. _`model_options/embedding[e3baseline_6]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_6``: 

        .. _`model_options/embedding[e3baseline_6]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``
            | argument path: ``model_options/embedding[e3baseline_6]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_6]/lmax`: 

        lmax: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_6]/lmax``

        .. _`model_options/embedding[e3baseline_6]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_6]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_6]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_6]/r_max``

        .. _`model_options/embedding[e3baseline_6]/n_layers`: 

        n_layers: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_6]/n_layers``

        .. _`model_options/embedding[e3baseline_6]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``10``
            | argument path: ``model_options/embedding[e3baseline_6]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_6]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_6]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_6]/cutoff_type`: 

        cutoff_type: 
            | type: ``str``, optional, default: ``polynomial``
            | argument path: ``model_options/embedding[e3baseline_6]/cutoff_type``

            The type of cutoff function. Default: polynomial

        .. _`model_options/embedding[e3baseline_6]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_6]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_6]/tp_radial_emb`: 

        tp_radial_emb: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_6]/tp_radial_emb``

            Whether to use tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_6]/tp_radial_channels`: 

        tp_radial_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_6]/tp_radial_channels``

            The number of channels in tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_6]/latent_channels`: 

        latent_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_6]/latent_channels``

            The number of channels in latent embedding.

        .. _`model_options/embedding[e3baseline_6]/latent_dim`: 

        latent_dim: 
            | type: ``int``, optional, default: ``256``
            | argument path: ``model_options/embedding[e3baseline_6]/latent_dim``

            The dimension of latent embedding.

        .. _`model_options/embedding[e3baseline_6]/res_update`: 

        res_update: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model_options/embedding[e3baseline_6]/res_update``

            Whether to use residual update.

        .. _`model_options/embedding[e3baseline_6]/res_update_ratios`: 

        res_update_ratios: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model_options/embedding[e3baseline_6]/res_update_ratios``

            The ratios of residual update, should in (0,1).

        .. _`model_options/embedding[e3baseline_6]/res_update_ratios_learnable`: 

        res_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_6]/res_update_ratios_learnable``

            Whether to make the ratios of residual update learnable.


        .. _`model_options/embedding[e3baseline_nonlocal]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_nonlocal``: 

        .. _`model_options/embedding[e3baseline_nonlocal]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_nonlocal]/lmax`: 

        lmax: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/lmax``

        .. _`model_options/embedding[e3baseline_nonlocal]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_nonlocal]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/r_max``

        .. _`model_options/embedding[e3baseline_nonlocal]/n_layers`: 

        n_layers: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/n_layers``

        .. _`model_options/embedding[e3baseline_nonlocal]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``10``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_nonlocal]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_nonlocal]/cutoff_type`: 

        cutoff_type: 
            | type: ``str``, optional, default: ``polynomial``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/cutoff_type``

            The type of cutoff function. Default: polynomial

        .. _`model_options/embedding[e3baseline_nonlocal]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_nonlocal]/tp_radial_emb`: 

        tp_radial_emb: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/tp_radial_emb``

            Whether to use tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_nonlocal]/tp_radial_channels`: 

        tp_radial_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/tp_radial_channels``

            The number of channels in tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_nonlocal]/latent_channels`: 

        latent_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/latent_channels``

            The number of channels in latent embedding.

        .. _`model_options/embedding[e3baseline_nonlocal]/latent_dim`: 

        latent_dim: 
            | type: ``int``, optional, default: ``256``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/latent_dim``

            The dimension of latent embedding.

        .. _`model_options/embedding[e3baseline_nonlocal]/res_update`: 

        res_update: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/res_update``

            Whether to use residual update.

        .. _`model_options/embedding[e3baseline_nonlocal]/res_update_ratios`: 

        res_update_ratios: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/res_update_ratios``

            The ratios of residual update, should in (0,1).

        .. _`model_options/embedding[e3baseline_nonlocal]/res_update_ratios_learnable`: 

        res_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/res_update_ratios_learnable``

            Whether to make the ratios of residual update learnable.

    .. _`model_options/prediction`: 

    prediction: 
        | type: ``dict``, optional
        | argument path: ``model_options/prediction``

        The parameters to define the prediction model


        Depending on the value of *method*, different sub args are accepted. 

        .. _`model_options/prediction/method`: 

        method:
            | type: ``str`` (flag key)
            | argument path: ``model_options/prediction/method`` 
            | possible choices: |code:model_options/prediction[sktb]|_, |code:model_options/prediction[e3tb]|_

            The options to indicate the prediction model. Can be sktb or e3tb.

            .. |code:model_options/prediction[sktb]| replace:: ``sktb``
            .. _`code:model_options/prediction[sktb]`: `model_options/prediction[sktb]`_
            .. |code:model_options/prediction[e3tb]| replace:: ``e3tb``
            .. _`code:model_options/prediction[e3tb]`: `model_options/prediction[e3tb]`_

        .. |flag:model_options/prediction/method| replace:: *method*
        .. _`flag:model_options/prediction/method`: `model_options/prediction/method`_


        .. _`model_options/prediction[sktb]`: 

        When |flag:model_options/prediction/method|_ is set to ``sktb``: 

        neural network options for prediction model.

        .. _`model_options/prediction[sktb]/neurons`: 

        neurons: 
            | type: ``list``
            | argument path: ``model_options/prediction[sktb]/neurons``

            neurons in the neural network.

        .. _`model_options/prediction[sktb]/activation`: 

        activation: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model_options/prediction[sktb]/activation``

            activation function.

        .. _`model_options/prediction[sktb]/if_batch_normalized`: 

        if_batch_normalized: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/prediction[sktb]/if_batch_normalized``

            if to turn on batch normalization


        .. _`model_options/prediction[e3tb]`: 

        When |flag:model_options/prediction/method|_ is set to ``e3tb``: 

        neural network options for prediction model.

        .. _`model_options/prediction[e3tb]/scales_trainable`: 

        scales_trainable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/prediction[e3tb]/scales_trainable``

            whether to scale the trianing target.

        .. _`model_options/prediction[e3tb]/shifts_trainable`: 

        shifts_trainable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/prediction[e3tb]/shifts_trainable``

            whether to shift the training target.

    .. _`model_options/nnsk`: 

    nnsk: 
        | type: ``dict``, optional
        | argument path: ``model_options/nnsk``

        The parameters to define the nnsk model.

        .. _`model_options/nnsk/onsite`: 

        onsite: 
            | type: ``dict``
            | argument path: ``model_options/nnsk/onsite``

            The onsite options to define the onsite of nnsk model.


            Depending on the value of *method*, different sub args are accepted. 

            .. _`model_options/nnsk/onsite/method`: 

            method:
                | type: ``str`` (flag key)
                | argument path: ``model_options/nnsk/onsite/method`` 
                | possible choices: |code:model_options/nnsk/onsite[strain]|_, |code:model_options/nnsk/onsite[uniform]|_, |code:model_options/nnsk/onsite[NRL]|_, |code:model_options/nnsk/onsite[none]|_

                The onsite correction mode, the onsite energy is expressed as the energy of isolated atoms plus the model correction, the correction mode are:
                                    Default: `none`: use the database onsite energy value.
                                    - `strain`: The strain mode correct the onsite matrix densly by $$H_{i,i}^{lm,l^\prime m^\prime} = \epsilon_l^0 \delta_{ll^\prime}\delta_{mm^\prime} + \sum_p \sum_{\zeta} \Big[ \mathcal{U}_{\zeta}(\hat{r}_{ip}) \ \epsilon_{ll^\prime \zeta} \Big]_{mm^\prime}$$ which is also parameterized as a set of Slater-Koster like integrals.

                                    - `uniform`: The correction is a energy shift respect of orbital of each atom. Which is formally written as: 
                                                $$H_{i,i}^{lm,l^\prime m^\prime} = (\epsilon_l^0+\epsilon_l^\prime) \delta_{ll^\prime}\delta_{mm^\prime}$$ Where $\epsilon_l^0$ is the isolated energy level from the DeePTB onsite database, and $\epsilon_l^\prime$ is the parameters to fit.
                                    - `NRL`: use the NRL-TB formula.
                

                .. |code:model_options/nnsk/onsite[strain]| replace:: ``strain``
                .. _`code:model_options/nnsk/onsite[strain]`: `model_options/nnsk/onsite[strain]`_
                .. |code:model_options/nnsk/onsite[uniform]| replace:: ``uniform``
                .. _`code:model_options/nnsk/onsite[uniform]`: `model_options/nnsk/onsite[uniform]`_
                .. |code:model_options/nnsk/onsite[NRL]| replace:: ``NRL``
                .. _`code:model_options/nnsk/onsite[NRL]`: `model_options/nnsk/onsite[NRL]`_
                .. |code:model_options/nnsk/onsite[none]| replace:: ``none``
                .. _`code:model_options/nnsk/onsite[none]`: `model_options/nnsk/onsite[none]`_

            .. |flag:model_options/nnsk/onsite/method| replace:: *method*
            .. _`flag:model_options/nnsk/onsite/method`: `model_options/nnsk/onsite/method`_


            .. _`model_options/nnsk/onsite[strain]`: 

            When |flag:model_options/nnsk/onsite/method|_ is set to ``strain``: 

            .. _`model_options/nnsk/onsite[strain]/rs`: 

            rs: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/onsite[strain]/rs``

                The smooth cutoff `fc` for strain model. rs is where fc = 0.5

            .. _`model_options/nnsk/onsite[strain]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/onsite[strain]/w``

                The decay factor of `fc` for strain and nrl model.


            .. _`model_options/nnsk/onsite[uniform]`: 

            When |flag:model_options/nnsk/onsite/method|_ is set to ``uniform``: 



            .. _`model_options/nnsk/onsite[NRL]`: 

            When |flag:model_options/nnsk/onsite/method|_ is set to ``NRL``: 

            .. _`model_options/nnsk/onsite[NRL]/rc`: 

            rc: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/onsite[NRL]/rc``

                The smooth cutoff of `fc` for nrl model, rc is where fc ~ 0.0

            .. _`model_options/nnsk/onsite[NRL]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/onsite[NRL]/w``

                The decay factor of `fc` for strain and nrl model.

            .. _`model_options/nnsk/onsite[NRL]/lda`: 

            lda: 
                | type: ``float``, optional, default: ``1.0``
                | argument path: ``model_options/nnsk/onsite[NRL]/lda``

                The lambda type encoding value in nrl model. now only support elementary substance


            .. _`model_options/nnsk/onsite[none]`: 

            When |flag:model_options/nnsk/onsite/method|_ is set to ``none``: 


        .. _`model_options/nnsk/hopping`: 

        hopping: 
            | type: ``dict``
            | argument path: ``model_options/nnsk/hopping``

            The hopping options to define the hopping of nnsk model.


            Depending on the value of *method*, different sub args are accepted. 

            .. _`model_options/nnsk/hopping/method`: 

            method:
                | type: ``str`` (flag key)
                | argument path: ``model_options/nnsk/hopping/method`` 
                | possible choices: |code:model_options/nnsk/hopping[powerlaw]|_, |code:model_options/nnsk/hopping[varTang96]|_, |code:model_options/nnsk/hopping[NRL0]|_, |code:model_options/nnsk/hopping[NRL1]|_, |code:model_options/nnsk/hopping[custom]|_

                The hopping formula. 
                                    -  `powerlaw`: the powerlaw formula for bond length dependence for sk integrals.
                                    -  `varTang96`: a variational formula based on Tang96 formula.
                                    -  `NRL0`: the old version of NRL formula for overlap, we set overlap and hopping share same options.
                                    -  `NRL1`: the new version of NRL formula for overlap. 
                    

                .. |code:model_options/nnsk/hopping[powerlaw]| replace:: ``powerlaw``
                .. _`code:model_options/nnsk/hopping[powerlaw]`: `model_options/nnsk/hopping[powerlaw]`_
                .. |code:model_options/nnsk/hopping[varTang96]| replace:: ``varTang96``
                .. _`code:model_options/nnsk/hopping[varTang96]`: `model_options/nnsk/hopping[varTang96]`_
                .. |code:model_options/nnsk/hopping[NRL0]| replace:: ``NRL0``
                .. _`code:model_options/nnsk/hopping[NRL0]`: `model_options/nnsk/hopping[NRL0]`_
                .. |code:model_options/nnsk/hopping[NRL1]| replace:: ``NRL1``
                .. _`code:model_options/nnsk/hopping[NRL1]`: `model_options/nnsk/hopping[NRL1]`_
                .. |code:model_options/nnsk/hopping[custom]| replace:: ``custom``
                .. _`code:model_options/nnsk/hopping[custom]`: `model_options/nnsk/hopping[custom]`_

            .. |flag:model_options/nnsk/hopping/method| replace:: *method*
            .. _`flag:model_options/nnsk/hopping/method`: `model_options/nnsk/hopping/method`_


            .. _`model_options/nnsk/hopping[powerlaw]`: 

            When |flag:model_options/nnsk/hopping/method|_ is set to ``powerlaw``: 

            .. _`model_options/nnsk/hopping[powerlaw]/rs`: 

            rs: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/hopping[powerlaw]/rs``

                The cut-off for smooth function fc for powerlaw and varTang96, fc(rs)=0.5

            .. _`model_options/nnsk/hopping[powerlaw]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/hopping[powerlaw]/w``

                 The decay w in fc


            .. _`model_options/nnsk/hopping[varTang96]`: 

            When |flag:model_options/nnsk/hopping/method|_ is set to ``varTang96``: 

            .. _`model_options/nnsk/hopping[varTang96]/rs`: 

            rs: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/hopping[varTang96]/rs``

                The cut-off for smooth function fc for powerlaw and varTang96, fc(rs)=0.5

            .. _`model_options/nnsk/hopping[varTang96]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/hopping[varTang96]/w``

                 The decay w in fc


            .. _`model_options/nnsk/hopping[NRL0]`: 

            When |flag:model_options/nnsk/hopping/method|_ is set to ``NRL0``: 

            .. _`model_options/nnsk/hopping[NRL0]/rc`: 

            rc: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/hopping[NRL0]/rc``

                The cut-off for smooth function fc for NRL, fc(rc) = 0.

            .. _`model_options/nnsk/hopping[NRL0]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/hopping[NRL0]/w``

                 The decay w in fc


            .. _`model_options/nnsk/hopping[NRL1]`: 

            When |flag:model_options/nnsk/hopping/method|_ is set to ``NRL1``: 

            .. _`model_options/nnsk/hopping[NRL1]/rc`: 

            rc: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/hopping[NRL1]/rc``

                The cut-off for smooth function fc for NRL, fc(rc) = 0.

            .. _`model_options/nnsk/hopping[NRL1]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/hopping[NRL1]/w``

                 The decay w in fc


            .. _`model_options/nnsk/hopping[custom]`: 

            When |flag:model_options/nnsk/hopping/method|_ is set to ``custom``: 


        .. _`model_options/nnsk/soc`: 

        soc: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``model_options/nnsk/soc``

            The soc options to define the soc of nnsk model,
                            Default: {} # empty dict

                            - {'method':'none'} : use database soc value. 
                            - {'method':uniform} : set lambda_il; assign a soc lambda value for each orbital -l on each atomtype i; l=0,1,2 for s p d.

        .. _`model_options/nnsk/freeze`: 

        freeze: 
            | type: ``str`` | ``bool`` | ``list``, optional, default: ``False``
            | argument path: ``model_options/nnsk/freeze``

            The parameters to define the freeze of nnsk model can be bool and string and list.

                                Default: False

                                 - True: freeze all the nnsk parameters

                                 - False: train all the nnsk parameters
 
                                 - 'hopping','onsite','overlap' and 'soc' to freeze the corresponding parameters.
                                 - list of the strings e.g. ['overlap','soc'] to freeze both overlap and soc parameters.

        .. _`model_options/nnsk/std`: 

        std: 
            | type: ``float``, optional, default: ``0.01``
            | argument path: ``model_options/nnsk/std``

            The std value to initialize the nnsk parameters. Default: 0.01

        .. _`model_options/nnsk/push`: 

        push: 
            | type: ``dict`` | ``bool``, optional, default: ``False``
            | argument path: ``model_options/nnsk/push``

            The parameters to define the push the soft cutoff of nnsk model.

            .. _`model_options/nnsk/push/rs_thr`: 

            rs_thr: 
                | type: ``float`` | ``int``, optional, default: ``0.0``
                | argument path: ``model_options/nnsk/push/rs_thr``

                The step size for cutoff value for smooth function in the nnsk anlytical formula.

            .. _`model_options/nnsk/push/rc_thr`: 

            rc_thr: 
                | type: ``float`` | ``int``, optional, default: ``0.0``
                | argument path: ``model_options/nnsk/push/rc_thr``

                The step size for cutoff value for smooth function in the nnsk anlytical formula.

            .. _`model_options/nnsk/push/w_thr`: 

            w_thr: 
                | type: ``float`` | ``int``, optional, default: ``0.0``
                | argument path: ``model_options/nnsk/push/w_thr``

                The step size for decay factor w.

            .. _`model_options/nnsk/push/period`: 

            period: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``model_options/nnsk/push/period``

                the interval of iterations to modify the rs w values.

