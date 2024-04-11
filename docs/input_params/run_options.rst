========================================
Run Options
========================================
.. _`run_op`: 

run_op: 
    | type: ``dict``
    | argument path: ``run_op``

    .. _`run_op/task_options`: 

    task_options: 
        | type: ``dict``, optional
        | argument path: ``run_op/task_options``

        the task to run, includes: band, dos, pdos, FS2D, FS3D, ifermi


        Depending on the value of *task*, different sub args are accepted. 

        .. _`run_op/task_options/task`: 

        task:
            | type: ``str`` (flag key)
            | argument path: ``run_op/task_options/task`` 
            | possible choices: |code:run_op/task_options[band]|_, |code:run_op/task_options[dos]|_, |code:run_op/task_options[pdos]|_, |code:run_op/task_options[FS2D]|_, |code:run_op/task_options[FS3D]|_, |code:run_op/task_options[write_sk]|_, |code:run_op/task_options[ifermi]|_, |code:run_op/task_options[negf]|_, |code:run_op/task_options[tbtrans_negf]|_

            The string define the task DeePTB conduct, includes: 
                                - `band`: for band structure plotting. 
                                - `dos`: for density of states plotting.
                                - `pdos`: for projected density of states plotting.
                                - `FS2D`: for 2D fermi-surface plotting.
                                - `FS3D`: for 3D fermi-surface plotting.
                                - `write_sk`: for transcript the nnsk model to standard sk parameter table
                                - `ifermi`: for fermi surface plotting.
                                - `negf`: for non-equilibrium green function calculation.
                                - `tbtrans_negf`: for non-equilibrium green function calculation with tbtrans.
                

            .. |code:run_op/task_options[band]| replace:: ``band``
            .. _`code:run_op/task_options[band]`: `run_op/task_options[band]`_
            .. |code:run_op/task_options[dos]| replace:: ``dos``
            .. _`code:run_op/task_options[dos]`: `run_op/task_options[dos]`_
            .. |code:run_op/task_options[pdos]| replace:: ``pdos``
            .. _`code:run_op/task_options[pdos]`: `run_op/task_options[pdos]`_
            .. |code:run_op/task_options[FS2D]| replace:: ``FS2D``
            .. _`code:run_op/task_options[FS2D]`: `run_op/task_options[FS2D]`_
            .. |code:run_op/task_options[FS3D]| replace:: ``FS3D``
            .. _`code:run_op/task_options[FS3D]`: `run_op/task_options[FS3D]`_
            .. |code:run_op/task_options[write_sk]| replace:: ``write_sk``
            .. _`code:run_op/task_options[write_sk]`: `run_op/task_options[write_sk]`_
            .. |code:run_op/task_options[ifermi]| replace:: ``ifermi``
            .. _`code:run_op/task_options[ifermi]`: `run_op/task_options[ifermi]`_
            .. |code:run_op/task_options[negf]| replace:: ``negf``
            .. _`code:run_op/task_options[negf]`: `run_op/task_options[negf]`_
            .. |code:run_op/task_options[tbtrans_negf]| replace:: ``tbtrans_negf``
            .. _`code:run_op/task_options[tbtrans_negf]`: `run_op/task_options[tbtrans_negf]`_

        .. |flag:run_op/task_options/task| replace:: *task*
        .. _`flag:run_op/task_options/task`: `run_op/task_options/task`_


        .. _`run_op/task_options[band]`: 

        When |flag:run_op/task_options/task|_ is set to ``band``: 

        .. _`run_op/task_options[band]/kline_type`: 

        kline_type: 
            | type: ``str``
            | argument path: ``run_op/task_options[band]/kline_type``

            The different type to build kpath line mode.
                                - "abacus" : the abacus format 
                                - "vasp" : the vasp format
                                - "ase" : the ase format
                    

        .. _`run_op/task_options[band]/kpath`: 

        kpath: 
            | type: ``str`` | ``list``
            | argument path: ``run_op/task_options[band]/kpath``

            for abacus, this is list, for vasp it is a string to specifc the kpath.

        .. _`run_op/task_options[band]/klabels`: 

        klabels: 
            | type: ``list``, optional, default: ``['']``
            | argument path: ``run_op/task_options[band]/klabels``

            the labels for high symmetry kpoint

        .. _`run_op/task_options[band]/E_fermi`: 

        E_fermi: 
            | type: ``float`` | ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/E_fermi``

            the fermi level used to plot band

        .. _`run_op/task_options[band]/emin`: 

        emin: 
            | type: ``float`` | ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/emin``

            the min energy to show the band plot

        .. _`run_op/task_options[band]/emax`: 

        emax: 
            | type: ``float`` | ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/emax``

            the max energy to show the band plot

        .. _`run_op/task_options[band]/nkpoints`: 

        nkpoints: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``run_op/task_options[band]/nkpoints``

            the max energy to show the band plot

        .. _`run_op/task_options[band]/ref_band`: 

        ref_band: 
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/ref_band``

            the reference band structure to be ploted together with dptb bands.

        .. _`run_op/task_options[band]/nel_atom`: 

        nel_atom: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[band]/nel_atom``

            the valence electron number of each type of atom.


        .. _`run_op/task_options[dos]`: 

        When |flag:run_op/task_options/task|_ is set to ``dos``: 

        .. _`run_op/task_options[dos]/mesh_grid`: 

        mesh_grid: 
            | type: ``list``
            | argument path: ``run_op/task_options[dos]/mesh_grid``

        .. _`run_op/task_options[dos]/sigma`: 

        sigma: 
            | type: ``float``
            | argument path: ``run_op/task_options[dos]/sigma``

        .. _`run_op/task_options[dos]/npoints`: 

        npoints: 
            | type: ``int``
            | argument path: ``run_op/task_options[dos]/npoints``

        .. _`run_op/task_options[dos]/width`: 

        width: 
            | type: ``list``
            | argument path: ``run_op/task_options[dos]/width``

        .. _`run_op/task_options[dos]/E_fermi`: 

        E_fermi: 
            | type: ``float`` | ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[dos]/E_fermi``

        .. _`run_op/task_options[dos]/gamma_center`: 

        gamma_center: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[dos]/gamma_center``


        .. _`run_op/task_options[pdos]`: 

        When |flag:run_op/task_options/task|_ is set to ``pdos``: 

        .. _`run_op/task_options[pdos]/mesh_grid`: 

        mesh_grid: 
            | type: ``list``
            | argument path: ``run_op/task_options[pdos]/mesh_grid``

        .. _`run_op/task_options[pdos]/sigma`: 

        sigma: 
            | type: ``float``
            | argument path: ``run_op/task_options[pdos]/sigma``

        .. _`run_op/task_options[pdos]/npoints`: 

        npoints: 
            | type: ``int``
            | argument path: ``run_op/task_options[pdos]/npoints``

        .. _`run_op/task_options[pdos]/width`: 

        width: 
            | type: ``list``
            | argument path: ``run_op/task_options[pdos]/width``

        .. _`run_op/task_options[pdos]/E_fermi`: 

        E_fermi: 
            | type: ``float`` | ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/task_options[pdos]/E_fermi``

        .. _`run_op/task_options[pdos]/atom_index`: 

        atom_index: 
            | type: ``list``
            | argument path: ``run_op/task_options[pdos]/atom_index``

        .. _`run_op/task_options[pdos]/orbital_index`: 

        orbital_index: 
            | type: ``list``
            | argument path: ``run_op/task_options[pdos]/orbital_index``

        .. _`run_op/task_options[pdos]/gamma_center`: 

        gamma_center: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[pdos]/gamma_center``


        .. _`run_op/task_options[FS2D]`: 

        When |flag:run_op/task_options/task|_ is set to ``FS2D``: 

        .. _`run_op/task_options[FS2D]/mesh_grid`: 

        mesh_grid: 
            | type: ``list``
            | argument path: ``run_op/task_options[FS2D]/mesh_grid``

        .. _`run_op/task_options[FS2D]/sigma`: 

        sigma: 
            | type: ``float``
            | argument path: ``run_op/task_options[FS2D]/sigma``

        .. _`run_op/task_options[FS2D]/E0`: 

        E0: 
            | type: ``int``
            | argument path: ``run_op/task_options[FS2D]/E0``

        .. _`run_op/task_options[FS2D]/intpfactor`: 

        intpfactor: 
            | type: ``int``
            | argument path: ``run_op/task_options[FS2D]/intpfactor``


        .. _`run_op/task_options[FS3D]`: 

        When |flag:run_op/task_options/task|_ is set to ``FS3D``: 

        .. _`run_op/task_options[FS3D]/mesh_grid`: 

        mesh_grid: 
            | type: ``list``
            | argument path: ``run_op/task_options[FS3D]/mesh_grid``

        .. _`run_op/task_options[FS3D]/sigma`: 

        sigma: 
            | type: ``float``
            | argument path: ``run_op/task_options[FS3D]/sigma``

        .. _`run_op/task_options[FS3D]/E0`: 

        E0: 
            | type: ``int``
            | argument path: ``run_op/task_options[FS3D]/E0``

        .. _`run_op/task_options[FS3D]/intpfactor`: 

        intpfactor: 
            | type: ``int``
            | argument path: ``run_op/task_options[FS3D]/intpfactor``


        .. _`run_op/task_options[write_sk]`: 

        When |flag:run_op/task_options/task|_ is set to ``write_sk``: 

        .. _`run_op/task_options[write_sk]/format`: 

        format: 
            | type: ``str``, optional, default: ``sktable``
            | argument path: ``run_op/task_options[write_sk]/format``

        .. _`run_op/task_options[write_sk]/thr`: 

        thr: 
            | type: ``float``, optional, default: ``0.001``
            | argument path: ``run_op/task_options[write_sk]/thr``


        .. _`run_op/task_options[ifermi]`: 

        When |flag:run_op/task_options/task|_ is set to ``ifermi``: 

        .. _`run_op/task_options[ifermi]/fermisurface`: 

        fermisurface: 
            | type: ``dict``
            | argument path: ``run_op/task_options[ifermi]/fermisurface``

            .. _`run_op/task_options[ifermi]/fermisurface/mesh_grid`: 

            mesh_grid: 
                | type: ``list``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/mesh_grid``

            .. _`run_op/task_options[ifermi]/fermisurface/mu`: 

            mu: 
                | type: ``float`` | ``int``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/mu``

            .. _`run_op/task_options[ifermi]/fermisurface/sigma`: 

            sigma: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/sigma``

            .. _`run_op/task_options[ifermi]/fermisurface/intpfactor`: 

            intpfactor: 
                | type: ``int``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/intpfactor``

            .. _`run_op/task_options[ifermi]/fermisurface/wigner_seitz`: 

            wigner_seitz: 
                | type: ``bool``, optional, default: ``True``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/wigner_seitz``

            .. _`run_op/task_options[ifermi]/fermisurface/nworkers`: 

            nworkers: 
                | type: ``int``, optional, default: ``-1``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/nworkers``

            .. _`run_op/task_options[ifermi]/fermisurface/plot_type`: 

            plot_type: 
                | type: ``str``, optional, default: ``plotly``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_type``

                plot_type: Method used for plotting. Valid options are: matplotlib, plotly, mayavi, crystal_toolkit.

            .. _`run_op/task_options[ifermi]/fermisurface/use_gui`: 

            use_gui: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/use_gui``

            .. _`run_op/task_options[ifermi]/fermisurface/plot_fs_bands`: 

            plot_fs_bands: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_fs_bands``

            .. _`run_op/task_options[ifermi]/fermisurface/fs_plane`: 

            fs_plane: 
                | type: ``list``, optional, default: ``[0, 0, 1]``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/fs_plane``

            .. _`run_op/task_options[ifermi]/fermisurface/fs_distance`: 

            fs_distance: 
                | type: ``float`` | ``int``, optional, default: ``0``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/fs_distance``

            .. _`run_op/task_options[ifermi]/fermisurface/plot_options`: 

            plot_options: 
                | type: ``dict``, optional, default: ``{}``
                | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_options``

                .. _`run_op/task_options[ifermi]/fermisurface/plot_options/colors`: 

                colors: 
                    | type: ``str`` | ``dict`` | ``NoneType`` | ``list``, optional, default: ``None``
                    | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_options/colors``

                    The color specification for the iso-surfaces. Valid options are:
                                    - A single color to use for all Fermi surfaces, specified as a tuple of
                                      rgb values from 0 to 1. E.g., red would be ``(1, 0, 0)``.
                                    - A list of colors, specified as above.
                                    - A dictionary of ``{Spin.up: color1, Spin.down: color2}``, where the
                                      colors are specified as above.
                                    - A string specifying which matplotlib colormap to use. See
                                      https://matplotlib.org/tutorials/colors/colormaps.html for more
                                      information.
                                    - ``None``, in which case the default colors will be used.
                

                .. _`run_op/task_options[ifermi]/fermisurface/plot_options/projection_axis`: 

                projection_axis: 
                    | type: ``list`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_options/projection_axis``

                    projection_axis: Projection axis that can be used to calculate the color of
                                    vector properties. If None, the norm of the properties will be used,
                                    otherwise the color will be determined by the dot product of the
                                    properties with the projection axis. Only has an effect when used with
                                    the ``vector_properties`` option.

                .. _`run_op/task_options[ifermi]/fermisurface/plot_options/hide_surface`: 

                hide_surface: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_options/hide_surface``

                    hide_surface: Whether to hide the Fermi surface. Only recommended in combination with the ``vector_properties`` option.

                .. _`run_op/task_options[ifermi]/fermisurface/plot_options/hide_labels`: 

                hide_labels: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_options/hide_labels``

                    hide_labels: Whether to show the high-symmetry k-point labels.

                .. _`run_op/task_options[ifermi]/fermisurface/plot_options/hide_cell`: 

                hide_cell: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_options/hide_cell``

                    hide_cell: Whether to show the reciprocal cell boundary.

                .. _`run_op/task_options[ifermi]/fermisurface/plot_options/vector_spacing`: 

                vector_spacing: 
                    | type: ``float``, optional, default: ``0.2``
                    | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_options/vector_spacing``

                    vector_spacing: The rough spacing between arrows. Uses a custom algorithm
                                    for resampling the Fermi surface to ensure that arrows are not too close
                                    together. Only has an effect when used with the ``vector_properties``
                                    option.

                .. _`run_op/task_options[ifermi]/fermisurface/plot_options/azimuth`: 

                azimuth: 
                    | type: ``float``, optional, default: ``45.0``
                    | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_options/azimuth``

                    azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended by the position vector on a sphere projected on to the x-y plane.

                .. _`run_op/task_options[ifermi]/fermisurface/plot_options/elevation`: 

                elevation: 
                    | type: ``float``, optional, default: ``35.0``
                    | argument path: ``run_op/task_options[ifermi]/fermisurface/plot_options/elevation``

                    The zenith angle of the viewpoint in degrees, i.e. the angle subtended by the position vector and the z-axis.

        .. _`run_op/task_options[ifermi]/property`: 

        property: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[ifermi]/property``

            .. _`run_op/task_options[ifermi]/property/velocity`: 

            velocity: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[ifermi]/property/velocity``

            .. _`run_op/task_options[ifermi]/property/color_properties`: 

            color_properties: 
                | type: ``str`` | ``bool``, optional, default: ``False``
                | argument path: ``run_op/task_options[ifermi]/property/color_properties``

                color_properties: Whether to use the properties to color the Fermi surface.
                                If the properties is a vector then the norm of the properties will be
                                used. Note, this will only take effect if the Fermi surface has
                                properties. If set to True, the viridis colormap will be used.
                                Alternative colormaps can be selected by setting ``color_properties``
                                to a matplotlib colormap name. This setting will override the ``colors``
                                option. For vector properties, the arrows are colored according to the
                                norm of the properties by default. If used in combination with the
                                ``projection_axis`` option, the color will be determined by the dot
                                product of the properties with the projection axis.

            .. _`run_op/task_options[ifermi]/property/colormap`: 

            colormap: 
                | type: ``str``, optional, default: ``viridis``
                | argument path: ``run_op/task_options[ifermi]/property/colormap``

            .. _`run_op/task_options[ifermi]/property/prop_plane`: 

            prop_plane: 
                | type: ``list``, optional, default: ``[0, 0, 1]``
                | argument path: ``run_op/task_options[ifermi]/property/prop_plane``

            .. _`run_op/task_options[ifermi]/property/prop_distance`: 

            prop_distance: 
                | type: ``float`` | ``int``, optional, default: ``0``
                | argument path: ``run_op/task_options[ifermi]/property/prop_distance``

            .. _`run_op/task_options[ifermi]/property/plot_options`: 

            plot_options: 
                | type: ``dict``, optional, default: ``{}``
                | argument path: ``run_op/task_options[ifermi]/property/plot_options``

                .. _`run_op/task_options[ifermi]/property/plot_options/colors`: 

                colors: 
                    | type: ``str`` | ``dict`` | ``NoneType`` | ``list``, optional, default: ``None``
                    | argument path: ``run_op/task_options[ifermi]/property/plot_options/colors``

                    The color specification for the iso-surfaces. Valid options are:
                                    - A single color to use for all Fermi surfaces, specified as a tuple of
                                      rgb values from 0 to 1. E.g., red would be ``(1, 0, 0)``.
                                    - A list of colors, specified as above.
                                    - A dictionary of ``{Spin.up: color1, Spin.down: color2}``, where the
                                      colors are specified as above.
                                    - A string specifying which matplotlib colormap to use. See
                                      https://matplotlib.org/tutorials/colors/colormaps.html for more
                                      information.
                                    - ``None``, in which case the default colors will be used.
                

                .. _`run_op/task_options[ifermi]/property/plot_options/projection_axis`: 

                projection_axis: 
                    | type: ``list`` | ``NoneType``, optional, default: ``None``
                    | argument path: ``run_op/task_options[ifermi]/property/plot_options/projection_axis``

                    projection_axis: Projection axis that can be used to calculate the color of
                                    vector properties. If None, the norm of the properties will be used,
                                    otherwise the color will be determined by the dot product of the
                                    properties with the projection axis. Only has an effect when used with
                                    the ``vector_properties`` option.

                .. _`run_op/task_options[ifermi]/property/plot_options/hide_surface`: 

                hide_surface: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[ifermi]/property/plot_options/hide_surface``

                    hide_surface: Whether to hide the Fermi surface. Only recommended in combination with the ``vector_properties`` option.

                .. _`run_op/task_options[ifermi]/property/plot_options/hide_labels`: 

                hide_labels: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[ifermi]/property/plot_options/hide_labels``

                    hide_labels: Whether to show the high-symmetry k-point labels.

                .. _`run_op/task_options[ifermi]/property/plot_options/hide_cell`: 

                hide_cell: 
                    | type: ``bool``, optional, default: ``False``
                    | argument path: ``run_op/task_options[ifermi]/property/plot_options/hide_cell``

                    hide_cell: Whether to show the reciprocal cell boundary.

                .. _`run_op/task_options[ifermi]/property/plot_options/vector_spacing`: 

                vector_spacing: 
                    | type: ``float``, optional, default: ``0.2``
                    | argument path: ``run_op/task_options[ifermi]/property/plot_options/vector_spacing``

                    vector_spacing: The rough spacing between arrows. Uses a custom algorithm
                                    for resampling the Fermi surface to ensure that arrows are not too close
                                    together. Only has an effect when used with the ``vector_properties``
                                    option.

                .. _`run_op/task_options[ifermi]/property/plot_options/azimuth`: 

                azimuth: 
                    | type: ``float``, optional, default: ``45.0``
                    | argument path: ``run_op/task_options[ifermi]/property/plot_options/azimuth``

                    azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended by the position vector on a sphere projected on to the x-y plane.

                .. _`run_op/task_options[ifermi]/property/plot_options/elevation`: 

                elevation: 
                    | type: ``float``, optional, default: ``35.0``
                    | argument path: ``run_op/task_options[ifermi]/property/plot_options/elevation``

                    The zenith angle of the viewpoint in degrees, i.e. the angle subtended by the position vector and the z-axis.


        .. _`run_op/task_options[negf]`: 

        When |flag:run_op/task_options/task|_ is set to ``negf``: 

        .. _`run_op/task_options[negf]/scf`: 

        scf: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/scf``

        .. _`run_op/task_options[negf]/block_tridiagonal`: 

        block_tridiagonal: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/block_tridiagonal``

        .. _`run_op/task_options[negf]/ele_T`: 

        ele_T: 
            | type: ``float`` | ``int``
            | argument path: ``run_op/task_options[negf]/ele_T``

        .. _`run_op/task_options[negf]/unit`: 

        unit: 
            | type: ``str``, optional, default: ``Hartree``
            | argument path: ``run_op/task_options[negf]/unit``

        .. _`run_op/task_options[negf]/scf_options`: 

        scf_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[negf]/scf_options``


            Depending on the value of *mode*, different sub args are accepted. 

            .. _`run_op/task_options[negf]/scf_options/mode`: 

            mode:
                | type: ``str`` (flag key), default: ``PDIIS``
                | argument path: ``run_op/task_options[negf]/scf_options/mode`` 
                | possible choices: |code:run_op/task_options[negf]/scf_options[PDIIS]|_

                .. |code:run_op/task_options[negf]/scf_options[PDIIS]| replace:: ``PDIIS``
                .. _`code:run_op/task_options[negf]/scf_options[PDIIS]`: `run_op/task_options[negf]/scf_options[PDIIS]`_

            .. |flag:run_op/task_options[negf]/scf_options/mode| replace:: *mode*
            .. _`flag:run_op/task_options[negf]/scf_options/mode`: `run_op/task_options[negf]/scf_options/mode`_


            .. _`run_op/task_options[negf]/scf_options[PDIIS]`: 

            When |flag:run_op/task_options[negf]/scf_options/mode|_ is set to ``PDIIS``: 

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/mixing_period`: 

            mixing_period: 
                | type: ``int``, optional, default: ``3``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/mixing_period``

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/step_size`: 

            step_size: 
                | type: ``float`` | ``int``, optional, default: ``0.05``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/step_size``

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/n_history`: 

            n_history: 
                | type: ``int``, optional, default: ``6``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/n_history``

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/abs_err`: 

            abs_err: 
                | type: ``float`` | ``int``, optional, default: ``1e-06``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/abs_err``

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/rel_err`: 

            rel_err: 
                | type: ``float`` | ``int``, optional, default: ``0.0001``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/rel_err``

            .. _`run_op/task_options[negf]/scf_options[PDIIS]/max_iter`: 

            max_iter: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``run_op/task_options[negf]/scf_options[PDIIS]/max_iter``

        .. _`run_op/task_options[negf]/stru_options`: 

        stru_options: 
            | type: ``dict``
            | argument path: ``run_op/task_options[negf]/stru_options``

            .. _`run_op/task_options[negf]/stru_options/device`: 

            device: 
                | type: ``dict``
                | argument path: ``run_op/task_options[negf]/stru_options/device``

                .. _`run_op/task_options[negf]/stru_options/device/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/stru_options/device/id``

                .. _`run_op/task_options[negf]/stru_options/device/sort`: 

                sort: 
                    | type: ``bool``, optional, default: ``True``
                    | argument path: ``run_op/task_options[negf]/stru_options/device/sort``

            .. _`run_op/task_options[negf]/stru_options/lead_L`: 

            lead_L: 
                | type: ``dict``
                | argument path: ``run_op/task_options[negf]/stru_options/lead_L``

                .. _`run_op/task_options[negf]/stru_options/lead_L/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_L/id``

                .. _`run_op/task_options[negf]/stru_options/lead_L/voltage`: 

                voltage: 
                    | type: ``float`` | ``int``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_L/voltage``

            .. _`run_op/task_options[negf]/stru_options/lead_R`: 

            lead_R: 
                | type: ``dict``
                | argument path: ``run_op/task_options[negf]/stru_options/lead_R``

                .. _`run_op/task_options[negf]/stru_options/lead_R/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_R/id``

                .. _`run_op/task_options[negf]/stru_options/lead_R/voltage`: 

                voltage: 
                    | type: ``float`` | ``int``
                    | argument path: ``run_op/task_options[negf]/stru_options/lead_R/voltage``

            .. _`run_op/task_options[negf]/stru_options/kmesh`: 

            kmesh: 
                | type: ``list``, optional, default: ``[1, 1, 1]``
                | argument path: ``run_op/task_options[negf]/stru_options/kmesh``

            .. _`run_op/task_options[negf]/stru_options/pbc`: 

            pbc: 
                | type: ``list``, optional, default: ``[False, False, False]``
                | argument path: ``run_op/task_options[negf]/stru_options/pbc``

            .. _`run_op/task_options[negf]/stru_options/gamma_center`: 

            gamma_center: 
                | type: ``list`` | ``bool``, optional, default: ``True``
                | argument path: ``run_op/task_options[negf]/stru_options/gamma_center``

            .. _`run_op/task_options[negf]/stru_options/time_reversal_symmetry`: 

            time_reversal_symmetry: 
                | type: ``list`` | ``bool``, optional, default: ``True``
                | argument path: ``run_op/task_options[negf]/stru_options/time_reversal_symmetry``

        .. _`run_op/task_options[negf]/poisson_options`: 

        poisson_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[negf]/poisson_options``


            Depending on the value of *solver*, different sub args are accepted. 

            .. _`run_op/task_options[negf]/poisson_options/solver`: 

            solver:
                | type: ``str`` (flag key), default: ``fmm``
                | argument path: ``run_op/task_options[negf]/poisson_options/solver`` 
                | possible choices: |code:run_op/task_options[negf]/poisson_options[fmm]|_

                .. |code:run_op/task_options[negf]/poisson_options[fmm]| replace:: ``fmm``
                .. _`code:run_op/task_options[negf]/poisson_options[fmm]`: `run_op/task_options[negf]/poisson_options[fmm]`_

            .. |flag:run_op/task_options[negf]/poisson_options/solver| replace:: *solver*
            .. _`flag:run_op/task_options[negf]/poisson_options/solver`: `run_op/task_options[negf]/poisson_options/solver`_


            .. _`run_op/task_options[negf]/poisson_options[fmm]`: 

            When |flag:run_op/task_options[negf]/poisson_options/solver|_ is set to ``fmm``: 

            .. _`run_op/task_options[negf]/poisson_options[fmm]/err`: 

            err: 
                | type: ``float`` | ``int``, optional, default: ``1e-05``
                | argument path: ``run_op/task_options[negf]/poisson_options[fmm]/err``

        .. _`run_op/task_options[negf]/sgf_solver`: 

        sgf_solver: 
            | type: ``str``, optional, default: ``Sancho-Rubio``
            | argument path: ``run_op/task_options[negf]/sgf_solver``

        .. _`run_op/task_options[negf]/espacing`: 

        espacing: 
            | type: ``float`` | ``int``
            | argument path: ``run_op/task_options[negf]/espacing``

        .. _`run_op/task_options[negf]/emin`: 

        emin: 
            | type: ``float`` | ``int``
            | argument path: ``run_op/task_options[negf]/emin``

        .. _`run_op/task_options[negf]/emax`: 

        emax: 
            | type: ``float`` | ``int``
            | argument path: ``run_op/task_options[negf]/emax``

        .. _`run_op/task_options[negf]/e_fermi`: 

        e_fermi: 
            | type: ``float`` | ``int``
            | argument path: ``run_op/task_options[negf]/e_fermi``

        .. _`run_op/task_options[negf]/density_options`: 

        density_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[negf]/density_options``


            Depending on the value of *method*, different sub args are accepted. 

            .. _`run_op/task_options[negf]/density_options/method`: 

            method:
                | type: ``str`` (flag key), default: ``Ozaki``
                | argument path: ``run_op/task_options[negf]/density_options/method`` 
                | possible choices: |code:run_op/task_options[negf]/density_options[Ozaki]|_

                .. |code:run_op/task_options[negf]/density_options[Ozaki]| replace:: ``Ozaki``
                .. _`code:run_op/task_options[negf]/density_options[Ozaki]`: `run_op/task_options[negf]/density_options[Ozaki]`_

            .. |flag:run_op/task_options[negf]/density_options/method| replace:: *method*
            .. _`flag:run_op/task_options[negf]/density_options/method`: `run_op/task_options[negf]/density_options/method`_


            .. _`run_op/task_options[negf]/density_options[Ozaki]`: 

            When |flag:run_op/task_options[negf]/density_options/method|_ is set to ``Ozaki``: 

            .. _`run_op/task_options[negf]/density_options[Ozaki]/R`: 

            R: 
                | type: ``float`` | ``int``, optional, default: ``1000000.0``
                | argument path: ``run_op/task_options[negf]/density_options[Ozaki]/R``

            .. _`run_op/task_options[negf]/density_options[Ozaki]/M_cut`: 

            M_cut: 
                | type: ``int``, optional, default: ``30``
                | argument path: ``run_op/task_options[negf]/density_options[Ozaki]/M_cut``

            .. _`run_op/task_options[negf]/density_options[Ozaki]/n_gauss`: 

            n_gauss: 
                | type: ``int``, optional, default: ``10``
                | argument path: ``run_op/task_options[negf]/density_options[Ozaki]/n_gauss``

        .. _`run_op/task_options[negf]/eta_lead`: 

        eta_lead: 
            | type: ``float`` | ``int``, optional, default: ``1e-05``
            | argument path: ``run_op/task_options[negf]/eta_lead``

        .. _`run_op/task_options[negf]/eta_device`: 

        eta_device: 
            | type: ``float`` | ``int``, optional, default: ``0.0``
            | argument path: ``run_op/task_options[negf]/eta_device``

        .. _`run_op/task_options[negf]/out_dos`: 

        out_dos: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/out_dos``

        .. _`run_op/task_options[negf]/out_tc`: 

        out_tc: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/out_tc``

        .. _`run_op/task_options[negf]/out_density`: 

        out_density: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/out_density``

        .. _`run_op/task_options[negf]/out_potential`: 

        out_potential: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/out_potential``

        .. _`run_op/task_options[negf]/out_current`: 

        out_current: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/out_current``

        .. _`run_op/task_options[negf]/out_current_nscf`: 

        out_current_nscf: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/out_current_nscf``

        .. _`run_op/task_options[negf]/out_ldos`: 

        out_ldos: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/out_ldos``

        .. _`run_op/task_options[negf]/out_lcurrent`: 

        out_lcurrent: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[negf]/out_lcurrent``


        .. _`run_op/task_options[tbtrans_negf]`: 

        When |flag:run_op/task_options/task|_ is set to ``tbtrans_negf``: 

        .. _`run_op/task_options[tbtrans_negf]/scf`: 

        scf: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/scf``

        .. _`run_op/task_options[tbtrans_negf]/block_tridiagonal`: 

        block_tridiagonal: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/block_tridiagonal``

        .. _`run_op/task_options[tbtrans_negf]/ele_T`: 

        ele_T: 
            | type: ``float`` | ``int``
            | argument path: ``run_op/task_options[tbtrans_negf]/ele_T``

        .. _`run_op/task_options[tbtrans_negf]/unit`: 

        unit: 
            | type: ``str``, optional, default: ``Hartree``
            | argument path: ``run_op/task_options[tbtrans_negf]/unit``

        .. _`run_op/task_options[tbtrans_negf]/scf_options`: 

        scf_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[tbtrans_negf]/scf_options``


            Depending on the value of *mode*, different sub args are accepted. 

            .. _`run_op/task_options[tbtrans_negf]/scf_options/mode`: 

            mode:
                | type: ``str`` (flag key), default: ``PDIIS``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options/mode`` 
                | possible choices: |code:run_op/task_options[tbtrans_negf]/scf_options[PDIIS]|_

                .. |code:run_op/task_options[tbtrans_negf]/scf_options[PDIIS]| replace:: ``PDIIS``
                .. _`code:run_op/task_options[tbtrans_negf]/scf_options[PDIIS]`: `run_op/task_options[tbtrans_negf]/scf_options[PDIIS]`_

            .. |flag:run_op/task_options[tbtrans_negf]/scf_options/mode| replace:: *mode*
            .. _`flag:run_op/task_options[tbtrans_negf]/scf_options/mode`: `run_op/task_options[tbtrans_negf]/scf_options/mode`_


            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]`: 

            When |flag:run_op/task_options[tbtrans_negf]/scf_options/mode|_ is set to ``PDIIS``: 

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/mixing_period`: 

            mixing_period: 
                | type: ``int``, optional, default: ``3``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/mixing_period``

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/step_size`: 

            step_size: 
                | type: ``float`` | ``int``, optional, default: ``0.05``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/step_size``

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/n_history`: 

            n_history: 
                | type: ``int``, optional, default: ``6``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/n_history``

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/abs_err`: 

            abs_err: 
                | type: ``float`` | ``int``, optional, default: ``1e-06``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/abs_err``

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/rel_err`: 

            rel_err: 
                | type: ``float`` | ``int``, optional, default: ``0.0001``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/rel_err``

            .. _`run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/max_iter`: 

            max_iter: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``run_op/task_options[tbtrans_negf]/scf_options[PDIIS]/max_iter``

        .. _`run_op/task_options[tbtrans_negf]/stru_options`: 

        stru_options: 
            | type: ``dict``
            | argument path: ``run_op/task_options[tbtrans_negf]/stru_options``

            .. _`run_op/task_options[tbtrans_negf]/stru_options/device`: 

            device: 
                | type: ``dict``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/device``

                .. _`run_op/task_options[tbtrans_negf]/stru_options/device/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/device/id``

                .. _`run_op/task_options[tbtrans_negf]/stru_options/device/sort`: 

                sort: 
                    | type: ``bool``, optional, default: ``True``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/device/sort``

            .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_L`: 

            lead_L: 
                | type: ``dict``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_L``

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_L/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_L/id``

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_L/voltage`: 

                voltage: 
                    | type: ``float`` | ``int``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_L/voltage``

            .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_R`: 

            lead_R: 
                | type: ``dict``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_R``

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_R/id`: 

                id: 
                    | type: ``str``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_R/id``

                .. _`run_op/task_options[tbtrans_negf]/stru_options/lead_R/voltage`: 

                voltage: 
                    | type: ``float`` | ``int``
                    | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/lead_R/voltage``

            .. _`run_op/task_options[tbtrans_negf]/stru_options/kmesh`: 

            kmesh: 
                | type: ``list``, optional, default: ``[1, 1, 1]``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/kmesh``

            .. _`run_op/task_options[tbtrans_negf]/stru_options/pbc`: 

            pbc: 
                | type: ``list``, optional, default: ``[False, False, False]``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/pbc``

            .. _`run_op/task_options[tbtrans_negf]/stru_options/gamma_center`: 

            gamma_center: 
                | type: ``list`` | ``bool``, optional, default: ``True``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/gamma_center``

            .. _`run_op/task_options[tbtrans_negf]/stru_options/time_reversal_symmetry`: 

            time_reversal_symmetry: 
                | type: ``list`` | ``bool``, optional, default: ``True``
                | argument path: ``run_op/task_options[tbtrans_negf]/stru_options/time_reversal_symmetry``

        .. _`run_op/task_options[tbtrans_negf]/poisson_options`: 

        poisson_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options``


            Depending on the value of *solver*, different sub args are accepted. 

            .. _`run_op/task_options[tbtrans_negf]/poisson_options/solver`: 

            solver:
                | type: ``str`` (flag key), default: ``fmm``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options/solver`` 
                | possible choices: |code:run_op/task_options[tbtrans_negf]/poisson_options[fmm]|_

                .. |code:run_op/task_options[tbtrans_negf]/poisson_options[fmm]| replace:: ``fmm``
                .. _`code:run_op/task_options[tbtrans_negf]/poisson_options[fmm]`: `run_op/task_options[tbtrans_negf]/poisson_options[fmm]`_

            .. |flag:run_op/task_options[tbtrans_negf]/poisson_options/solver| replace:: *solver*
            .. _`flag:run_op/task_options[tbtrans_negf]/poisson_options/solver`: `run_op/task_options[tbtrans_negf]/poisson_options/solver`_


            .. _`run_op/task_options[tbtrans_negf]/poisson_options[fmm]`: 

            When |flag:run_op/task_options[tbtrans_negf]/poisson_options/solver|_ is set to ``fmm``: 

            .. _`run_op/task_options[tbtrans_negf]/poisson_options[fmm]/err`: 

            err: 
                | type: ``float`` | ``int``, optional, default: ``1e-05``
                | argument path: ``run_op/task_options[tbtrans_negf]/poisson_options[fmm]/err``

        .. _`run_op/task_options[tbtrans_negf]/sgf_solver`: 

        sgf_solver: 
            | type: ``str``, optional, default: ``Sancho-Rubio``
            | argument path: ``run_op/task_options[tbtrans_negf]/sgf_solver``

        .. _`run_op/task_options[tbtrans_negf]/espacing`: 

        espacing: 
            | type: ``float`` | ``int``
            | argument path: ``run_op/task_options[tbtrans_negf]/espacing``

        .. _`run_op/task_options[tbtrans_negf]/emin`: 

        emin: 
            | type: ``float`` | ``int``
            | argument path: ``run_op/task_options[tbtrans_negf]/emin``

        .. _`run_op/task_options[tbtrans_negf]/emax`: 

        emax: 
            | type: ``float`` | ``int``
            | argument path: ``run_op/task_options[tbtrans_negf]/emax``

        .. _`run_op/task_options[tbtrans_negf]/e_fermi`: 

        e_fermi: 
            | type: ``float`` | ``int``
            | argument path: ``run_op/task_options[tbtrans_negf]/e_fermi``

        .. _`run_op/task_options[tbtrans_negf]/density_options`: 

        density_options: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``run_op/task_options[tbtrans_negf]/density_options``


            Depending on the value of *method*, different sub args are accepted. 

            .. _`run_op/task_options[tbtrans_negf]/density_options/method`: 

            method:
                | type: ``str`` (flag key), default: ``Ozaki``
                | argument path: ``run_op/task_options[tbtrans_negf]/density_options/method`` 
                | possible choices: |code:run_op/task_options[tbtrans_negf]/density_options[Ozaki]|_

                .. |code:run_op/task_options[tbtrans_negf]/density_options[Ozaki]| replace:: ``Ozaki``
                .. _`code:run_op/task_options[tbtrans_negf]/density_options[Ozaki]`: `run_op/task_options[tbtrans_negf]/density_options[Ozaki]`_

            .. |flag:run_op/task_options[tbtrans_negf]/density_options/method| replace:: *method*
            .. _`flag:run_op/task_options[tbtrans_negf]/density_options/method`: `run_op/task_options[tbtrans_negf]/density_options/method`_


            .. _`run_op/task_options[tbtrans_negf]/density_options[Ozaki]`: 

            When |flag:run_op/task_options[tbtrans_negf]/density_options/method|_ is set to ``Ozaki``: 

            .. _`run_op/task_options[tbtrans_negf]/density_options[Ozaki]/R`: 

            R: 
                | type: ``float`` | ``int``, optional, default: ``1000000.0``
                | argument path: ``run_op/task_options[tbtrans_negf]/density_options[Ozaki]/R``

            .. _`run_op/task_options[tbtrans_negf]/density_options[Ozaki]/M_cut`: 

            M_cut: 
                | type: ``int``, optional, default: ``30``
                | argument path: ``run_op/task_options[tbtrans_negf]/density_options[Ozaki]/M_cut``

            .. _`run_op/task_options[tbtrans_negf]/density_options[Ozaki]/n_gauss`: 

            n_gauss: 
                | type: ``int``, optional, default: ``10``
                | argument path: ``run_op/task_options[tbtrans_negf]/density_options[Ozaki]/n_gauss``

        .. _`run_op/task_options[tbtrans_negf]/eta_lead`: 

        eta_lead: 
            | type: ``float`` | ``int``, optional, default: ``1e-05``
            | argument path: ``run_op/task_options[tbtrans_negf]/eta_lead``

        .. _`run_op/task_options[tbtrans_negf]/eta_device`: 

        eta_device: 
            | type: ``float`` | ``int``, optional, default: ``0.0``
            | argument path: ``run_op/task_options[tbtrans_negf]/eta_device``

        .. _`run_op/task_options[tbtrans_negf]/out_dos`: 

        out_dos: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_dos``

        .. _`run_op/task_options[tbtrans_negf]/out_tc`: 

        out_tc: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_tc``

        .. _`run_op/task_options[tbtrans_negf]/out_density`: 

        out_density: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_density``

        .. _`run_op/task_options[tbtrans_negf]/out_potential`: 

        out_potential: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_potential``

        .. _`run_op/task_options[tbtrans_negf]/out_current`: 

        out_current: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_current``

        .. _`run_op/task_options[tbtrans_negf]/out_current_nscf`: 

        out_current_nscf: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_current_nscf``

        .. _`run_op/task_options[tbtrans_negf]/out_ldos`: 

        out_ldos: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_ldos``

        .. _`run_op/task_options[tbtrans_negf]/out_lcurrent`: 

        out_lcurrent: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``run_op/task_options[tbtrans_negf]/out_lcurrent``

    .. _`run_op/structure`: 

    structure: 
        | type: ``str`` | ``NoneType``, optional, default: ``None``
        | argument path: ``run_op/structure``

        the structure to run the task

    .. _`run_op/use_gui`: 

    use_gui: 
        | type: ``bool``, optional, default: ``False``
        | argument path: ``run_op/use_gui``

        To use the GUI or not

    .. _`run_op/AtomicData_options`: 

    AtomicData_options: 
        | type: ``dict``
        | argument path: ``run_op/AtomicData_options``

        .. _`run_op/AtomicData_options/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``run_op/AtomicData_options/r_max``

            the cutoff value for bond considering in TB model.

        .. _`run_op/AtomicData_options/er_max`: 

        er_max: 
            | type: ``dict`` | ``float`` | ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/AtomicData_options/er_max``

            The cutoff value for environment for each site for env correction model. should set for nnsk+env correction model.

        .. _`run_op/AtomicData_options/oer_max`: 

        oer_max: 
            | type: ``dict`` | ``float`` | ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``run_op/AtomicData_options/oer_max``

            The cutoff value for onsite environment for nnsk model, for now only need to set in strain and NRL mode.

        .. _`run_op/AtomicData_options/pbc`: 

        pbc: 
            | type: ``bool``
            | argument path: ``run_op/AtomicData_options/pbc``

            The periodic condition for the structure, can bool or list of bool to specific x,y,z direction.

