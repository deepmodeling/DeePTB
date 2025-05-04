========================================
Set Info
========================================
.. _`setinfo`: 

setinfo: 
    | type: ``dict``
    | argument path: ``setinfo``

    .. _`setinfo/nframes`: 

    nframes: 
        | type: ``int``
        | argument path: ``setinfo/nframes``

        Number of frames in this trajectory.

    .. _`setinfo/natoms`: 

    natoms: 
        | type: ``int``, optional, default: ``-1``
        | argument path: ``setinfo/natoms``

        Number of atoms in each frame.

    .. _`setinfo/pos_type`: 

    pos_type: 
        | type: ``str``
        | argument path: ``setinfo/pos_type``

        Type of atomic position input. Can be frac / cart / ase.

    .. _`setinfo/pbc`: 

    pbc: 
        | type: ``list`` | ``bool``
        | argument path: ``setinfo/pbc``

        The periodic condition for the structure, can bool or list of bool to specific x,y,z direction.

    .. _`setinfo/bandinfo`: 

    bandinfo: 
        | type: ``dict``, optional
        | argument path: ``setinfo/bandinfo``

        .. _`setinfo/bandinfo/band_min`: 

        band_min: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``setinfo/bandinfo/band_min``

            the minum band index for the training band window with respected to the correctly selected DFT bands.
                               `important`: before setting this tag you should make sure you have already  exclude all the irrelevant in your training data.
                                            This logic for band_min and max is based on the simple fact the total number TB bands > the bands you care.   
                   

        .. _`setinfo/bandinfo/band_max`: 

        band_max: 
            | type: ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``setinfo/bandinfo/band_max``

            The maxmum band index for training band window

        .. _`setinfo/bandinfo/emin`: 

        emin: 
            | type: ``float`` | ``NoneType``, optional, default: ``None``
            | argument path: ``setinfo/bandinfo/emin``

            the minmum energy window, 0 meand the min value of the band at index band_min

        .. _`setinfo/bandinfo/emax`: 

        emax: 
            | type: ``float`` | ``NoneType``, optional, default: ``None``
            | argument path: ``setinfo/bandinfo/emax``

            the max energy window, emax value is respect to the min value of the band at index band_min

