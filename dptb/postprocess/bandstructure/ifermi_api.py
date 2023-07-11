import numpy as np
import logging
import sys
import ase
from ase.io import read
from dptb.utils.tools import j_must_have
from dptb.utils.make_kpoints import  kmesh_fs

from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

try:
    import ifermi
    from ifermi.interpolate import FourierInterpolator
    from ifermi.kpoints import kpoints_from_bandstructure
    from ifermi.surface import FermiSurface
    from ifermi.plot import FermiSurfacePlotter
    from ifermi.plot import FermiSlicePlotter
    from ifermi.plot import show_plot, save_plot
    ifermi_installed = True

except ImportError:
    log.error('ifermi is not installed. Thus the ifermiaip is not available, Please install it first.')
    ifermi_installed = False

try:
    from pymatgen.electronic_structure.bandstructure import BandStructure
    from pymatgen.io.ase import AseAtomsAdaptor as ase2pmg
    from pymatgen.core import Lattice, Structure, Molecule
    from pymatgen.electronic_structure.core import Spin
    pymatgen_installed = True

except:
    log.error('pymatgen is not installed. Thus the ifermiaip is not available, Please install it first.')
    pymatgen_installed = False



class ifermiapi (object):
    def __init__ (self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk
        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
            self.structpmg = Structure.from_file(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
            self.structpmg = ase2pmg.get_structure(self.structase)
        else:
            raise ValueError('structure must be ase.Atoms or str')
    
        self._pass_paras(jdata)
        self.results_path = run_opt.get('results_path')
        self.apiH.update_struct(self.structase)

    
    def _pass_paras(self,jdata):
        self.ifermi_plot_options = jdata
        self.fs_plot_options = j_must_have(self.ifermi_plot_options , 'fermisurface')
        self.property_options = self.ifermi_plot_options.get('property',None)

        self.mesh_grid = self.fs_plot_options['mesh_grid']
        assert len(self.mesh_grid)==3, 'mesh_grid must have 3 elements'
        self.mu = self.fs_plot_options.get('mu',0)
        self.sigma = self.fs_plot_options.get('sigma',0.1)
        self.intpfactor = self.fs_plot_options.get('intpfactor',1)
        self.wigner_seitz = self.fs_plot_options.get('wigner_seitz',False)
        self.nworkers = self.fs_plot_options.get('nworkers',-1)
        self.plot_type = self.fs_plot_options.get('plot_type','matplotlib')
        assert self.plot_type in [ "matplotlib", "plotly", "mayavi", "crystal_toolkit"], \
                    'plot_type must be one of matplotlib, plotly, mayavi, crystal_toolkit'
        self.use_gui = self.fs_plot_options.get('use_gui',False)
        self.plot_fs_bands = self.fs_plot_options.get('plot_fs_bands',False)
        self.fs_plane = self.fs_plot_options.get('fs_plane',None)
        self.fs_distance = self.fs_plot_options.get('fs_distance',0)

        self.extra_plot_fs_options = self.fs_plot_options.get('plot_options',{})
        
        if self.property_options is not None:
            self.color_properties = self.property_options.get('color_properties',False) 
            self.plot_velocity = self.property_options.get('velocity',False)
            self.prop_colormap = self.property_options.get('colormap','viridis')
            self.plot_prop_options = self.property_options.get('plot_options',{})
            self.prop_plane = self.property_options.get('prop_plane',None)
            self.prop_distance = self.property_options.get('prop_distance',0)
        else:
            self.color_properties = False
            
    def get_band_structure(self):

        _, kpoints = kmesh_fs(meshgrid= self.mesh_grid)

        # eigenvalues : shape (nkpoints, nbands)
        eigenvalues, E_fermi = self.get_eigenvalues(kpoints= kpoints)
        assert len(eigenvalues.shape) == 2, 'eigenvalues must be 2D array'
        assert eigenvalues.shape[0] == kpoints.shape[0], 'eigenvalues.shape[0] must be equal to kpoints.shape[0]' 

        ist = np.sum(np.sum(eigenvalues - (E_fermi + self.mu - 10*self.sigma) < 0, axis=0) / eigenvalues.shape[0] >=1) - 1
        ied = np.sum(np.sum(eigenvalues - (E_fermi + self.mu + 10*self.sigma) > 0, axis=0) / eigenvalues.shape[0] < 1) + 1
        if ist <0:
            ist = 0
        

        eig_pick = eigenvalues[:,ist:ied]
        # Since DPTB code now is only support non-spin polarized calculation, and soc calculations, 
        # but in order to be compatible with ifermi, we need to store the eigenvalues as it is  spin up eigenvalues.
        eigenvals: DefaultDict[Spin, list] = defaultdict(list)

        # swap the axes to make the shape of eigenvals[Spin.up] is (nbands, nkpoints)
        eigenvals[Spin.up] = np.swapaxes(eig_pick,0,1)
        Latt_rev = Lattice(self.structpmg.lattice.reciprocal_lattice.matrix)

        self.bandstr = BandStructure(kpoints, eigenvals, Latt_rev, E_fermi, structure=self.structpmg)
        
        return self.bandstr
    

    def get_fs(self,bs):
        """
        get fermi surface from band structure
        """
        # get fermi surface
        interpolator = FourierInterpolator(bs) 
        if self.plot_velocity:
            dense_bs, velocities = interpolator.interpolate_bands(
                                            interpolation_factor=self.intpfactor, 
                                            return_velocities=True,
                                            nworkers=self.nworkers
                                            )
            dense_kpoints = kpoints_from_bandstructure(dense_bs)
            fs = FermiSurface.from_band_structure(
                band_structure=dense_bs,
                mu=self.mu,
                wigner_seitz=self.wigner_seitz,
                property_data=velocities,
                property_kpoints=dense_kpoints,
            )
        else:
            dense_bs = interpolator.interpolate_bands(
                                            interpolation_factor=self.intpfactor, 
                                            return_velocities=False, 
                                            nworkers=self.nworkers
                                            )

            fs = FermiSurface.from_band_structure(
                band_structure=dense_bs,
                mu=self.mu,
                wigner_seitz=self.wigner_seitz
            )

        log.info(f'Fermi surface is generated with totally {fs.n_surfaces} surfaces at {self.mu} eV.')
        for isoface in fs.isosurfaces[Spin.up]:
            log.info(f'The band index of isosurface is {isoface.band_idx}')
        
        return fs
    

    def fs_plot(self, fs): 
        plotter = FermiSurfacePlotter(fs)

        plot = plotter.get_plot(
            color_properties=self.color_properties, 
            plot_type=self.plot_type, 
            **self.extra_plot_fs_options
            )
        
        save_plot(plot, f'{self.results_path}/Ifermi_FS.png')
        if self.use_gui:
            show_plot(plot)

        if self.plot_fs_bands:
            for isoface in fs.isosurfaces[Spin.up]:
                idx = isoface.band_idx

                plot = plotter.get_plot(
                    color_properties=self.color_properties, 
                    plot_index={Spin.up: [idx]},
                    plot_type=self.plot_type,
                    **self.extra_plot_fs_options
                    )
                save_plot(plot, f'{self.results_path}/Ifermi_FS_band_{idx}.png')
                if self.use_gui:
                    show_plot(plot)

        if self.fs_plane:
            fermi_slice = fs.get_fermi_slice(
                plane_normal=self.fs_plane, 
                distance=self.fs_distance
                )
            slice_plotter = FermiSlicePlotter(fermi_slice)
            plot = slice_plotter.get_plot(color_properties=self.color_properties)
            save_plot(plot, f'{self.results_path}/Ifermi_FS_slice.png')
            if self.use_gui:
                show_plot(plot)
        
        if self.plot_velocity:
            plotter = FermiSurfacePlotter(fs)
            plot = plotter.get_plot(
                            vector_properties=self.prop_colormap,
                            plot_type=self.plot_type,
                            **self.plot_prop_options
                            )
            save_plot(plot, f'{self.results_path}/Ifermi_FS_velocity.png')
            if self.use_gui:
                show_plot(plot)
            
            if self.prop_plane:
                fermi_slice = fs.get_fermi_slice(
                    plane_normal=self.prop_plane, 
                    distance=self.prop_distance
                    )
                slice_plotter = FermiSlicePlotter(fermi_slice)
                plot = slice_plotter.get_plot(vector_properties=True)
                save_plot(plot, f'{self.results_path}/Ifermi_FS_velcoity_slice.png')
                if self.use_gui:
                    show_plot(plot)


    def get_eigenvalues(self, kpoints):
        all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
        self.eigenvalues, self.estimated_E_fermi = self.apiH.get_eigenvalues(kpoints)
        if self.fs_plot_options.get('E_fermi',None) != None:
            self.E_fermi = self.fs_plot_options['E_fermi']
            log.info(f'set E_fermi from jdata: {self.E_fermi} , While the estimated value is {self.estimated_E_fermi} .')
        else:
            self.E_fermi = self.estimated_E_fermi
            log.info(f'set E_fermi by estimated value {self.estimated_E_fermi} .')

        return self.eigenvalues, self.E_fermi

