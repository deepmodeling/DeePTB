import numpy as  np
from dptb.utils.constants import NumHvals
from dptb.sktb.skParam import sk_init, read_skfiles, interp_sk_gridvalues

class SKIntegrals(object):
    def __init__(self, proj_atom_anglr_m, sk_file_path, **kwargs):
        self.skfiles = sk_init(proj_atom_anglr_m, sk_file_path, **kwargs)


        grid_distance, num_grids, HSintgrl, \
                self.SiteE, self.HubdU, self.Occu = read_skfiles(self.skfiles)

        self.max_min_bond_length, self.interp_skfunc = \
                interp_sk_gridvalues(list(self.skfiles.keys()), grid_distance, num_grids, HSintgrl)

    def sk_integral(self, itype, jtype, dist):
        ''' get the bond integrals based on the distance of i,j atoms.

            Parameters
            ----------
            itype,jtype:
                the atom type of i, j atoms.
            dist:
                the distance between atoms i,j.

            Returns
            -------
                A list of bond integrals.
        '''
        sktype = itype + '-' + jtype
        max_length = self.max_min_bond_length[sktype][0]
        min_length = self.max_min_bond_length[sktype][1]

        assert dist > min_length, "Error, the distance between atoms i,j is too short."
        # add if distance_ij = 0, return the on_site value.
        if dist > max_length:
            res = np.zeros(2 * NumHvals)
        else:
            res = self.interp_skfunc[sktype](dist)
        return res
