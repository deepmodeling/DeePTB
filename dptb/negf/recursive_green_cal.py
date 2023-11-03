import torch.linalg as tLA
import torch

def recursive_gf_cal(energy, mat_l_list, mat_d_list, mat_u_list, sd, su, sl, s_in=0, s_out=0, eta=1e-5):
    """The recursive Green's function algorithm is taken from
    M. P. Anantram, M. S. Lundstrom and D. E. Nikonov, Proceedings of the IEEE, 96, 1511 - 1550 (2008)
    DOI: 10.1109/JPROC.2008.927355

    In order to get the electron correlation function output, the parameters s_in has to be set.
    For the hole correlation function, the parameter s_out has to be set.

    Parameters
    ----------
    energy : numpy.ndarray (dtype=numpy.float)
        Energy array
    mat_d_list : list of numpy.ndarray (dtype=numpy.float)
        List of diagonal blocks
    mat_u_list : list of numpy.ndarray (dtype=numpy.float)
        List of upper-diagonal blocks
    mat_l_list : list of numpy.ndarray (dtype=numpy.float)
        List of lower-diagonal blocks
    s_in :
         (Default value = 0)
    s_out :
         (Default value = 0)
    damp :
         (Default value = 0.000001j)

    Returns
    -------
    g_trans : numpy.ndarray (dtype=numpy.complex)
        Blocks of the retarded Green's function responsible for transmission
    grd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    grl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gru : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gr_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    gnd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    gnl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gnu : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gin_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    gpd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    gpl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gpu : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gip_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    """
    # -------------------------------------------------------------------
    # ---------- convert input arrays to the matrix data type -----------
    # ----------------- in case they are not matrices -------------------
    # -------------------------------------------------------------------
    for jj, item in enumerate(mat_d_list):
        mat_d_list[jj] = mat_d_list[jj] - (energy+1j*eta) * sd[jj]

    for jj, item in enumerate(mat_l_list):
        mat_l_list[jj] = mat_l_list[jj] - (energy+1j*eta) * sl[jj]
    for jj, item in enumerate(mat_u_list):
        mat_u_list[jj] = mat_u_list[jj] - (energy+1j*eta) * su[jj]
    # computes matrix sizes
    num_of_matrices = len(mat_d_list)  # Number of diagonal blocks.
    mat_shapes = [item.shape for item in mat_d_list]  # This gives the sizes of the diagonal matrices.
    # -------------------------------------------------------------------
    # -------------- compute retarded Green's function ------------------
    # -------------------------------------------------------------------
    # allocate empty lists of certain lengths
    gr_left = [None for _ in range(num_of_matrices)]
    gr_left[0] = tLA.solve(-mat_d_list[0], torch.eye(mat_shapes[0][0], dtype=mat_d_list[0].dtype))  # Initialising the retarded left connected.

    for q in range(num_of_matrices - 1):  # Recursive algorithm (B2)
        gr_left[q + 1] = tLA.solve(-mat_d_list[q + 1] - mat_l_list[q] @ gr_left[q] @ mat_u_list[q],
                                      torch.eye(mat_shapes[q + 1][0], dtype=mat_d_list[0].dtype))  # The left connected recursion.
    # -------------------------------------------------------------------

    grl = [None for _ in range(num_of_matrices-1)]
    gru = [None for _ in range(num_of_matrices-1)]
    grd = [i.clone() for i in gr_left]  # Our glorious benefactor.
    g_trans = gr_left[len(gr_left) - 1].clone()
    for q in range(num_of_matrices - 2, -1, -1):  # Recursive algorithm
        grl[q] = grd[q + 1] @ mat_l_list[q] @ gr_left[q]  # (B5) We get the off-diagonal blocks for free.
        gru[q] = gr_left[q] @ mat_u_list[q] @ grd[q + 1]  # (B6) because we need .Tthem.T for the next calc:
        grd[q] = gr_left[q] + gr_left[q] @ mat_u_list[q] @ grl[q]  # (B4) I suppose I could also use the lower.
        g_trans = gr_left[q] @ mat_u_list[q] @ g_trans

    # -------------------------------------------------------------------
    # ------ compute the electron correlation function if needed --------
    # -------------------------------------------------------------------

    if isinstance(s_in, list):

        gin_left = [None for _ in range(num_of_matrices)]
        gin_left[0] = gr_left[0] @ s_in[0] @ gr_left[0].conj().T

        for q in range(num_of_matrices - 1):
            sla2 = mat_l_list[q] @ gin_left[q] @ mat_u_list[q].conj().T
            prom = s_in[q + 1] + sla2
            gin_left[q + 1] = gr_left[q + 1] @ prom @ gr_left[q + 1].conj().T

        # ---------------------------------------------------------------

        gnl = [None for _ in range(num_of_matrices-1)]
        gnu = [None for _ in range(num_of_matrices-1)]
        gnd = [i.clone() for i in gin_left]

        for q in range(num_of_matrices - 2, -1, -1):  # Recursive algorithm
            gnl[q] = grd[q + 1] @ mat_l_list[q] @ gin_left[q] + \
                     gnd[q + 1] @ mat_l_list[q].conj().T @ gr_left[q].conj().T
            gnd[q] = gin_left[q] + \
                             gr_left[q] @ mat_u_list[q] @ gnd[q + 1] @ mat_l_list[q].conj().T @ \
                                 gr_left[q].conj().T + \
                             ((gin_left[q] @ mat_u_list[q].conj().T @ grl[q].conj().T) + (gru[q] @
                                 mat_l_list[q] @ gin_left[q]))

            gnu[q] = gnl[q].conj().T

    # -------------------------------------------------------------------
    # -------- compute the hole correlation function if needed ----------
    # -------------------------------------------------------------------
    if isinstance(s_out, list):

        gip_left = [None for _ in range(num_of_matrices)]
        gip_left[0] = gr_left[0] @ s_out[0] @ gr_left[0].conj().T

        for q in range(num_of_matrices - 1):
            sla2 = mat_l_list[q] @ gip_left[q] @ mat_u_list[q].conj().T
            prom = s_out[q + 1] + sla2
            gip_left[q + 1] = gr_left[q + 1] @ prom @ gr_left[q + 1].conj().T

        # ---------------------------------------------------------------

        gpl = [None for _ in range(num_of_matrices-1)]
        gpu = [None for _ in range(num_of_matrices-1)]
        gpd = [i.clone() for i in gip_left]

        for q in range(num_of_matrices - 2, -1, -1):  # Recursive algorithm
            gpl[q] = grd[q + 1] @ mat_l_list[q] @ gip_left[q] + \
                     gpd[q + 1] @ mat_l_list[q].conj().T @ gr_left[q].conj().T
            gpd[q] = gip_left[q] + \
                             gr_left[q] @ mat_u_list[q] @ gpd[q + 1] @ mat_l_list[q].conj().T @ \
                                 gr_left[q].conj().T + \
                             ((gip_left[q]@ mat_u_list[q].conj().T @ grl[q].conj().T) + (gru[q] @
                                mat_l_list[q] @ gip_left[q]))

            gpu[0] = gpl[0].conj().T

    # -------------------------------------------------------------------
    # -- remove energy from the main diagonal of th Hamiltonian matrix --
    # -------------------------------------------------------------------

    for jj, item in enumerate(mat_d_list):
        mat_d_list[jj] = mat_d_list[jj] + (energy - 1j * eta) * sd[jj]
    for jj, item in enumerate(mat_l_list):
        mat_l_list[jj] = mat_l_list[jj] + (energy - 1j * eta) * sl[jj]
    for jj, item in enumerate(mat_u_list):
        mat_u_list[jj] = mat_u_list[jj] + (energy - 1j * eta) * su[jj]

    # -------------------------------------------------------------------
    # ---- choose a proper output depending on the list of arguments ----
    # -------------------------------------------------------------------

    if not isinstance(s_in, list) and not isinstance(s_out, list):
        return g_trans, \
               grd, grl, gru, gr_left, \
               None, None, None, None, \
               None, None, None, None

    elif isinstance(s_in, list) and not isinstance(s_out, list):
        return g_trans, \
               grd, grl, gru, gr_left, \
               gnd, gnl, gnu, gin_left, \
               None, None, None, None

    elif not isinstance(s_in, list) and isinstance(s_out, list):
        return g_trans, \
               grd, grl, gru, gr_left, \
               None, None, None, None, \
               gpd, gpl, gpu, gip_left

    else:
        return g_trans, \
               grd, grl, gru, gr_left, \
               gnd, gnl, gnu, gin_left, \
               gpd, gpl, gpu, gip_left


def recursive_gf(energy, hl, hd, hu, sd, su, sl, left_se, right_se, seP=None, chemiPot=0.0, s_in=0, s_out=0,
                 eta=1e-5):
    
    """The recursive Green's function algorithm is taken from
    M. P. Anantram, M. S. Lundstrom and D. E. Nikonov, Proceedings of the IEEE, 96, 1511 - 1550 (2008)
    DOI: 10.1109/JPROC.2008.927355

    Obitan various green function for later calculations.

    Parameters
    ----------
    energy : numpy.ndarray (dtype=numpy.float)
        Energy array
    mat_d_list : list of numpy.ndarray (dtype=numpy.float)
        List of diagonal blocks
    mat_u_list : list of numpy.ndarray (dtype=numpy.float)
        List of upper-diagonal blocks
    mat_l_list : list of numpy.ndarray (dtype=numpy.float)
        List of lower-diagonal blocks
    s_in : Sigma_in contains self-energy about electron phonon scattering
         (Default value = 0)
    s_out :
         (Default value = 0)
    damp :
         (Default value = 0.000001j)

    Returns
    -------
    g_trans : numpy.ndarray (dtype=numpy.complex)
        Blocks of the retarded Green's function responsible for transmission
    grd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    grl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gru : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gr_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    gnd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    gnl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gnu : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gin_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    gpd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    gpl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gpu : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gip_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    """

    shift_energy = energy + chemiPot

    temp_mat_d_list = [hd[i] * 1. for i in range(len(hd))]
    temp_mat_l_list = [hl[i] * 1. for i in range(len(hl))]
    temp_mat_u_list = [hu[i] * 1. for i in range(len(hu))]
    if seP is not None:
        for i in range(len(temp_mat_d_list)):
            temp_mat_d_list[i] = temp_mat_d_list[i] + seP[i]

    if isinstance(left_se, torch.Tensor):
        s01, s02 = temp_mat_d_list[0].shape
        se01, se02 = left_se.shape
        idx0, idy0 = min(s01, se01), min(s02, se02)
        temp_mat_d_list[0][:idx0,:idy0] = temp_mat_d_list[0][:idx0,:idy0] + left_se[:idx0,:idy0]

    if isinstance(right_se, torch.Tensor):
        s11, s12 = temp_mat_d_list[-1].shape
        se11, se12 = right_se.shape
        idx1, idy1 = min(s11, se11), min(s12, se12)
        # right_se = right_se[-idx1:, -idy1:]
        temp_mat_d_list[-1][-idx1:, -idy1:] = temp_mat_d_list[-1][-idx1:, -idy1:] + right_se[-idx1:, -idy1:]

    ans = recursive_gf_cal(shift_energy, temp_mat_l_list, temp_mat_d_list, temp_mat_u_list, sd, su, sl, s_in=s_in, s_out=s_out, eta=eta)

    if isinstance(left_se, torch.Tensor):
        temp_mat_d_list[0][:idx0, :idy0] = temp_mat_d_list[0][:idx0, :idy0] - left_se[:idx0, :idy0]

    if isinstance(right_se, torch.Tensor):
        temp_mat_d_list[-1][-idx1:, -idy1:] = temp_mat_d_list[-1][-idx1:, -idy1:] - right_se[-idx1:, -idy1:]

    if seP is not None:
        for i in range(len(temp_mat_d_list)):
            temp_mat_d_list[i] = temp_mat_d_list[i] - seP[i]

    return ans