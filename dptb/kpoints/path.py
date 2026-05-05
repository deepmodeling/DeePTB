import numpy as np


def abacus_kpath(structase, kpath):
    """Build an ABACUS-style high-symmetry k-path."""
    kpath = np.asarray(kpath)
    assert kpath.shape[-1] == 4
    assert len(kpath.shape) == 2
    kpoints = kpath[:, 0:3]
    num_kp = kpath[:, 3].astype(int)
    assert num_kp[-1] == 1

    kpath_list = []
    for i in range(len(kpoints) - 1):
        tmp = np.linspace(kpoints[i], kpoints[i + 1], num_kp[i] + 1)[0:num_kp[i]]
        kpath_list.append(tmp)

    kpath_list.append(kpoints[-1:])
    kpath_list = np.concatenate(kpath_list, axis=0)

    rev_latt = np.linalg.inv(np.array(structase.cell).T)
    kdiff = kpoints[1:] - kpoints[:-1]
    kdiff_cart = np.dot(kdiff, rev_latt)
    kdist = np.linalg.norm(kdiff_cart, axis=1)

    kdist_list = []
    high_sym_kpoints = []
    for i in range(len(kdist)):
        if num_kp[i] == 1:
            kdist[i] = 0
    for i in range(len(kdist)):
        tmp = np.linspace(np.sum(kdist[:i]), np.sum(kdist[:i + 1]), num_kp[i] + 1)[0:num_kp[i]]
        high_sym_kpoints.append(tmp[0])
        if i == 0:
            kdist_list = tmp.copy()
        else:
            kdist_list = np.concatenate([kdist_list, tmp], axis=0)
    kdist_list = np.append(kdist_list, [np.sum(kdist)])
    high_sym_kpoints.append(np.sum(kdist))
    high_sym_kpoints = np.asarray(high_sym_kpoints)

    return kpath_list, kdist_list, high_sym_kpoints


def ase_kpath(structase, pathstr: str, total_nkpoints: int):
    """Build an ASE high-symmetry k-path."""
    kpath = structase.cell.bandpath(pathstr, npoints=total_nkpoints)
    xlist, high_sym_kpoints, labels = kpath.get_linear_kpoint_axis()
    klist = kpath.kpts
    return klist, xlist, high_sym_kpoints, labels


def vasp_kpath(structase, pathstr: list[str], high_sym_kpoints_dict: dict, number_in_line: int):
    """Build a VASP-style high-symmetry k-path."""
    kpath = []
    klist = []

    for i in range(len(pathstr)):
        kline = pathstr[i].split("-")
        kpath.append([high_sym_kpoints_dict[kline[0]], high_sym_kpoints_dict[kline[1]]])
        kline_list = np.linspace(
            high_sym_kpoints_dict[kline[0]],
            high_sym_kpoints_dict[kline[1]],
            number_in_line,
        )
        klist.append(kline_list)
        if i == 0:
            klabels = [pathstr[i].split("-")[0]]
        else:
            if pathstr[i].split("-")[0] == pathstr[i - 1].split("-")[1]:
                klabels.append(pathstr[i].split("-")[0])
            else:
                klabels.append(pathstr[i - 1].split("-")[1] + "|" + pathstr[i].split("-")[0])
            if i == len(pathstr) - 1:
                klabels.append(pathstr[i].split("-")[1])

    kpath = np.asarray(kpath)
    klist = np.concatenate(klist)

    rev_latt = np.linalg.inv(np.array(structase.cell)).T
    kdiff = kpath[:, 1] - kpath[:, 0]
    kdiff_cart = np.dot(kdiff, rev_latt)
    kdist = np.linalg.norm(kdiff_cart, axis=1)

    xlist_label = [0]
    for i in range(len(kdist)):
        if i == 0:
            xlist = np.linspace(0, kdist[i], number_in_line)
        else:
            xlist = np.concatenate([xlist, xlist[-1] + np.linspace(0, kdist[i], number_in_line)])
        xlist_label.append(xlist[-1])

    return klist, xlist, xlist_label, klabels
