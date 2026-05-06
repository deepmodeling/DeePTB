from ase.data import atomic_masses
from dptb.utils.constants import atomic_num_dict

mass_dict = {
    symbol: float(atomic_masses[atomic_number])
    for symbol, atomic_number in atomic_num_dict.items()
    if atomic_number > 0
}
