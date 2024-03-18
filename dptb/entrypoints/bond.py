from typing import Dict, List, Optional, Any
import ase.io as io
from pathlib import Path
from dptb.utils.tools import get_neighbours
import os


def bond(
        struct: str,
        accuracy: float,
        cutoff: float,
        log_level: int,
        log_path: Optional[str],
        **kwargs
):
    atom = io.read(struct)
    nb = get_neighbours(atom=atom, cutoff=cutoff, thr=accuracy)

    count = 0
    out = ""
    for k,v in nb.items():
        out += "%10s" % k
        if len(v)>count:
            count = len(v)
        if len(v) != 0:
             for i in v:
                 out += '%10.2f'%i
        out += "\n"

    out = "%10s"*(count+1) % tuple(["Bond Type"] + list(range(1,count+1))) + \
        "\n"+ "--"*6*(count+1)+"\n"+ out

    print(out)

    return out
