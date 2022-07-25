import sys
import argparse

import h5pyd

from DeepUrfold.scripts.rearrange_ss import MLP

def deloop_proteins(name, *keys):
    with h5pyd.File(name, use_cache=False) as f:

    mlp = MLP(run=False)
    ss_atoms = [a for ss in mlp.ss_groups for a in ss["serial_number"]]
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("key", nargs="+")

    args = parser.parse_args()
