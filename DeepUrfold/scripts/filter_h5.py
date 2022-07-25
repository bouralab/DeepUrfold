#! /project/ppi_workspace/py37/bin/python
#SBATCH -A muragroup
#SBATCH -p standard
#SBATCH -n 20
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --mem 72000

import os, sys
import pandas as pd
from glob import glob
from joblib import Parallel, delayed

def filter_h5(h5_file):
    try:
        store = pd.HDFStore(h5_file)
        n_keys = len(store.keys())
        store.close()
    except Exception:
        #Invalid file
        n_keys = 0

    if n_keys == 0:
        try:
            os.remove(h5_file)
        except (OSError, FileNotFoundError):
            return

    print(f"Finished {h5_file}")

if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else os.gtcwd()
    hf_files = list(glob(os.path.join(prefix, "*", "*", "*", "*", "*split*.h5")))
    print(f"Verifying {len(hf_files)} datasets")
    Parallel(n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))(
        delayed(filter_h5)(f) for f in hf_files)
