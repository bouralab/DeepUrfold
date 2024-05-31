import json
import shutil
import contextlib
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm
from biotite.structure.io.pdb import PDBFile
from biotite.structure import get_residue_starts
from biotite.structure import renumber_res_ids
import joblib
from joblib import Parallel, delayed

scop_path = Path("/media/smb-rivanna/ed4bu/SCOPe/pdbstyle-2.08")
fragment_dir = Path("/media/smb-rivanna/ed4bu/SCOPe/fuzzle")
fragment_dir.mkdir(exist_ok=True)

frag_reps = fragment_dir / "representatives"
frag_reps.mkdir(exist_ok=True)

import requests
import shutil
import functools

def download_file(url, outdir=None):
    if url.endswith("/"):
        local_filename = url[:-1].split('/')[-1]
    else:
        local_filename = url.split('/')[-1]

    if outdir is not None:
        local_filename = Path(outdir) / local_filename
    else:
        local_filename = Path(local_filename)

    with requests.get(url, stream=True) as r:
        with local_filename.open('wb') as f:
            r.raw.read = functools.partial(r.raw.read, decode_content=True)
            shutil.copyfileobj(r.raw, f)

    return local_filename

@contextlib.contextmanager
def tqdm_joblib(tqdm_object=None, desc=None, total=None):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    if tqdm_object is None and total is None:
        raise RuntimeError("Must input either tqdm_object or total")
    elif tqdm_object is None:
        tqdm_object = tqdm(desc=desc, total=total)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def process_fragment(frag_id, node):
    pdb_file = scop_path / node["domain"][2:4] / f"{node['domain']}.ent"

    if not pdb_file.is_file():
        return None, None

    fragment_file = fragment_dir / str(frag_id) / f"{node['name']}.pdb"
    fragment_file.parent.mkdir(exist_ok=True)

    if node["selected_domain"]==node["domain"]:
        fragment_file = fragment_file.with_suffix(".representative.pdb")

    protein = PDBFile.read(str(pdb_file)).get_structure(model=1, extra_fields=["b_factor", "atom_id"], altloc='first')
    protein = renumber_res_ids(protein, start=0)

    cut_fragment = protein[np.isin(protein.res_id, np.arange(node['start'], node['end']+1))]
    small_pdb_file = PDBFile()
    small_pdb_file.set_structure(cut_fragment)
    small_pdb_file.write(str(fragment_file))

    if node["selected_domain"]==node["domain"]:
        shutil.copyfile(str(fragment_file), str(frag_reps / f"fragment{frag_id}.pdb"))

    return  node['name'], fragment_file

def process_fragments(frag_id):
    fragments_file = download_fragments(frag_id)
    print(fragments_file)
    with fragments_file.open() as f:
        fragments = json.load(f)["nodes"]
    
    for fragment in fragments:
        process_fragment(frag_id, fragment)

def download_fragments(frag_id):
    frag_json = fragment_dir / str(frag_id) / f"2.07.1.25-3.fragment{frag_id}.json"
    frag_json.parent.mkdir(exist_ok=True)

    def check(f):
        with f.open() as fh:
            return "<html>" not in next(f)

    if not frag_json.is_file() or not check(frag_json):
        download_file(f"https://fuzzle.uni-bayreuth.de:8443/static/js/2.07.1.25-3.fragment{frag_id}.json", outdir=frag_json.parent)
    return frag_json

def parse_fuzzle_old(json_file):
    with open(json_file) as f:
        fragments = json.load(f)

    with tqdm_joblib(total=len(fragments["nodes"])):
        Parallel(n_jobs=64)(delayed(process_fragments)(frag) for frag in fragments["nodes"])

def parse_fuzzle(csv_file):
    with csv_file.open() as f:
        num_fragments = sum(1 for _ in f)

    with tqdm_joblib(total=num_fragments):
        Parallel(n_jobs=1)(delayed(process_fragments)(frag) for frag in range(num_fragments))

if __name__ == "__main__":
    # json_file = "psiblast2.06.1.25-3.json"
    # if not Path(json_file).is_file():
    #     download_file("https://fuzzle.uni-bayreuth.de/static/js/psiblast2.06.1.25-3.json")

    csv_file = Path("download_fragment_table")
    if not csv_file.is_file():
        download_file("https://fuzzle.uni-bayreuth.de:8443/fragments/download_fragment_table/")

    parse_fuzzle(csv_file)