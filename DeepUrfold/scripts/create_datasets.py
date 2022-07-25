#!/project/ppi_workspace/toil-env36/bin/python
#SBATCH -A muragroup
#SBATCH -N 1
#SBATCH --time=3-00:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000

import os, sys
sys.path.append('/project/ppi_workspace/py37/lib/python3.7/site-packages')

from toil.realtimeLogger import RealtimeLogger

import itertools as it


from DeepUrfold.Datasets.DDIBindingSiteDataset import DDIBindingSiteDataset
from DeepUrfold.Datasets.DomainStructureDataset import DomainStructureDataset, get_superfamilies
#get_superfamilies = lambda: ["3.80.10.10"]
#["1.10.10.10", "1.10.490.10", "1.20.1260.10", "1.10.238.10"]
#["3.30.310.60", "3.30.1380.10"] #, "3.10.20.30", "3.90.79.10", "3.90.420.10", "1.10.510.10"]
#["3.40.50.300", "3.30.310.60", "3.30.1360.40", "3.30.1380.10", "3.30.230.10", "3.30.300.20", "3.30.1370.10"]#["3.10.20.30", "3.90.79.10", "3.90.420.10"] #["3.40.50.720"] #["2.60.40.10", "2.30.30.100", "2.40.50.140"]

from molmimic.util.toil import createToilOptions, map_job
from molmimic.util import getcwd

def split_dataset_at_level(job, level, dataset_type="ddi", data_dir=None):
    assert level in DomainStructureDataset.hierarchy, "Level ({}) not in hierarchy".format(level, DomainStructureDataset.hierarchy)
    assert dataset_type in ["ddi", "ddi-mask", "domain"] #, DDIBindingSiteDataset, DomainStructureDataset]

    if data_dir is None:
        data_dir = getcwd()

    train_files = os.path.join(data_dir, "train_files")
    if dataset_type == "ddi":
        dataset = DDIBindingSiteDataset(os.path.join(train_files, "DDIBindingSiteDataset-full-train.h5"))
        prefix = "DDIBindingSiteDataset"
    elif dataset_type == "ddi":
        dataset = DDIBindingSiteDataset(os.path.join(train_files, "DDIBindingSiteDataset-full-train-mask.h5"))
        prefix = "DDIBindingSiteDataset-mask"
    elif dataset_type == "domain":
        dataset = DomainStructureDataset(os.path.join(train_files, "DomainStructureDataset-full-train.h5"))
        prefix = "DomainStructureDataset"

    #Create train/validation
    train, valid_test = dataset.split(0.8, level=level)

    #Split into 10% / 10%, half of the new dataset
    valid, test = valid_test.split(0.5, level=level)

    cluster_key = ".".join(DomainStructureDataset.hierarchy[:DomainStructureDataset.hierarchy.index(level)+1])

    train.to_hdf(os.path.join(train_files, "{}-train-{}-split0.8.h5".format(prefix, cluster_key)))
    valid.to_hdf(os.path.join(train_files, "{}-valid-{}-split0.1.h5".format(prefix, cluster_key)))
    test.to_hdf(os.path.join(train_files, "{}-test-{}-split0.1.h5".format(prefix, cluster_key)))

def split_sfam_dataset_at_level(job, sfam, level, dataset_type="ddi", data_dir=None):
    assert level in DomainStructureDataset.hierarchy
    assert dataset_type in ["ddi", "ddi-mask", "domain"]

    if data_dir is None:
        data_dir = getcwd()

    train_files = os.path.join(data_dir, "train_files", *sfam.replace("/", ".").split("."))
    if dataset_type == "ddi":
        dataset = DDIBindingSiteDataset(os.path.join(train_files, "DDIBindingSiteDataset-full-train.h5"))
        prefix = "DDIBindingSiteDataset"
    elif dataset_type == "ddi":
        dataset = DDIBindingSiteDataset(os.path.join(train_files, "DDIBindingSiteDataset-full-train-mask.h5"))
        prefix = "DDIBindingSiteDataset-mask"
    elif dataset_type == "domain":
        dataset = DomainStructureDataset(os.path.join(train_files, "DomainStructureDataset-full-train.h5"))
        prefix = "DomainStructureDataset"

    #Create train/validation
    train, valid_test = dataset.split(0.8, level=level)

    #Split into 10% / 10%, half of the new dataset
    valid, test = valid_test.split(0.5, level=level)

    cluster_key = ".".join(DomainStructureDataset.hierarchy[:DomainStructureDataset.hierarchy.index(level)+1])

    train.to_hdf(os.path.join(train_files, "{}-train-{}-split0.8.h5".format(prefix, cluster_key)))
    valid.to_hdf(os.path.join(train_files, "{}-valid-{}-split0.1.h5".format(prefix, cluster_key)))
    test.to_hdf(os.path.join(train_files, "{}-test-{}-split0.1.h5".format(prefix, cluster_key)))

def split_sfam_dataset(job, sfam, dataset_type="ddi", data_dir=None):
    if data_dir is None:
        data_dir = getcwd()

    if dataset_type == "ddi":
        prefix = "DDIBindingSiteDataset"
    elif dataset_type == "ddi-mask":
        prefix = "DDIBindingSiteDataset-mask"
    elif dataset_type == "domain":
        prefix = "DomainStructureDataset"

    sfam_dir = os.path.join(data_dir, "train_files", *sfam.replace("/", ".").split("."))

    cluster_key = lambda l: ".".join(DomainStructureDataset.hierarchy[:DomainStructureDataset.hierarchy.index(l)+1])

    def train_file_exists(level, train_type):
        split = 0.8 if train_type=="train" else 0.1
        base = "{}-train-{}-split{}.h5".format(prefix, cluster_key(level), split)
        return os.path.isfile(os.path.join(sfam_dir, base))

    for level in ["S35", "S60", "S95", "S100"]:
        if not all(train_file_exists(level, t) for t in ["train", "valid", "test"]):
            job.addChildJobFn(split_sfam_dataset_at_level, sfam, level, dataset_type=dataset_type, data_dir=data_dir)

def split_sfam_datasets(job, dataset_type="ddi", data_dir=None, force=False, include_sfam=None, exclude_sfam=None):
    if data_dir is None:
        data_dir = getcwd()

    if dataset_type == "ddi":
        prefix = "DDIBindingSiteDataset"
    elif dataset_type == "ddi-mask":
        prefix = "DDIBindingSiteDataset-mask"
    elif dataset_type == "domain":
        prefix = "DomainStructureDataset"

    #Create individual superfamily Datasets
    sfam_dir = lambda sfam: os.path.join(data_dir, "train_files", *sfam.replace("/", ".").split("."))

    if include_sfam is not None and len(include_sfam)>0:
        sfams = include_sfam
    else:
        sfams = get_superfamilies()

    sfams = [sfam for sfam in sfams if \
        os.path.isfile(os.path.join(sfam_dir(sfam), "{}-full-train.h5".format(prefix)))]

    if exclude_sfam is not None and len(exclude_sfam)>0:
        sfams = list(set(sfams)-set(exclude_sfam))

    if len(sfams) == 0:
        return

    cluster_key = lambda l: ".".join(DomainStructureDataset.hierarchy[:DomainStructureDataset.hierarchy.index(l)+1])

    def train_file_exists(sfam, level, train_type):
        split = 0.8 if train_type=="train" else 0.1
        base = "{}-train-{}-split{}.h5".format(prefix, cluster_key(level), split)
        return os.path.isfile(os.path.join(sfam_dir(sfam), base))

    sfams = list(set([sfam for sfam in sfams for level in ["S35", "S60", "S95", "S100"] if \
        force or not all(train_file_exists(sfam, level, t) for t in ["train", "valid", "test"])]))

    RealtimeLogger.info("Running {} SFAMS".format(len(sfams)))
    if len(sfams) == 0:
        return

    map_job(job, split_sfam_dataset, sfams, dataset_type=dataset_type, data_dir=data_dir)

def split_dataset(job, dataset_type="ddi", data_dir=None, force=False):
    if data_dir is None:
        data_dir = getcwd()

    if dataset_type == "ddi":
        prefix = "DDIBindingSiteDataset"
    elif dataset_type == "ddi-mask":
        prefix = "DDIBindingSiteDataset-mask"
    elif dataset_type == "domain":
        prefix = "DomainStructureDataset"

    train_files = os.path.join(data_dir, "train_files")

    #Create mixed superfamily datasets
    for level in DomainStructureDataset.hierarchy:
        cluster_key = ".".join(DomainStructureDataset.hierarchy[:DomainStructureDataset.hierarchy.index(level)+1])

        train_file = os.path.join(train_files, "{}-train-{}-split0.8.h5".format(prefix, cluster_key))
        valid_file = os.path.join(train_files, "{}-valid-{}-split0.1.h5".format(prefix, cluster_key))
        test_file = os.path.join(train_files, "{}-test-{}-split0.1.h5".format(prefix, cluster_key))

        if not all(os.path.isfile(f) for f in (train_file, valid_file, test_file)):
            job.addChildJobFn(split_dataset_at_level, level, dataset_type=dataset_type, data_dir=data_dir)

def create_dataset(job, dataset_type="ddi", data_dir=None, force=False, include_sfam=None, exclude_sfam=None):
    assert dataset_type in ["ddi", "ddi-mask", "domain"] #, DDIBindingSiteDataset, DomainStructureDataset]

    if data_dir is None:
        data_dir = getcwd()

    if dataset_type == "ddi":
        dataset = DDIBindingSiteDataset
    elif dataset_type == "domain":
        dataset = DomainStructureDataset

    RealtimeLogger.info("Running {}".format(dataset_type))

    #Returns a rv promise of a datframe
    dataset.from_superfamilies_directory(
        data_dir,
        with_sequences=True,
        job=job,
        include_sfam=include_sfam,
        exclude_sfam=exclude_sfam,
        force=force
    )

def start_toil(job, dataset_type="ddi", data_dir=None, force=False, include_sfam=None, exclude_sfam=None):
    assert dataset_type in ["ddi", "ddi-mask", "domain"]

    if data_dir is None:
        data_dir = getcwd()

    train_files = os.path.join(data_dir, "train_files")

    if dataset_type == "ddi":
        dataset_file = os.path.join(train_files, "DDIBindingSiteDataset-full-train.h5")
        dataset_types = ["ddi", "domain"]
    elif dataset_type == "ddi-mask":
        dataset_file = os.path.join(train_files, "DDIBindingSiteDataset-full-train-mask.h5")
        dataset_types = ["ddi-mask", "domain"]
    else:
        dataset_file = os.path.join(train_files, "DomainStructureDataset-full-train.h5")
        dataset_types = ["domain"]

    if force or not os.path.isfile(dataset_file):
        job.addChildJobFn(create_dataset, dataset_type, data_dir=data_dir, force=force, include_sfam=include_sfam, exclude_sfam=exclude_sfam)

    #job.addFollowOnJobFn(split_dataset, dataset_type, data_dir=data_dir)
    for dtypes in dataset_types:
        job.addFollowOnJobFn(split_sfam_datasets, dtypes, data_dir=data_dir,
            force=force, include_sfam=include_sfam, exclude_sfam=exclude_sfam)

if __name__ == "__main__":
    from toil.common import Toil
    from toil.job import Job

    parser = createToilOptions()
    parser.add_argument("--dataset-type", default="ddi", choices=["domain", "ddi", "ddi-mask"])
    parser.add_argument("--data-dir", default=getcwd())
    parser.add_argument("--force", default=False, action="store_true")
    parser.add_argument("--include-sfam", default=None, nargs="+")
    parser.add_argument("--exclude-sfam", default=None, nargs="+")

    options = parser.parse_args()

    with Toil(options) as workflow:
        workflow.start(Job.wrapJobFn(start_toil, dataset_type=options.dataset_type,
            data_dir=options.data_dir, include_sfam=options.include_sfam,
            exclude_sfam=options.exclude_sfam, force=options.force))
