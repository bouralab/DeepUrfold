#! /bin/bash
#SBATCH -A muragroup
#SBATCH -N 1
#SBATCH --time=7-00:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000

source /project/ppi_workspace/toil-env36/etc/profile.d/conda.sh
conda activate /project/ppi_workspace/py37
export PYTHONPATH=/project/ppi_workspace/py37/lib/python3.7/site-packages

if [[ $# -eq 0 ]] ; then
  export TOIL_SLURM_ARGS="-n 1 -N 1 -A muragroup -t 10:00:00 -p standard"
  python -m DeepUrfold.scripts.create_datasets file:run-create_dataset-`ls -d -1 run-create_dataset-* | wc -l` --realTimeLogging --batchSystem Slurm --defaultCores 1 --defaultMemory 12G --retryCount 5 --dataset-type domain --force --exclude-sfam 2.60.40.10 2.30.30.100 2.40.50.140 --data-dir /project/ppi_workspace/data_eppic_cath_features/
else
  python -m DeepUrfold.scripts.create_datasets file:run-create_dataset-`ls -d -1 run-create_dataset-* | wc -l` --realTimeLogging --defaultCores 1 --maxLocalJobs 20 --defaultMemory 12G --dataset-type domain --force --exclude-sfam 2.60.40.10 2.30.30.100 2.40.50.140 --data-dir /project/ppi_workspace/data_eppic_cath_features/
fi
