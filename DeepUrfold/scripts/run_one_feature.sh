#! /bin/bash
#SBATCH -N 1
#SBATCH -c 28
#SBATCH -p gpu
#SBATCH -A muragroup
#SBATCH --mem=112G
#SBATCH --gres=gpu:p100:4
#SBATCH --time=3-00:00:00
#SBATCH --array=1-9

source /project/ppi_workspace/toil-env36/etc/profile.d/conda.sh
conda activate /project/ppi_workspace/py37

features=$(cat /project/ppi_workspace/BindingSitePredictor/DeepUrfold/scripts/features.txt | sed -n ${SLURM_ARRAY_TASK_ID}p)
#prefix=${features// /_}

echo $features
echo $prefix

PYTHONPATH=/project/ppi_workspace/py37/lib/python3.7/site-packages/ singularity run --nv /project/ppi_workspace/containers/SingularityTorch_latest.simg /project/ppi_workspace/BindingSitePredictor/DeepUrfold/Trainers/TrainSuperfamilyVAE.py --distributed_backend=ddp --superfamily=2.60.40.10 --data-dir=/project/ppi_workspace/data-eppic-cath-features/ --distributed_backend=ddp --num_workers=5 --prefix=2.60.40.10_feat$SLURM_ARRAY_TASK_ID --max_epochs=100 --lr=0.2 --features $features
