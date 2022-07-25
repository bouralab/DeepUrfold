--include="3/40/50/300/*" --include="*/3/30/310/60/*" --include="*/3/30/1360/40/*" --include="*/3/30/1380/10/*" --include="*/3/30/230/10/*" --include="*/3/30/300/20/*" --include="*/3/30/1370/10/*"

sfams="3/40/50/300 3/30/310/60 3/30/1360/40 3/30/1380/10 3/30/230/10 3/30/300/20 3/30/1370/10"
for sfam in $sfams; do
    echo $sfam
    aws s3 sync ./prepared-cath-structures/$sfam/ s3://data-eppic-cath-features/prepared-cath-structures/$sfam/
    aws s3 sync ./cath_features/$sfam/ s3://data-eppic-cath-features/cath_features/$sfam/
    aws s3 sync ./train_files/$sfam/ s3://data-eppic-cath-features/train_files/$sfam/
done

sfams="3.40.50.300 3.30.310.60 3.30.1360.40 3.30.1380.10 3.30.230.10 3.30.300.20 3.30.1370.10"
for sfam in $sfams; do
    echo $sfam
    PYTHONPATH=/home/ubuntu/.local/lib/python3.8/site-packages singularity run --nv /home/ubuntu/SingularityTorch_latest.sif /home/ubuntu/BindingSitePredictor/DeepUrfold/Trainers/TrainSuperfamilyVAE.py --distributed_backend=ddp --superfamily=$sfam --data-dir=/home/ubuntu/data-eppic-cath-features/ --distributed_backend=ddp --num_workers=6 --prefix=$sfam --batch_size=91 --lr=0.2
done
