cd
export VERSION=1.15.6 OS=linux ARCH=amd64 && \
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz
echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && \
    echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && \
<<<<<<< HEAD
    source ~/.bashrc\
export GOPATH=${HOME}/go
export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin
=======
    source ~/.bashrc
>>>>>>> menuka_edits
git clone https://github.com/hpcng/singularity.git
cd singularity
./mconfig
./mconfig && \
    make -C ./builddir && \
    sudo make -C ./builddir install
cd
singularity pull shub://edraizen/SingularityTorch

git clone https://github.com/edraizen/Prop3D.git
cd Prop3D
git checkout conv-py3
python3 -m pip install -U .

cd

git clone https://github.com/edraizen/BindingSitePredictor.git
cd BindingSitePredictor
git checkout Minkowski
python3 -m pip install -U .

cd

pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@master --upgrade

<<<<<<< HEAD
python3 -m pip install awscli
python3 -m awscli configure
=======
sudo apt install awscli -y
aws configure
>>>>>>> menuka_edits

SFAM=$1
SFAM_PATH=${SFAM//./\/}

mkdir data-eppic-cath-features
cd data-eppic-cath-features
<<<<<<< HEAD
python3 -m awscli s3 sync sync s3://data-eppic-cath-features . --exclude="*" --include="*/$SFAM_PATH/*" --exclude="cath_interfaces/*"
=======
aws s3 sync s3://data-eppic-cath-features . --exclude="*" --include="*/$SFAM_PATH/*" --exclude="cath_interfaces/*"
>>>>>>> menuka_edits
cd

pip install tensorboard --user --force-reinstall
pip uninstall psutil

mkdir runs
#cd runs
#aws s3 cp s3://singularity-torch/pytorch-minkowski.simg .
#PYTHONPATH=/home/ubuntu/.local/lib/python3.8/site-packages singularity run --nv ./pytorch-1.6.0-ed4bu.simg /home/ubuntu/BindingSitePredictor/DeepUrfold/Trainers/TrainSuperfamilyVAE.py --superfamily $1 --data-dir /home/ubuntu/data_eppic_cath_features/ --distributed_backend ddp --batch_size 64 --num_workers 28

#ingularity run --nv -B /home/ubuntu/.local/lib/python3.8/site-packages/pytorch_lightning:/opt/conda/lib/python3.7/site-packages/pytorch_lightning pytorch-minkowski.simg /home/ubuntu/BindingSitePredictor/DeepUrfold/Trainers/TrainSuperfamilyVAE.py --superfamily $SFAM --data-dir /home/ubuntu/data_eppic_cath_features/ --distributed_backend ddp --batch_size 28 --num_workers 28 --prefix $SFAM --max_epochs 30
