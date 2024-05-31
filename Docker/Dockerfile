#FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="3.7 5.0 6.0 6.1 7.0 7.0+PTX 7.5 8.0 8.6"
#"6.0 6.1 7.0+PTX"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
                           --install-option="--blas=openblas"

RUN apt-get update && apt-get install -y git



WORKDIR /opt
RUN git clone https://github.com/edraizen/Prop3D.git
WORKDIR /opt/Prop3D
RUN python setup.py install
WORKDIR /opt
RUN rm -rf Prop3D

WORKDIR /opt
RUN git clone https://github.com/edraizen/Pytorch-LRP.git
WORKDIR /opt/Pytorch-LRP
RUN python setup.py install
WORKDIR /opt
RUN rm -rf Pytorch-LRP

WORKDIR /opt
ADD . / /opt/DeepUrfold/
WORKDIR /opt/DeepUrfold
RUN python setup.py install

RUN pip install -U requests
RUN pip install -U urllib3

ENTRYPOINT ["python", "-m", "DeepUrfold.Trainers.DistributedTrainSuperfamilyVAE"]
