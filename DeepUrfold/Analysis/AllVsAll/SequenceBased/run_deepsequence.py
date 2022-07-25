import numpy as np
import time
import sys
sys.path.insert(0, "/opt/DeepSequence/DeepSequence/")
import model
import helper
import train
import os

"""DeepSequence from https://github.com/debbiemarkslab/DeepSequence

This script is intended to be run in the docker image docker://edraizen/deepsequence
that has all of the requirements and correct version of python (2.7) installed.

This script and input files must be in the same directory in order to be found by docker.

Train:

docker run -v $PWD:/data -w /data --gpus all edraizen/deepsequence {file} \
    --alignment datasets/alignment.a2m

Score:

docker run -v $PWD:/data -w /data --gpus all edraizen/deepsequence {file} \
    --alignment datasets/alignment.a2m \
    --parameters path/to/paramters.pkl \
    --aligned_sequences path/to/new_alignment.a2m
""".format(file=os.path.basename(__file__))

os.environ["THEANO_FLAGS"]='floatX=float32,device=cuda'

model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "final_decode_nonlin":  "sigmoid",
    "final_pwm_scale"   :   True,
    "n_pat"             :   4,
    "r_seed"            :   12345,
    "conv_pat"          :   True,
    "d_c_size"          :   40
    }

train_params = {
    "num_updates"       :   300000,
    "save_progress"     :   True,
    "verbose"           :   True,
    "save_parameters"   :   False,
    }

def create_model(alignment_file):
    data_helper = helper.DataHelper(alignment_file=alignment_file,
                                    calc_weights=True)

    vae_model   = model.VariationalAutoencoder(data_helper,
        batch_size                     =   model_params["bs"],
        encoder_architecture           =   [model_params["encode_dim_zero"],
                                                model_params["encode_dim_one"]],
        decoder_architecture           =   [model_params["decode_dim_zero"],
                                                model_params["decode_dim_one"]],
        n_latent                       =   model_params["n_latent"],
        logit_p                        =   model_params["logit_p"],
        sparsity                       =   model_params["sparsity"],
        encode_nonlinearity_type       =   "relu",
        decode_nonlinearity_type       =   "relu",
        final_decode_nonlinearity      =   model_params["final_decode_nonlin"],
        final_pwm_scale                =   model_params["final_pwm_scale"],
        conv_decoder_size              =   model_params["d_c_size"],
        convolve_patterns              =   model_params["conv_pat"],
        n_patterns                     =   model_params["n_pat"],
        random_seed                    =   model_params["r_seed"],
        )

    job_string = helper.gen_job_string({"alignment_file":alignment_file}, model_params)

    return vae_model, data_helper, job_string

def train_model(data_helper, vae_model, job_string):
    train.train(data_helper, vae_model,
        num_updates             =   train_params["num_updates"],
        save_progress           =   train_params["save_progress"],
        save_parameters         =   train_params["save_parameters"],
        verbose                 =   train_params["verbose"],
        job_string              =   job_string)

    vae_model.save_parameters(file_prefix=job_string)

def score(vae_model, parameter_file, sequences_to_score):
    vae_model.load_parameters(parameter_file)
    sequences = DataHelper(alignment_file=sequences_to_score)
    results = np.zeros(sequences.x_train.shape[0], 3)
    for i in range(sequences.x_train.shape[0]):
        batch_preds, _, _ = vae_model.all_likelihood_components(sequences.x_train[i])
        results[i] = batch_preds
    np.save(results, os.path.join(os.path.dirname(sequences_to_score),
        "{}_{}_scores.npy".format(
          os.path.basename(os.path.splitext(parameter_file)[0]),
          os.path.basename(os.path.splitext(sequences_to_score)[0]))))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alignment_file", help="Alignment used to train the model", default=None)
    parser.add_argument("-p", "--parameters", help="Model parameters pickle file prefix", default=None)
    parser.add_argument("-s", "--aligned_sequences", help="Fasta file of sequences to search", default=None)

    args = parser.parse_args()

    if [args.parameters, args.aligned_sequences].count(None) < 2:
        raise RuntimeError("Paramters and aligned sequences must be inputted together")

    if not os.path.isfile(os.path.basename(args.alignment_file)):
        raise RuntimeError("Alignment file must be in same directory")

    args.alignment_file = os.path.basename(args.alignment_file)

    for dir_name in ["logs", "params", "mutations"]:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    vae_model, data_helper, job_string = create_model(args.alignment_file)

    if args.parameters is None:
        train_model(data_helper, vae_model, job_string)
    else:
        if not os.path.isfile(os.path.basename(args.aligned_sequences)):
            raise RuntimeError("Aligned Sequences file must be in same directory")

        args.aligned_sequences = os.path.basename(args.aligned_sequences)
        score(vae_model, args.parameters, args.aligned_sequences)
