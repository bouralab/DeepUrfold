import os
import sys
import glob
from DeepUrfold.Metrics.MetricSaver import RawMetricSaver

def umap(directory):
    all_models = {}
    single_models = {}

    for f in glob.glob(os.path.join(directory, "*", "*", "latent_space", "*-mean-mean.pt")):
        fields = f.split("/")
        sfam_model = fields[-4]
        sfam_input = fields[-3]

        if sfam_model == sfam_input:
            all_models[sfam_model] = f

        try:
            single_models[sfam_model][sfam_input] = f
        except KeyError:
            single_models[sfam_model] = {sfam_input: f}

    for method in ["umap", "pca", "t-sne"]:
        RawMetricSaver.reduce(prefix="All models", method=method, **all_models)

    for sfam_model, labelled_files in single_models.items():
        for method in ["umap", "pca", "t-sne"]:
            RawMetricSaver.reduce(prefix=sfam_model, method=method, **labelled_files)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        umap(os.getcwd())
    else:
        umap(sys.argv[1])
