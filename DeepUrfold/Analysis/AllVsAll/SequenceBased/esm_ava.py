import os
import glob
import time
import argparse
import multiprocessing

from DeepUrfold.Analysis.AllVsAll.SequenceBased import SequenceBasedAllVsAll

import torch
import numpy as np
from Bio import SeqIO

from esm import pretrained, MSATransformer



class ESM(SequenceBasedAllVsAll):
    NJOBS_CPU = 12
    NJOBS = torch.cuda.device_count() #Number of GPUs
    GPU = True
    METHOD = None #Must subclass

    def create_superfamily_alignment(self, i, superfamily, sequences, out_file=None):
        #Alignments not needed
        return superfamily, None

    def train_all(self):
        #Make sure model is downloaded
        pretrained.load_model_and_alphabet(self.METHOD)

        #Use same model for all superfamilies since it was trained on all of Uniref90
        return {sfam:self.METHOD for sfam in self.superfamily_datasets.keys()}

    def load_sequences(self, sequences, ungap=True):
        if ungap:
            ungap = lambda x: x.replace(".", "").replace("-", "")
        else:
            ungap = lambda x: x
        sequence_dict = [
            (record.description, ungap(str(record.seq).upper()))
            for record in SeqIO.parse(sequences, "fasta")
        ]
        return sequence_dict

    def encode_sequences(self, sequences, batch_converter):
        _, _, tokens = batch_converter(sequences)
        return tokens

    def score_tokens(self, tokens, gpu):
        with torch.no_grad():
            token_probs = torch.log_softmax(self.esm_model(tokens.cuda(gpu))["logits"], dim=-1)

        return token_probs

    def generate_per_sequences_scores(self, token_probs, sequences_dict):
        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, (_, seq) in enumerate(sequences_dict):
            print(token_probs.size())
            print(token_probs[i, 1 : len(seq) + 1])
            print(token_probs[i, 1 : len(seq) + 1].mean(0))
            print()
            sequence_representations.append(token_probs[i, 1 : len(seq) + 1].mean(0))
        #print(sequence_representations)
        sequence_representations = torch.FloatTensor(sequence_representations)

        return sequence_representations

    def score_sequences(self, sequences, batch_converter, gpu, ungap=True):
        sequences_dict = self.load_sequences(sequences, ungap=ungap)
        tokens = self.encode_sequences(sequences_dict, batch_converter)
        token_probs = self.score_tokens(tokens, gpu)
        return self.generate_per_sequences_scores(token_probs, sequences_dict)

    def infer_gpu(self, i, model_name, model_path, combined_sequences, gpu=0):
        print(i, model_name, model_path, combined_sequences, gpu)

        results_file = f"{self.METHOD}_{model_name}_infer_results.hd5"

        if not self.force and os.path.isfile(results_file):
            results.to_hdf(results_file, "table")
            return results

        self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(model_path)
        self.esm_model.eval()
        if torch.cuda.is_available():
            self.esm_model = self.esm_model.cuda(gpu)

        batch_converter = self.alphabet.get_batch_converter()

        superfamily_token_probs = self.score_sequences(self.raw_sequences[model_name], batch_converter, gpu)
        combined_token_probs = self.score_sequences(combined_sequences, batch_converter, gpu)

        #From esm predict.py: mutant-wildtype
        #score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]

        cathDomains = [cathDomain for cathDomain, _ in self.iterate_sequences(combined_sequences)]
        cathDomains = pd.Series(cathDomains, name="cathDomain")
        results = cathDomains.to_frame().T
        results = pd.merge(results, self.representative_domains, on="cathDomain")
        results = results.rename(columns={"superfamily":"true_sfam"})

        scores = torch.mean(torch.stack([batch_representations-sequence_representations \
            for (seq_id, _), sequence_token in \
            zip(superfamily_datasets, superfamily_tokens)]))

        results = results.assign(scores=scores.cpu().numpy())
        results = results.set_index(["cathDomain", "true_sfam"])

        results.to_hdf(results_file, "table")

        return results

class ESM1b(ESM):
    METHOD = "esm1b_t33_650M_UR50S"

class ESMMSA1b(ESM):
    METHOD = "esm_msa1b_t12_100M_UR50S"

    def score_tokens(self, batch_tokens, sequences_dict, gpu):
        all_token_probs = []
        for i in range(tokens.size(2)):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, 0, i] = self.alphabet.mask_idx  # mask out first sequence
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    self.esm_model(batch_tokens_masked.cuda())["logits"], dim=-1
                )
            all_token_probs.append(token_probs[:, 0, i])  # vocab size
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)

class ESM1v(ESM):
    METHOD = "esm1v_t33_650M_UR90S_1"

class ESMSeqRep(SequenceBasedAllVsAll):
    NJOBS = 1
    METHOD = "ESM-1b"
    DATA_INPUT = "CATH S35"
    SCORE_INCREASING = False
    MODEL_TYPE = "HMM"
    SCORE_METRIC = "Euclidean Distance"

    def create_superfamily_alignment(self, i, superfamily, sequences, out_file=None):
        #Alignments not needed
        return superfamily, None

    def train_all(self):
        #Use same model for all superfamilies since it was trained on all of Uniref90
        return {sfam:None for sfam in self.superfamily_datasets.keys()}

    def infer_all(self, models, combined_sequences):
        import pandas as pd
        import esm
        sfam_labels = {}
        data = []
        embeddings = {}

        results_file = f"ESM-1b_results.h5"
        if os.path.isfile(results_file):
            return results_file

        for superfamily, seq_file in sorted(self.raw_sequences.items(), key=lambda x:x[0]):
            for sequence in SeqIO.parse(seq_file, "fasta"):
                m = self.id_parser.search(sequence.id)
                if m and m.groups()[0] in self.representative_domains["cathDomain"].values:
                    cathDomain = m.groups()[0]
                    data.append((cathDomain, str(sequence.seq)))
                    sfam_labels[cathDomain] = superfamily
                    files = list(glob.glob(os.path.join("out", sequence.id.split("/")[0], "*.pt")))
                    if len(files)>0:
                        embeddings[cathDomain] = torch.load(files[0])["mean_representations"][33].numpy()

        df = pd.DataFrame([d[0] for d in data], columns=["cathDomain"])
        df = df.assign(superfamily=df["cathDomain"].apply(lambda d: sfam_labels[d]))

        if len(embeddings)>0:
            X = np.stack(df["cathDomain"].apply(lambda d: embeddings[d]))
        else:
            assert 0
            #old use extract.py
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            batch_converter = alphabet.get_batch_converter()
            model = model.cuda()
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            batch_size = 4
            sequence_representations = torch.zeros((len(data), 1280))
            with torch.no_grad():
                for i in range(0, len(batch_tokens), batch_size):
                    print("Start", i)
                    torch.cuda.empty_cache()
                    batch = batch_tokens[i:i+batch_size].clone().cuda()
                    results = model(batch, repr_layers=[33], return_contacts=True)
                    token_representations = results["representations"][33]

                    for s, (_, seq) in enumerate(data[i:i+batch_size]):
                        sequence_representations[i:i+s] = token_representations[s, 1 : len(seq) + 1].mean(0).cpu()

                    del batch, token_representations, results

                    torch.cuda.empty_cache()

            # models = [model.cuda(i) for i in range torch.cuda.device_count()]
            # def run_batch(i, batch_size):
            #     from DeepUrfold.Analysis.AllVsAll import get_available_gpu
            #     gpu = get_available_gpu("esm", i)
            #     batch = batch_tokens[i:i+batch_size].clone().cuda()
            #     results = model(batch, repr_layers=[33])
            #     token_representations = results["representations"][33]
            #
            #     sequence_representations
            #     for s, (_, seq) in enumerate(data[i:i+batch_size]):
            #         sequence_representations.append(token_representations[s, 1 : len(seq) + 1].mean(0).cpu())
            #
            #     del batch, results
            #
            #     return sequence_representations

        import umap
        from sklearn.manifold import TSNE
        from matplotlib import pyplot as plt
        import seaborn as sns

        X_embedded_t = TSNE(n_components=2).fit_transform(X)
        tsne_df = df.assign(**{"Dimension 1": X_embedded_t[:,0], "Dimension 2": X_embedded_t[:,1]})
        sns.scatterplot(data=tsne_df, x="Dimension 1", y="Dimension 2", hue="superfamily")
        means = tsne_df.groupby("superfamily")[["Dimension 1", "Dimension 2"]].mean().reset_index(drop=False)
        sns.scatterplot(data=means, x="Dimension 1", y="Dimension 2", hue="superfamily", marker="X", edgecolors="black", size=100, legend=False, s=100)

        plt.title("T-SNE of ESM data")
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("esm-tsne.png", bbox_inches="tight")

        from sklearn.metrics import pairwise_distances

        distances = pairwise_distances(X_embedded_t, n_jobs=20)
        distances = pd.DataFrame(distances, index=df.index, columns=df.index)

        sfam_groups = df.groupby("superfamily")

        ordered_superfamilies = sorted(self.raw_sequences.keys())

        results_df = pd.DataFrame(np.nan, index=df.index, columns=ordered_superfamilies)
        results_df = results_df.assign(cathDomain=df["cathDomain"], true_sfam=df["superfamily"])
        for sfam1_name in ordered_superfamilies:
            sfam1_domains = sfam_groups.get_group(sfam1_name)
            for sfam2_name in ordered_superfamilies:
                sfam2_domains = sfam_groups.get_group(sfam2_name)
                sfam1_domains_to_sfam2 = distances.loc[list(sfam1_domains.index), list(sfam2_domains.index)].median(axis=1)
                results_df.loc[sfam1_domains.index, sfam2_name] = sfam1_domains_to_sfam2

        results_df = results_df.set_index(["cathDomain", "true_sfam"])
        return results_df

        #results_df.to_hdf(results_file, "table")
        #return results_file

        #
        # plt.clf()
        #
        # X_embedded_u = umap.UMAP(n_components=2).fit_transform(X)
        # umap_df = df.assign(**{"Dimension 1": X_embedded_u[:,0], "Dimension 2": X_embedded_u[:,1]})
        # sns.scatterplot(data=umap_df, x="Dimension 1", y="Dimension 2", hue="superfamily")
        # plt.title("UMAP of ESM data")
        # plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.savefig("esm-umap.png", bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-m", "--model_name", default="ESM-1b", choices=["ESM-1b", "ESM-MSA-1b", "ESM-1v"])
    parser.add_argument("superfamily", nargs="+")
    args = parser.parse_args()

    if args.model_name == "ESM-1v":
        ava_model = ESM1v
    elif args.model_name == "ESM-1b":
        ava_model = ESM1b
    elif args.model_name == "ESM-MSA-1b":
        ava_model = ESMMSA1b

    runner = ESMSeqRep(args.superfamily, args.data_dir, permutation_dir=args.permutation_dir,
        work_dir=args.work_dir, force=args.force)
    runner.run()
