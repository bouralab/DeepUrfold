import os
import glob
import pandas as pd
#import infomap

def cluster(compare_dir):
    distances = {}
    for f in glob.glob(os.path.join(compare_dir, "*.csv")):
        df = f.read_csv(f)
        elbo = df["ELBO"].mean(), df["ELBO"].std()
        sources = dict(kv.split("=") for kv in f.split("-").split("_"))
        distances[(source["model"],  sources["input"])] = elbo
        del df

    source, target = zip(*distances.keys())
    superfamiles = list(set(source).union(set(target)))
    superfamiles = {superfamily:i+1 for i, superfamily in enumerate(superfamiles)}

    #im = Infomap("--directed")
    for (source, target), elbo in distances.items():
        #im.add_link(superfamiles[source], superfamiles[target], elbo[0])
        print(superfamiles[source], superfamiles[target], elbo[0])
