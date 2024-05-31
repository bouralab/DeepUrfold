import os
import shutil
import subprocess

from pylab import rcParams
params = {
   'axes.labelsize': 6,
   'font.size': 12,
   #'title.fontweight': 'bold',
   'legend.fontsize': 10,
   'xtick.labelsize': 6,
   'ytick.labelsize': 6,
   'text.usetex': True,
   'text.latex.preamble': r'\usepackage{xcolor}',
   }
rcParams.update(params)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("ticks")

import torch
import glob
import h5pyd
import pandas as pd
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def get_color(name, colors):
    if isinstance(colors, dict):
        for color_name, color_code in colors.items():
            if color_name in name:
                return color_code

    return None

def create_kde_figure(sfam_code, sfam_name=None, title=None, prefix=None, get_representatives=False,
  other_representatives=None, distance_file=None, reps_file=None, other_elbo_scores=None,
  other_single_elbo_scores=None, corner_image=None, corner_image_zoom=0.05, data_directory=None, regular_model=True,
  ax=None, legend=True, colors=None):
      """
      distance_file: all vs all distance file from DeepUrfold.AllVsAll.StructureBased.TMAlign
      reps_file: path to sfam representives in h5
      """

      if not isinstance(sfam_code, dict):
          if not isinstance(sfam_code, (list, tuple)) and not isinstance(sfam_name, (list, tuple)):
              sfam_name = sfam_name if sfam_name is not None else sfam_code
              sfam_code = {sfam_name: sfam_code}
          elif isinstance(sfam_code, (list, tuple)) and isinstance(sfam_name, (list, tuple)):
              assert len(sfam_code)==len(sfam_name)
              sfam_code = {n:c for n,c in zip(sfam_name,sfam_code)}
          else:
              raise RuntimeError("sfam_code must be a dict {name:code} or a single code with an optioanl sfam_name.")

      #Load data for intial sfam and then other_sfams (in that order)
      sfam_to_load = {} #{sfam_name:sfam_code}
      if other_elbo_scores is not None:
          if not isinstance(other_elbo_scores, dict):
              if not isinstance(other_elbo_scores, (list, tuple)):
                  other_elbo_scores = {other_elbo_scores:other_elbo_scores}
              else:
                  other_elbo_scores = {o:o for o in other_elbo_scores}
          sfam_to_load.update(other_elbo_scores)

      elbo_data = {}


      # #Add MLP distrubtion to data after other elbo distribtions
      # if mlp_elbo_file is not None:
      #     elbo_scores = pd.DataFrame({"ELBO":torch.load(mlp_elbo_file), "Dataset":"Multiple Loop Permutation"})
      #     elbo_data["Multiple Loop Permutation"] = torch.load(mlp_elbo_file).numpy()

      #Generate Background distribution (All structures that have a tm-score <= 0.3)
      if distance_file is None:
          distance_file = "/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/seq_based/struct_based/tmalign2/all_distances.csv"

      distances = pd.read_csv(distance_file)
      reps = []

      with h5pyd.File("/home/ed4bu/deepurfold-paper-1.h5", use_cache=False) as f:
          for input_sfam_code in sfam_code.values():
              key = f"{input_sfam_code.replace('.', '/')}/representatives"
              reps2 = list(f[key].keys())
              if len(reps2)==0:
                  reps2 = list(f[key].attrs["missing_domains"])
              reps += reps2

      # distances.chain1 = distances.chain1.apply(lambda s: s.split("/")[-1].split(".")[0])
      # distances.chain2 = distances.chain2.apply(lambda s: s.split("/")[-1].split(".")[0])
      # dissimalar = distances[(distances.chain1.isin(reps.cathDomain))&(distances.moving_tm_score<=0.3)]
      # worst = dissimalar[["chain2", "moving_tm_score"]].drop_duplicates(subset="chain2")
      # sfams = pd.read_csv("ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt",
      #     header=None, names=["cathDomain", *list("CATH"), "S35", "S60", "S95", "S100", "dcount", "len", "res"],
      #     delim_whitespace=True, comment="#")
      # worst_sfam = pd.merge(worst, sfams, left_on="chain2", right_on="cathDomain").dropna()
      # worst_sfam_groups = worst_sfam.groupby(["C", "A", "T", "H"])
      # print(worst_sfam.tail())

      # if get_representatives and other_representatives is not None:
      #     other_rep_sfams = set([map(float, k.split(".")) for k in other_representatives.keys()])
      #     missing_o = other_rep_sfams-set(worst_sfam_groups.groups.keys())
      #     if len(missing_o)>0:
      #         for o in missing_o:
      #             worst_sfam = pd.concat((worst_sfam, pd.DataFrame([["TEST", 0.0, "TEST", *o, *[0]*7]], columns=worst_sfam.columns)))
      #         print(worst_sfam.tail())
      #         worst_sfam_groups = worst_sfam.groupby(["C", "A", "T", "H"])
      #         print(list(worst_sfam_groups.groups.keys()))
      #
      # test_file = None
      # for sfam, domains in worst_sfam_groups:
      #     sfam = list(map(lambda x: str(int(x)), sfam))
      #     for input_sfam_name, input_sfam_code in sfam_code.items():
      #         print(os.path.join("/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/superfamilies_for_paper", input_sfam_code, ".".join(sfam), "elbo", "*.pt"))
      #         elbo_file = list(glob.glob(os.path.join("/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/superfamilies_for_paper", input_sfam_code, ".".join(sfam), "elbo", "*.pt")))[0]
      #         elbo_scores = torch.load(elbo_file).numpy()
      #         full_file = os.path.join("/home/bournelab/data-eppic-cath-features/train_files/", *sfam, "DomainStructureDataset-representatives.h5")
      #         sfam_test_file = pd.read_hdf(full_file, "table")
      #         sfam_test_file = sfam_test_file.assign(superfamily=".".join(sfam), elbo=elbo_scores)
      #         if len(sfam_test_file) > 0 or (len(sfam_test_file)==1 and sfam_test_file.iloc[0].cathDomain!="TEST"):
      #             sfam_test_file = sfam_test_file[sfam_test_file["cathDomain"].isin(worst_sfam.chain2)]
      #             if test_file is None:
      #                 test_file = sfam_test_file
      #             else:
      #                 test_file = pd.concat((test_file, sfam_test_file))
      #         if get_representatives:
      #             sfam_n = ".".join(sfam)
      #             print(sfam_n, input_sfam_code)
      #             if input_sfam_code == sfam_n:
      #                 #Add representives before other elbo distributions
      #                 elbo_data[f"{input_sfam_name} Representatives"] = elbo_scores
      #             elif isinstance(other_representatives, dict) and sfam_n in other_representatives:
      #                 elbo_data[f"{other_representatives[sfam_n]} Representatives"] = elbo_scores

      print(sfam_to_load)
      for other_sfam_name, other_sfam_code in sfam_to_load.items():
          if os.path.isfile(other_sfam_code):
              elbo_data[other_sfam_name] = torch.load(other_sfam_code).numpy()
          else:
              try:
                  with h5pyd.File(other_sfam_code, use_cache=False) as f:
                      pass
                  elbo_prefix = f"ELBO (MODEL={'+'.join(sfam_code.values())}, INPUT={other_sfam_name})"
                  output_file = os.path.join(data_directory, f"{elbo_prefix}-elbo-elbo.pt")
                  if os.path.isfile(output_file):
                      elbo_data[other_sfam_name] = torch.load(output_file).numpy()
                  else:
                      elbo_scores = run_nn(data_directory, elbo_prefix, h5_dir=other_sfam_code)
                      elbo_data[other_sfam_name] = elbo_scores.numpy()
              except Exception:
                  print(other_sfam_name, other_sfam_code)
                  #All vs all
                  path = "/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/superfamilies_for_paper"
                  for input_sfam in sfam_code.values():
                      elbo_file = list(glob.glob(os.path.join(path, input_sfam, other_sfam_code, "elbo", "*.pt")))[0]
                      elbo_data[other_sfam_name] = torch.load(elbo_file).numpy()

      background_scores, representative_scores = get_background(reps, data_directory,
        sfam_code, regular_model=regular_model, other_representatives=other_representatives)

      elbo_data[f"Background (TM-Score $\leq$ 0.3)"] = background_scores

      if representative_scores is not None:
          elbo_data.update(representative_scores)
          print(list(sfam_code.keys()), "=>", elbo_data)

      save_fig = False
      if ax is None:
          fig, ax = plt.subplots()
          save_fig = True

      for name, df in elbo_data.items():
          sns.kdeplot(df, ax=ax, label=name, color=get_color(name, colors))

      if other_single_elbo_scores is not None:
          if isinstance(other_single_elbo_scores, str) and os.path.isfile(other_single_elbo_scores):
              other_single_elbo_scores = pd.read_csv(other_single_elbo_scores)
          elif not isinstance(other_single_elbo_scores, pd.DataFrame):
              raise RuntimeError("other_single_elbo_scores must be a path to csv or dataframe")
          for i, row in other_single_elbo_scores.iterrows():
              name, elbo = row.values.tolist()[-2:]
              name = name.replace("Wild Type", "wild-type")
              if name.startswith("uL") and "ancestral" not in name.lower():
                name_parts = name.split()
                name = f"{name_parts[0]} ({name_parts[1][1:-1]} ancestral reconstruction)"
                print(name)
              color = get_color(name, colors)
              sns.rugplot(x=[elbo], ax=ax, color=color)
              plt.plot([], [], marker="+", color=color, label="{}".format(name.replace("_", "\_"), elbo))
      else:
          other_single_elbo_scores = []

      print(*ax.get_legend_handles_labels())

      _legend = {l: h for h,l in zip(*ax.get_legend_handles_labels())}

      if legend:
          _legend = ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.05, 0.5), #bbox_to_anchor=(0.5, -0.25),
              fancybox=True, frameon=False) #, nrow=np.floor((len(elbo_data)+len(other_single_elbo_scores))/2))
          _legend.get_frame().set_facecolor('none')

      ax.set_xlabel("-(ELBO)")

      if corner_image is not None:
          """Need to autoate! TM-ALign best reps, and orient the standard way, color"""
          im = plt.imread(corner_image)
          imagebox = OffsetImage(im, zoom=corner_image_zoom)
          imagebox.image.axes = ax

          ab = AnnotationBbox(imagebox, (.73,.8),
                              #xybox=(120., -80.),
                              xycoords='axes fraction',
                              boxcoords="axes fraction",
                              #pad=0.5,
                              frameon=False,
                              )

          ax.add_artist(ab)

      if title is not None:
        sfam_code = {title:None}
      if len(sfam_code)==1:
          sfam_name = list(sfam_code.keys())[0]
          color = get_color(f"{sfam_name} Rep", colors) if regular_model else "0000000"
          ending = "-only" if regular_model else ""
          title = f"Using an\n\\textcolor[HTML]{{{color[1:]}}}{{\\textbf{{{sfam_name}{ending}}}}}\nmodel"
      else:
          sfam_names = " $\\cup$ ".join(sfam_code.keys())
          title = f"Using a \\textbf{{Joint}}\n\\textbf{{{sfam_names}}}\nmodel"
          
      #ax.set_title(f"\\textbf{{{prefix}{title}}}", weight='bold')

      if regular_model or len(sfam_code)==2:
        prefix_artist = ax.text(0.091, 253, f"(\\textbf{{{prefix}}})",
          fontsize=10)

        ax.text(0.105, 219, title, fontsize=10)
      else:
        prefix_artist = ax.text(0.091, 138, f"(\\textbf{{{prefix}}})",
          fontsize=10)

        ax.text(0.105, 113, title, fontsize=10)

      if prefix != "A":
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            right=False,
            labelbottom=False) # labels along the bottom edge are off
        ax.axes.yaxis.set_visible(False)

      if save_fig:
          plt.savefig(f"{title.replace(' ', '_')}_kde.pdf")

      return _legend

def get_background(reps, path, sfam_code, regular_model=False, get_representatives=False,
  other_representatives=None, distance_file=None):
    if path is None:
        path = os.getcwd()

    background_file = os.path.join(path, "background.pt")
    elbo_scores = None
    elbo_scores_exist = False
    if os.path.isfile(background_file):
        elbo_scores = torch.load(background_file)
        elbo_scores_exist = True
        if not regular_model:
            return elbo_scores.numpy(), {}
    
    #assert 0, (os.path.isfile(background_file), regular_model)

    if distance_file is None:
        distance_file = "/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/seq_based/struct_based/tmalign2/all_distances.csv"

    distances = pd.read_csv(distance_file)

    distances.chain1 = distances.chain1.apply(lambda s: s.split("/")[-1].split(".")[0])
    distances.chain2 = distances.chain2.apply(lambda s: s.split("/")[-1].split(".")[0])

    #Get scores for representives vs all with a score <=0.3
    dissimalar = distances[(distances.chain1.isin(reps))&(distances.moving_tm_score<=0.3)]
    worst = dissimalar[["chain2", "moving_tm_score"]].drop_duplicates(subset="chain2")
    sfams = pd.read_csv("ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt",
        header=None, names=["cathDomain", *list("CATH"), "S35", "S60", "S95", "S100", "dcount", "len", "res"],
        delim_whitespace=True, comment="#")
    worst_sfam = pd.merge(worst, sfams, left_on="chain2", right_on="cathDomain").dropna()

    worst_sfam_groups = worst_sfam.groupby(["C", "A", "T", "H"])

    cathcodes = " ".join(worst_sfam[["C", "A", "T", "H"]].drop_duplicates().dropna().astype(str).agg('.'.join, axis=1))
    domains = worst_sfam["cathDomain"].dropna().drop_duplicates()

    if not regular_model:
        #raise RuntimeError(f"Background not found, please run evaluation code with )
        background_file = f"ELBO (MODEL={'+'.join(sfam_code.values())}, INPUT=Background)"
        # cmd = "python -m DeepUrfold.Evaluators.EvaluateDistrubutedDomainStructureVAE "
        # cmd += "--data_dir=/home/ed4bu/deepurfold-paper-1.h5 --gpus=1 --batch_size 16 "
        # cmd += f"--num_workers 32 --raw elbo --batchwise_loss False --checkpoint {os.path.join(path, 'last.ckpt')} "
        # cmd += f"--prefix \"{background_file}\" "
        # cmd += "--features 'H;HD;HS;C;A;N;NA;NS;OA;OS;SA;S;Unk_atom__is_helix;is_sheet;Unk_SS__residue_buried__is_hydrophobic__pos_charge__is_electronegative' "
        # cmd += "--feature_groups 'Atom Type;Secondary Structure;Solvent Accessibility;Hydrophobicity;Charge;Electrostatics' "
        # cmd += f"--superfamily {cathcodes} --domains {' '.join(domains.tolist())}"
        # print(f"Running {cmd}")
        # subprocess.call(cmd, cwd=path, shell=True)
        elbo_scores = run_nn(path, background_file, cathcodes=cathcodes, domains=domains)
        # background_file += "-elbo-elbo.pt"
        # background_file = os.path.join(path, background_file)
        # assert os.path.isfile(background_file), "Failed running model"
        # elbo_scores = torch.load(background_file)
        shutil.copyfile(background_file, os.path.join(path, "background.pt"))
        return elbo_scores, {}



    else:
        if regular_model and other_representatives is not None:
            other_rep_sfams = set([map(float, k.split(".")) for k in other_representatives.keys()])
            missing_o = other_rep_sfams-set(worst_sfam_groups.groups.keys())
            if len(missing_o)>0:
                for o in missing_o:
                    worst_sfam = pd.concat((worst_sfam, pd.DataFrame([["TEST", 0.0, "TEST", *o, *[0]*7]], columns=worst_sfam.columns)))
                print(worst_sfam.tail())
                worst_sfam_groups = worst_sfam.groupby(["C", "A", "T", "H"])
                print(list(worst_sfam_groups.groups.keys()))

        representatives = {}
        for sfam, domains in worst_sfam_groups:
            sfam = list(map(lambda x: str(int(x)), sfam))
            for input_sfam_name, input_sfam_code in sfam_code.items():
                #if not elbo_scores_exist:
                #print(os.path.join("/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/superfamilies_for_paper", input_sfam_code, ".".join(sfam), "elbo", "*.pt"))
                elbo_file = list(glob.glob(os.path.join("/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/superfamilies_for_paper", input_sfam_code, ".".join(sfam), "elbo", "*.pt")))[0]
                _elbo_scores = torch.load(elbo_file)
                if not elbo_scores_exist:
                    if elbo_scores is None:
                        elbo_scores = _elbo_scores
                    else:
                        elbo_scores = torch.cat((elbo_scores, _elbo_scores))
                # full_file = os.path.join("/home/bournelab/data-eppic-cath-features/train_files/", *sfam, "DomainStructureDataset-representatives.h5")
                # sfam_test_file = pd.read_hdf(full_file, "table")
                # sfam_test_file = sfam_test_file.assign(superfamily=".".join(sfam), elbo=elbo_scores)
                # if len(sfam_test_file) > 0 or (len(sfam_test_file)==1 and sfam_test_file.iloc[0].cathDomain!="TEST"):
                #     sfam_test_file = sfam_test_file[sfam_test_file["cathDomain"].isin(worst_sfam.chain2)]
                #     if test_file is None:
                #         test_file = sfam_test_file
                #     else:
                #         test_file = pd.concat((test_file, sfam_test_file))
                if regular_model: #get_representatives:
                    sfam_n = ".".join(sfam)
                    print("Get Rep", sfam_n, input_sfam_code, input_sfam_code == sfam_n, isinstance(other_representatives, dict) and sfam_n in other_representatives)
                    if input_sfam_code == sfam_n:
                        #Add representives before other elbo distributions
                        representatives[f"{input_sfam_name} Representatives"] = _elbo_scores.numpy()
                    elif isinstance(other_representatives, dict) and sfam_n in other_representatives:
                        representatives[f"{other_representatives[sfam_n]} Representatives"] = _elbo_scores.numpy()
        #torch.save(elbo_scores, background_file)
        return elbo_scores.numpy(), representatives

def run_nn(path, prefix, cathcodes=None, domains=None, h5_dir="/home/ed4bu/deepurfold-paper-1.h5"):
    cmd = "python -m DeepUrfold.Evaluators.EvaluateDistrubutedDomainStructureVAE "
    cmd += f"--data_dir={h5_dir} --gpus=1 --batch_size 16 "
    cmd += f"--num_workers 32 --raw elbo --batchwise_loss False --checkpoint {os.path.join(path, 'last.ckpt')} "
    cmd += f"--prefix \"{prefix}\" "
    cmd += "--features 'H;HD;HS;C;A;N;NA;NS;OA;OS;SA;S;Unk_atom__is_helix;is_sheet;Unk_SS__residue_buried__is_hydrophobic__pos_charge__is_electronegative' "
    cmd += "--feature_groups 'Atom Type;Secondary Structure;Solvent Accessibility;Hydrophobicity;Charge;Electrostatics' "
    if cathcodes is not None:
        cmd += f"--superfamily {cathcodes} "
    if domains is not None:
        cmd += f"--domains {' '.join(domains.tolist())}"

    print(f"Running {cmd}")
    subprocess.call(cmd, cwd=path, shell=True)

    prefix += "-elbo-elbo.pt"
    prefix = os.path.join(path, prefix)
    assert os.path.isfile(prefix), "Failed running model"
    elbo_scores = torch.load(prefix)

    return elbo_scores

def create_figure2():
    fig, ax = plt.subplots(1,3, sharey=True, sharex=True,figsize=(8,3))

    #https://carto.com/carto-colors/ Safe
    colors = [
        '#88CCEE',
        '#CC6677',
        '#DDCC77',
        '#117733',
        '#332288',
        '#AA4499',
        '#44AA99',
        '#999933',
        '#882255',
        '#661100',
        '#6699CC',
        '#888888'
    ]

    colors = {
        "SH3 Rep": "#BE1E2D", #Dark Red
        "OB Rep": "#2880B2", #Dark Blue
        "SH3 Per": "#DD7C83", #Light Red
        "OB Per": "#4CBFE8", #Light Blue
        "Background": "#A09747", #Gold/yellow
        "SH3 wild": "#BE1E2D", #Dark Red
        "OB wild": "#2880B2", #Dark Blue
        "uL2 (OB": "#2DB29C", #Cyan  (OB)
        "uL24 (SH3": "#D57928", #Orange  (SH3)
    }

    groups = {
        ""
    }

    image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

    legend = {}

    #SH3 Model
    sh3_dir = "/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/multiple_loop_permutations/sh3_3/nn_model"
    sh3_input_sfam = {"SH3":"2.30.30.100"}
    sh3_other_elbo_scores= {
        "SH3 Permutatants": os.path.join(sh3_dir, "ELBO (MODEL=2.30.30.100, domain=1kq2A00_MLP)-elbo-elbo.pt"),
    }
    sh3_other_single_elbo_scores = os.path.join(sh3_dir, "single_elbo_scores.csv")
    sh3_legend = create_kde_figure(sh3_input_sfam,
        prefix= "A",
        get_representatives=True,
        other_representatives={"2.40.50.140":"OB"},
        other_elbo_scores=sh3_other_elbo_scores,
        other_single_elbo_scores=sh3_other_single_elbo_scores,
        data_directory=sh3_dir,
        ax=ax[0],
        legend=False,
        colors=colors,
        corner_image=os.path.join(image_path, "1kq2A00_vs_MLP.png")
    )
    legend.update(sh3_legend)

    #OB Model
    ob_dir = "/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/multiple_loop_permutations/ob_2"
    ob_input_sfam = {"OB":"2.40.50.140"}
    ob_other_elbo_scores = {
        "OB Permutatants": os.path.join(ob_dir, "ELBO (MODEL=2.40.50.140, domain=1uebA03_MLP)-elbo-elbo.pt"),
    }
    ob_other_single_elbo_scores = os.path.join(ob_dir, "other_elbo_scores.csv")
    ob_legend = create_kde_figure(ob_input_sfam,
        prefix= "B",
        get_representatives=True,
        other_representatives={"2.30.30.100":"SH3"},
        other_elbo_scores=ob_other_elbo_scores,
        other_single_elbo_scores=ob_other_single_elbo_scores,
        data_directory=ob_dir,
        ax=ax[1],
        legend=False,
        colors=colors,
        corner_image=os.path.join(image_path, "1uebA03_vs_MLP.png"),
        corner_image_zoom=0.06
    )
    legend.update(ob_legend)

    #Joint SH3 and OB Model Undersampled
    under_joint_dir = "/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/architecture_training/undersample"
    under_joint_input_sfam = {"SH3":"2.30.30.100", "OB":"2.40.50.140"}
    under_joint_other_elbo_scores = {
        "SH3 Representatives": os.path.join(under_joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140_undersample, INPUT=2.30.30.100)-elbo-elbo.pt"),
        "OB Representatives": os.path.join(under_joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140_undersample, INPUT=2.40.50.140)-elbo-elbo.pt"),
        "SH3 Permutatants": "/home/ed4bu/sh3_mlp.h5",
        "OB Permutatants": "/home/ed4bu/ob_mlp.h5",
        #"SH3 Permutatants\nfrom 1kq2A00)": os.path.join(joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140_undersample, INPUT=SH3 Representatives)-elbo-elbo.pt"),
        #"OB Permutatants\nfrom 1uebA03": os.path.join(joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140_undersample, INPUT=OB Multiple Loop Permutations)-elbo-elbo.pt")
    }

    under_joint_other_single_elbo_scores = []
    for name, torch_file in [
      ("1kq2A00 SH3 wild-type", "ELBO (MODEL=2.30.30.100+2.40.50.140_undersample, INPUT=1kq2A00_SH3)-elbo-elbo.pt"),
      ("1uebA03 OB wild-type", "ELBO (MODEL=2.30.30.100+2.40.50.140_undersample, INPUT=1uebA03_OB)-elbo-elbo.pt"),
      (["uL2 (OB ancestral reconstruction)", "uL24 (SH3 ancestral reconstruction)"], "ELBO (MODEL=2.30.30.100+2.40.50.140_undersample, INPUT=AncestralSBB)-elbo-elbo.pt")
    ]:
        if not isinstance(name, (list, tuple)):
            name = [name]
        values = torch.load(os.path.join(under_joint_dir, torch_file)).tolist()
        assert len(values)==len(name)
        under_joint_other_single_elbo_scores += list(zip(name, values))
    under_joint_other_single_elbo_scores = pd.DataFrame(under_joint_other_single_elbo_scores)

    under_joint_legend = create_kde_figure(under_joint_input_sfam,
        prefix= "C",
        other_elbo_scores=under_joint_other_elbo_scores,
        other_single_elbo_scores=under_joint_other_single_elbo_scores,
        data_directory=under_joint_dir,
        regular_model=False,
        ax=ax[2],
        legend=False,
        colors=colors
    )
    legend.update(under_joint_legend)

    distribution_legend_items = []
    single_legend_items = []
    for i, (l, h) in enumerate(legend.items()):
        if "wild" in l or "uL" in l:
            single_legend_items.append((l,h))
        else:
            distribution_legend_items.append((l,h))

    distribution_legend_items = list(sorted(distribution_legend_items,
        key=lambda x:x[0], reverse=True))

    single_legend_items = [
        *single_legend_items[:2],
        single_legend_items[3],
        single_legend_items[2]
    ]

    h, l = zip(*[(h, l) for k in colors.keys() for l, h in distribution_legend_items if k in l])
    distribution_legend = ax[0].legend(h, l, fontsize=6, bbox_to_anchor=(1.3, -.15), #loc='upper right', #bbox_to_anchor=(0.5, -0.25),
        fancybox=True, frameon=False, title="$\\textbf{Domain groups ($\Rightarrow$ distributions)}$", title_fontsize=6)
    distribution_legend._legend_box.align = "left"
    distribution_legend.get_frame().set_facecolor('none')

    h, l = zip(*[(h, l) for k in colors.keys() for l, h in single_legend_items if k in l])
    single_legend = ax[2].legend(h, l, fontsize=6, bbox_to_anchor=(0.6, -.15), #loc='upper right', #bbox_to_anchor=(0.5, -0.25),
        fancybox=True, frameon=False, title="$\\textbf{Single-domain inference calculations ($\Rightarrow$ single tick marks)}$", title_fontsize=6)
    single_legend._legend_box.align = "left"
    single_legend.get_frame().set_facecolor('none')

    ax[0].add_artist(distribution_legend)

    # h, l = zip(*[(h, l) for k in colors.keys() for l, h in legend.items() if k in l])

    # _legend = fig.legend(h, l, fontsize=10, loc='outside lower left', #bbox_to_anchor=(0, 0), #loc='upper right', #bbox_to_anchor=(0.5, -0.25),
    #     fancybox=True, frameon=False) #len(legend))
    # _legend.get_frame().set_facecolor('none')


    plt.subplots_adjust(wspace=0.05)

    plt.savefig("Figure2.png", bbox_inches='tight', dpi=600)
    plt.savefig("Figure2.pdf", bbox_inches='tight', dpi=600) #, 
    plt.savefig("Figure2.ps", bbox_inches='tight', dpi=600, transparent=False) #Colors added

    w, h = ax[0].get_xlim()

    plt.clf()

    supp_fig, supp_ax = plt.subplots(1,3,sharey=True, sharex=True,figsize=(8,3))
    supp_ax[0].set_xlim(w,h)
    supp_ax[1].set_xlim(w,h)
    supp_ax[2].set_xlim(w,h)

    #Joint SH3 and OB Model Imbalanced
    joint_dir = "/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/architecture_training/sfams_for_paper"
    joint_input_sfam = {"SH3":"2.30.30.100", "OB":"2.40.50.140"}
    joint_other_elbo_scores = {
        "SH3 Representatives": os.path.join(joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140, INPUT=2.30.30.100)-elbo-elbo.pt"),
        "OB Representatives": os.path.join(joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140, INPUT=2.40.50.140)-elbo-elbo.pt"),
        "SH3 Permutatants": os.path.join(joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140, INPUT=SH3 Representatives)-elbo-elbo.pt"),
        "OB Permutatants": os.path.join(joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140, INPUT=OB Multiple Loop Permutations)-elbo-elbo.pt")
    }
    joint_other_single_elbo_scores = os.path.join(joint_dir, "single_elbo_scores.csv")
    create_kde_figure(joint_input_sfam,
        prefix="A",
        title="Imbalanced",
        other_elbo_scores=joint_other_elbo_scores,
        other_single_elbo_scores=joint_other_single_elbo_scores,
        data_directory=joint_dir,
        regular_model=False,
        ax=supp_ax[0],
        legend=False,
        colors=colors
    )

    #Joint SH3 and OB Model Oversampled
    over_joint_dir = "/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/architecture_training/oversampled"
    over_joint_input_sfam = {"SH3":"2.30.30.100", "OB":"2.40.50.140"}
    over_joint_other_elbo_scores = {
        "SH3 Representatives": os.path.join(over_joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140_oversample, INPUT=2.30.30.100)-elbo-elbo.pt"),
        "OB Representatives": os.path.join(over_joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140_oversample, INPUT=2.40.50.140)-elbo-elbo.pt"),
        "SH3 Permutatants": "/home/ed4bu/sh3_mlp.h5",
        "OB Permutatants": "/home/ed4bu/ob_mlp.h5",
        #"SH3 Permutatants": os.path.join(over_joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140_oversample, INPUT=SH3 Representatives)-elbo-elbo.pt"),
        #"OB Permutatants": os.path.join(over_joint_dir, "ELBO (MODEL=2.30.30.100+2.40.50.140_oversample, INPUT=OB Multiple Loop Permutations)-elbo-elbo.pt")
    }

    over_joint_other_single_elbo_scores = []
    for name, torch_file in [
      ("1kq2A00 SH3 wild-type", "ELBO (MODEL=2.30.30.100+2.40.50.140_oversample, INPUT=1kq2A00_SH3)-elbo-elbo.pt"),
      ("1uebA03 OB wild-type", "ELBO (MODEL=2.30.30.100+2.40.50.140_oversample, INPUT=1uebA03_OB)-elbo-elbo.pt"),
      (["uL2 (OB ancestral reconstruction)", "uL24 (SH3 ancestral reconstruction)"], "ELBO (MODEL=2.30.30.100+2.40.50.140_oversample, INPUT=AncestralSBB)-elbo-elbo.pt")
    ]:
        if not isinstance(name, (list, tuple)):
            name = [name]
        values = torch.load(os.path.join(over_joint_dir, torch_file)).tolist()
        assert len(values)==len(name)
        over_joint_other_single_elbo_scores += list(zip(name, values))
    over_joint_other_single_elbo_scores = pd.DataFrame(over_joint_other_single_elbo_scores)

    create_kde_figure(over_joint_input_sfam,
        prefix="B",
        title="Oversampled",
        other_elbo_scores=over_joint_other_elbo_scores,
        other_single_elbo_scores=over_joint_other_single_elbo_scores,
        data_directory=over_joint_dir,
        regular_model=False,
        ax=supp_ax[1],
        legend=False,
        colors=colors
    )

    create_kde_figure(under_joint_input_sfam,
        prefix="C",
        title="Undersampled",
        other_elbo_scores=under_joint_other_elbo_scores,
        other_single_elbo_scores=under_joint_other_single_elbo_scores,
        data_directory=under_joint_dir,
        regular_model=False,
        ax=supp_ax[2],
        legend=False,
        colors=colors
    )

    h, l = zip(*[(h, l) for k in colors.keys() for l, h in distribution_legend_items if k in l])
    distribution_legend = supp_ax[0].legend(h, l, fontsize=6, bbox_to_anchor=(1.3, -.15), #loc='upper right', #bbox_to_anchor=(0.5, -0.25),
        fancybox=True, frameon=False, title="$\\textbf{Domain groups ($\Rightarrow$ distributions)}$", title_fontsize=6)
    distribution_legend._legend_box.align = "left"
    distribution_legend.get_frame().set_facecolor('none')

    h, l = zip(*[(h, l) for k in colors.keys() for l, h in single_legend_items if k in l])
    single_legend = supp_ax[2].legend(h, l, fontsize=6, bbox_to_anchor=(0.6, -.15), #loc='upper right', #bbox_to_anchor=(0.5, -0.25),
        fancybox=True, frameon=False, title="$\\textbf{Single-domain inference calculations ($\Rightarrow$ single tick marks)}$", title_fontsize=6)
    single_legend._legend_box.align = "left"
    single_legend.get_frame().set_facecolor('none')

    supp_ax[0].add_artist(distribution_legend)

    plt.subplots_adjust(wspace=0.05)

    plt.savefig("Figure2_supp.png", bbox_inches='tight', dpi=600)
    plt.savefig("Figure2_supp.pdf", bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    create_figure2()
