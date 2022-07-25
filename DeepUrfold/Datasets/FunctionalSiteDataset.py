import os, sys
import re
from glob import glob, iglob

import numpy as np
from Bio import SeqIO

from molmimic.generate_data import data_stores
from molmimic.util.pdb import get_pdb_residues
from molmimic.common.ProteinTables import three_to_one
from DeepUrfold.Datasets.DomainStructureDataset import DomainStructureDataset, merge_superfamilies, get_superfamilies, download_file
from molmimic.util.toil import map_job

from toil.realtimeLogger import RealtimeLogger

import pandas as pd
from pandarallel import pandarallel

class FunctionalSiteDataset(DomainStructureDataset):
    def __init__(self, data_file, annotated_site_col_name, data_key="table", use_features=None, split_level="H", volume=256, nClasses=1):
        super().__init__(data_file, data_key=data_key, use_features=use_features,
            split_level=split_level, use_domain_index=True,
            structure_key="structure_file", feature_key="feature_file",
            truth_key=annotated_site_col_name, volume=volume, nClasses=nClasses)

    
