# author: "Yozu_Roo"
# date: 2024/12/10 9:44
import argparse
import pickle
import dgl
import rdkit
import torch
import os
import logging
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from functools import total_ordering
from typing import List, Set
from pathlib import Path
from rdkit import RDLogger
from tqdm.auto import tqdm

from model.pgmg import PGMG
from utils.file_utils import load_phar_file
from utils.utils import seed_torch
from utils.build_scoring_function import build_scoring_function
from utils.dataset import Tokenizer, SemiSmilesDataset
from utils.smiles2ppgraph import smiles2ppgraph
from torch.utils.data import DataLoader
from train_chembl_baseline import CFG
from torch.optim import AdamW
RDLogger.DisableLog('rdApp.*')


def load_model(model_path, tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    model_params = {
        "max_len": 128,
        "pp_v_dim": 7 + 1,
        "pp_e_dim": 1,
        "pp_encoder_n_layer": 4,
        "hidden_dim": 384,
        "n_layers": 8,
        "ff_dim": 1024,
        "n_head": 8,
    }

    model = PGMG(model_params, tokenizer)
    states = torch.load(model_path, map_location='cpu')
    print(model.load_state_dict(states['model'], strict=False))

    return model, tokenizer


def format_smiles(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    smiles = rdkit.Chem.MolToSmiles(mol, isomericSmiles=True)
    return smiles


def get_ppgraph(smiles_list):
    pp_graphs = []
    for smile in smiles_list:
        try:
            pp_graph, mapping = smiles2ppgraph(smile)
            pp_graph.ndata['h'] = \
                torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
            pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
            pp_graphs.append(pp_graph)
        except:
            print(smile)
            continue
    return pp_graphs


def generate(init_mols, model, tokenizer, args):

    results = list(init_mols)
    final_results = []

    model.eval()
    model.to(args.device)

    while len(final_results) < args.n_mol:
        # sample
        g = get_ppgraph(results)
        res = []
        for i in tqdm(range(len(g) // args.batch_size + 1)):
            start_idx = i * args.batch_size
            end_idx = (i + 1) * args.batch_size
            g_batch = g[start_idx:end_idx]
            g_batch = dgl.batch(g_batch).to(args.device)
            res.extend(tokenizer.get_text(model.generate(g_batch, random_sample=True)))
        final_results.extend(res)
        print(f'generated mols:{len(final_results)}')


    # save final sample
    count = 0
    with open(os.path.join(args.output_dir, args.save_file), 'w') as handle:
        for d in final_results:
            if count <= args.n_mol:
                handle.write(f'{d}\n')
                count += 1
            else:
                break



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_name', nargs=2, help='the names of two targets', default=['ROR_gamma', 'DHODH'])
    parser.add_argument('--init_smi_path', type=str, required=True, help='Path to the input SMILES file used for molecule generation.')
    parser.add_argument('--output_dir', type=Path, help='the output directory', default='./generate_output_RD')
    parser.add_argument('--save_file', type=Path, help='the save file', default='RD_gen.csv')
    parser.add_argument('--model_path', type=Path, help='the weights file (xxx.pth)',
                        default='./finetune_output_RD/epoch_15_finetuned_model.pth')
    parser.add_argument('--tokenizer_path', type=Path, help='the saved tokenizer (tokenizer.pkl)',
                        default='./pretrain_output/tokenizer.pkl')
    parser.add_argument('--n_mol', type=int, default=10000, help='number of generated molecules for each '
                                                                 'pharmacophore file')
    parser.add_argument('--device', type=str, default='cuda', help='`cpu` or `cuda`')
    parser.add_argument('--filter', action='store_true', help='whether to save only the unique valid molecules')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.seed != -1:
        seed_torch(args.seed)

    args.output_dir.mkdir(parents=False, exist_ok=True)

    model, tokenizer = load_model(args.model_path, args.tokenizer_path)
    init_smi = pd.read_csv(args.init_smi_path)['SMILES'].tolist()

    generate(init_smi, model, tokenizer, args)


    print('done')
