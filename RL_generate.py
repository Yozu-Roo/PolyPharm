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
import csv
from functools import total_ordering
from typing import List, Set
from pathlib import Path
from rdkit import RDLogger
from rdkit import Chem
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import AllChem
import statistics
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


from model.pgmg import PGMG
from utils.file_utils import load_phar_file
from utils.utils import seed_torch
from utils.build_scoring_function import build_scoring_function
from utils.dataset import Tokenizer, SemiSmilesDataset
from utils.smiles2ppgraph import get_batched_pharmacophore_types, smiles2ppgraph
from utils.utils import stratified_sampling
from torch.utils.data import DataLoader
from train_chembl_baseline import CFG
from torch.optim import AdamW
RDLogger.DisableLog('rdApp.*')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
PP_TYPE_WEIGHT = [1.4891304347826086, 1.0, 8.058823529411764, 1.0378787878787878, 1.8026315789473686, 2.174603174603175,
                  17.125]
@total_ordering
class OptResult:
    def __init__(self, smiles: str, score: float, p_a: float, p_b: float) -> None:
        self.smiles = smiles
        self.p_a = p_a
        self.p_b = p_b
        self.score = score

    @property
    def mean(self):
        return (self.p_a + self.p_b) / 2

    @property
    def abs_diff(self):
        return abs(self.p_a - self.p_b)

    def __eq__(self, other):
        if not isinstance(other, OptResult):
            return NotImplemented
        return self.smiles == other.smiles

    def __lt__(self, other):
        if not isinstance(other, OptResult):
            return NotImplemented
        return (-self.mean, self.abs_diff) < (-other.mean, other.abs_diff)


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


def scoring_function(scoring_definition):
    score_fn_params = {
        'scoring_definition': os.path.join(scoring_definition,'scoring_definition.csv'),
        'encoding': './data/Encoding_CpG_Enhancers',
        'cell_line_model': None,
        'fscores': './data/fpscores.pkl.gz',
        'opti': 'gauss'
    }
    scoring_function = build_scoring_function(
        score_fn_params['scoring_definition'],
        score_fn_params['encoding'],
        score_fn_params['cell_line_model'],
        score_fn_params['fscores'],
        opti=score_fn_params['opti'],
        return_individual=False
    )
    return scoring_function

def check_ppgraph(smile):
    mol = Chem.MolFromSmiles(smile)
    rsmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False, doRandom=True)
    try:
        pp_graph, mapping = smiles2ppgraph(rsmiles)
    except Exception as e:
        return False
    return True

def get_ppgraph(smiles_list):
    pp_graphs = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        rsmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False, doRandom=True)
        try:
            pp_graph, mapping = smiles2ppgraph(rsmiles)
            pp_graph.ndata['h'] = pp_graph.ndata['h'].float()
            pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()
            pp_graphs.append(pp_graph)
        except Exception as e:
            print(f"pp_graphs error:{rsmiles}")
            continue
    return pp_graphs


def finetune(inint_population, model, tokenizer, objective, args):
    results: List[OptResult] = []
    results_fp = []
    seen: Set[str] = set()
    candidate_set: List[str] = []
    elite_set: List[str] = []

    offset = 0.05
    threshold = args.threshold

    optimizer = AdamW(model.parameters(), lr=CFG.init_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
    model.to(args.device)

    for epoch in range(0, 1 + args.n_epochs):
        if epoch == 0:
            canonicalized_samples = set(inint_population)
        else:
            # sample
            model.eval()
            model.to(args.device)
            with torch.no_grad():
                g = get_ppgraph(candidate_set) * (args.n_mol // len(candidate_set) + 2)
                res = []
                for i in tqdm(range(len(g) // args.batch_size + 1)):
                    start_idx = i * args.batch_size
                    end_idx = (i + 1) * args.batch_size
                    g_batch = g[start_idx:end_idx]
                    g_batch = dgl.batch(g_batch).to(args.device)
                    res.extend(tokenizer.get_text(model.generate(g_batch, random_sample=True)))
                canonicalized_samples = set([i for i in res if format_smiles(i) != None])

        # new molecules seen
        payload = list(canonicalized_samples.difference(seen))
        # add the new stuff to tracker
        seen.update(payload)
    
        scores, p_a, p_b = objective.score_list(payload)
        
        int_results = [OptResult(smiles=smiles, score=score, p_a=p_a_i, p_b=p_b_i) for smiles, score, p_a_i, p_b_i in zip(payload, scores, p_a, p_b) if check_ppgraph(smiles)]
        int_results = sorted(int_results)
        
        # update threshold
        if epoch > 0 and epoch % 5 == 0:
            threshold += offset
        
        # store the molecules
        int_candidate_set = int_results[:int(len(int_results) * args.keep_top_ratio)]
        candidate_set += [s.smiles for s in int_candidate_set]
        elite_set += [s.smiles for s in int_candidate_set if s.score >= threshold]
               
        np.random.shuffle(elite_set)
        print(f"elite size: {len(elite_set)}")

        # run training
        if args.optimize_n_epochs > 0:
            train_dataset = SemiSmilesDataset(elite_set, tokenizer, use_random_input_smiles=True,
                                              use_random_target_smiles=True, include_atoms=True)
            train_loader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=CFG.num_workers,
                                      pin_memory=True,
                                      drop_last=False,
                                      collate_fn=train_dataset.collate_fn)
            model.train()
            for _ in range(args.optimize_n_epochs):
                for step, batch_data in tqdm(enumerate(train_loader)):
                    inputs, input_mask, pp_graphs, mappings, targets, *others = [i.to(args.device) for i in batch_data]
                    prediction_scores, mapping_scores, lm_loss, kl_loss, cl_loss = model(inputs, input_mask, pp_graphs,
                                                                                         targets, flag="finetune")

                    x = torch.zeros(inputs.shape[0], 8, len(PP_TYPE_WEIGHT)).to(args.device)
                    xx = get_batched_pharmacophore_types(pp_graphs)
                    x[:, :xx.shape[1], :] = xx

                    a = torch.Tensor(PP_TYPE_WEIGHT).to(args.device)
                    sample_weight = x @ a  # (512, MAX_NUM_PP_GRAPHS)

                    mapping_loss_weight = (mappings == 1) * (8 / (0.001 + (mappings == 1).sum(1))).unsqueeze(
                        1)  # balance pos/neg samples
                    mapping_loss_weight += (mappings != -100) * sample_weight.unsqueeze(
                        1)  # balance rare pharmacophore types

                    mask = mappings != -100
                    mapping_loss = F.binary_cross_entropy(mapping_scores[mask], mappings[mask],
                                                          weight=mapping_loss_weight[mask])

                    # print(f'lm_loss:{lm_loss} mapping_loss:{mapping_loss}')
                    loss = 0.5 * mapping_loss + 0.5 * lm_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                scheduler.step()

            if epoch > args.n_epochs // 2:
                torch.save({'model': model.state_dict()}, args.output_dir / f'epoch_{epoch}_finetuned_model.pth')

        # write out payload
        if args.save_payloads:
            with open(os.path.join(args.output_dir, f'GDM_{epoch}_payload.txt'), 'w') as handle:
                for smiles, score in sorted(zip(payload, scores), key=lambda x: x[1], reverse=True):
                    handle.write(f'{smiles},{score}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_name', nargs=2, help='the names of two targets', default=['ROR_gamma', 'DHODH'])
    parser.add_argument('--data_path', type=Path, required=True, help='Path to the fine-tuning dataset CSV')
    parser.add_argument('--output_dir', type=Path, help='the output directory', default='./finetune_output_RD')
    parser.add_argument('--model_path', type=Path, help='the weights file (xxx.pth)', default='./pretrain_output-v2/fold0_epoch32.pth')
    parser.add_argument('--tokenizer_path', type=Path, help='the saved tokenizer (tokenizer.pkl)', default='./pretrain_output-v2/tokenizer.pkl')

    parser.add_argument('--n_mol', type=int, default=30000, help='number of generated molecules for each pharmacophore file')
    parser.add_argument('--device', type=str, default='cuda', help='`cpu` or `cuda`')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--threshold', type=float, default=0.7, help='Threshold for filter')
    parser.add_argument('--n_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--optimize_n_epochs', type=int, default=5, help='Number of epochs for optimization')
    parser.add_argument('--save_frequency', type=int, default=10, help='Frequency of saving models')
    parser.add_argument('--save_payloads', action='store_true', default=True, help='Whether to save payloads')
    parser.add_argument('--keep_top_ratio', type=float, default=0.5, help='Keep the top N molecules')
    args = parser.parse_args()

    if args.seed != -1:
        seed_torch(args.seed)

    args.output_dir.mkdir(parents=False, exist_ok=True)

    model, tokenizer = load_model(args.model_path, args.tokenizer_path)
    objective = scoring_function(os.path.join('./data', '+'.join(map(str, args.target_name))))
    finetune_smi = (
        pd.read_csv(args.data_path)['SMILES']
        .dropna()
        .astype(str)
        .tolist()
    )

    finetune(finetune_smi, model, tokenizer, objective, args)
    print('done')

