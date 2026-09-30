import os
import random

import dgl
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from torch.nn.utils.rnn import pad_sequence

MAX_NUM_PP_GRAPHS = 8
EDGE_TYPE_BOND = 0
EDGE_TYPE_PHARMACOPHORE = 1
EDGE_TYPE_DEPENDENCY = 2


def get_batched_pharmacophore_types(graph):
    node_counts = graph.batch_num_nodes().tolist()
    type_chunks = torch.split(graph.ndata['type'], node_counts)
    if 'is_pharmacophore' not in graph.ndata:
        return pad_sequence(type_chunks, batch_first=True)

    mask_chunks = torch.split(graph.ndata['is_pharmacophore'].bool(), node_counts)
    phar_types = [types[mask] for types, mask in zip(type_chunks, mask_chunks)]
    return pad_sequence(phar_types, batch_first=True)


def _bond_weight(bond_type):
    return {'SINGLE': 1.0, 'DOUBLE': 0.87, 'AROMATIC': 0.91}.get(bond_type, 0.78)


def sample_probability(elment_array, plist, N):
    Psample = []
    n = len(plist)
    index = int(random.random() * n)
    mw = max(plist)
    beta = 0.0
    for i in range(N):
        beta = beta + random.random() * 2.0 * mw
        while beta > plist[index]:
            beta = beta - plist[index]
            index = (index + 1) % n
        Psample.append(elment_array[index])

    return Psample


def six_encoding(atom):
    # actually seven
    orgin_phco = [0, 0, 0, 0, 0, 0, 0, 0]
    for j in atom:
        orgin_phco[j] = 1
    return torch.HalfTensor(orgin_phco[1:])


def cal_dist(mol, start_atom, end_tom):
    list_ = []
    list_.append(start_atom)
    seen = set()
    seen.add(start_atom)
    parent = {start_atom: None}
    nei_atom = []
    bond_num = mol.GetNumBonds()
    while (len(list_) > 0):
        vertex = (list_[0])
        del (list_[0])
        nei_atom = ([n.GetIdx() for n in mol.GetAtomWithIdx(vertex).GetNeighbors()])
        for w in nei_atom:
            if w not in seen:
                list_.append(w)
                seen.add(w)
                parent[w] = vertex
    path_atom = []
    while end_tom != None:
        path_atom.append(end_tom)
        end_tom = parent[end_tom]
    nei_bond = []
    for i in range(bond_num):
        nei_bond.append((mol.GetBondWithIdx(i).GetBondType().name, mol.GetBondWithIdx(i).GetBeginAtomIdx(),
                         mol.GetBondWithIdx(i).GetEndAtomIdx()))
    bond_collection = []
    for idx in range(len(path_atom) - 1):
        bond_start = path_atom[idx]
        bond_end = path_atom[idx + 1]
        for bond_type in nei_bond:
            if len(list(set([bond_type[1], bond_type[2]]).intersection(set([bond_start, bond_end])))) == 2:
                bond_ = bond_type[0]
                if [bond_, bond_type[1], bond_type[2]] not in bond_collection:
                    bond_collection.append([bond_, bond_type[1], bond_type[2]])
    dist = 0
    for elment in bond_collection:
        dist += _bond_weight(elment[0])
    return dist


def smiles_code_(smiles, g, e_list):
    smiles = smiles
    dgl = g
    e_elment = e_list
    mol = Chem.MolFromSmiles(smiles)
    atom_num = mol.GetNumAtoms()
    atom_index_list = []
    smiles_code = np.zeros((atom_num, MAX_NUM_PP_GRAPHS))
    for elment_i in range(len(e_elment)):  ##定位这个元素在第几个药效团
        elment = e_elment[elment_i]
        for e_i in range(len(elment)):
            e_index = elment[e_i]
            for atom in mol.GetAtoms():  ##定位这个原子在分子中的索引
                if e_index == atom.GetIdx():
                    list_ = ((dgl.ndata['type'])[elment_i]).tolist()
                    for list_i in range(len(list_)):
                        if list_[list_i] == 1:
                            smiles_code[atom.GetIdx(), elment_i] = 1.0
    return smiles_code


def smiles2ppgraph(smiles: str, include_atoms=False):
    '''
    :param smiles: a molecule
    :return: (pp_graph, mapping)
        pp_graph: DGLGraph containing pharmacophore nodes. If include_atoms is True,
            molecular atoms are appended and connected to pharmacophores for GCN propagation.
        mapping: np.Array ((atom_num, MAX_NUM_PP_GRAPHS)) mapping atoms to pharmacophore node columns
    '''

    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    atom_index_list = []
    pharmocophore_all = []

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)
    feature_mapping = {'Aromatic': 1, 'Hydrophobe': 2, 'PosIonizable': 3,
                       'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6}
    atom_features = torch.zeros((mol.GetNumAtoms(), 7), dtype=torch.float32) if include_atoms else None
    if include_atoms:
        element_features = {6: 0, 7: 1, 8: 2, 16: 3, 15: 4, 9: 5, 17: 5, 35: 5, 53: 5}
        for atom in mol.GetAtoms():
            feature_index = element_features.get(atom.GetAtomicNum(), 6)
            atom_features[atom.GetIdx(), feature_index] = -1

    # only keep 2 phar
    # feats_t = list(feats)
    # random.shuffle(feats_t)
    # feats = tuple(feats_t[:3])

    for f in feats:
        phar = f.GetFamily()
        atom_index = f.GetAtomIds()
        atom_index = tuple(sorted(atom_index))
        atom_type = f.GetType()
        phar_index = feature_mapping.setdefault(phar, 7)
        pharmocophore_ = [phar_index, atom_index]  # some pharmacophore feature
        pharmocophore_all.append(pharmocophore_)  # all pharmacophore features within a molecule
        atom_index_list.append(atom_index)  # atom indices of one pharmacophore feature
    random.shuffle(pharmocophore_all)
    num = [3, 4, 5, 6, 7]
    num_p = [0.086, 0.0864, 0.389, 0.495, 0.0273]  # P(Number of Pharmacophore points)
    num_ = sample_probability(num, num_p, 1)

    type_list = []
    size_ = []

    ## The randomly generated clusters are obtained,
    # and the next step is to perform a preliminary merging of these randomly generated clusters with identical elements
    if len(pharmocophore_all) >= int(num_[0]):
        mol_phco = pharmocophore_all[:int(num_[0])]
    else:
        mol_phco = pharmocophore_all

    for pharmocophore_all_i in range(len(mol_phco)):
        for pharmocophore_all_j in range(len(mol_phco)):
            if mol_phco[pharmocophore_all_i][1] == mol_phco[pharmocophore_all_j][1] \
                    and mol_phco[pharmocophore_all_i][0] != mol_phco[pharmocophore_all_j][0]:
                index_ = [min(mol_phco[pharmocophore_all_i][0], mol_phco[pharmocophore_all_j][0]),
                          max(mol_phco[pharmocophore_all_i][0], mol_phco[pharmocophore_all_j][0])]
                mol_phco[pharmocophore_all_j] = [index_, mol_phco[pharmocophore_all_i][1]]
                mol_phco[pharmocophore_all_i] = [index_, mol_phco[pharmocophore_all_i][1]]
            else:
                index_ = mol_phco[pharmocophore_all_i][0]
    unique_index_filter = []
    unique_index = []
    for mol_phco_candidate_single in mol_phco:
        if mol_phco_candidate_single not in unique_index:
            if type(mol_phco[0]) == list:
                unique_index.append(mol_phco_candidate_single)
            else:
                unique_index.append([[mol_phco_candidate_single[0]], mol_phco_candidate_single[1]])
    for unique_index_single in unique_index:
        if unique_index_single not in unique_index_filter:
            unique_index_filter.append(unique_index_single)  ## The following is the order of the pharmacophores by atomic number
    sort_index_list = []
    for unique_index_filter_i in unique_index_filter:  ## Collect the mean of the participating elements
        sort_index = sum(unique_index_filter_i[1]) / len(unique_index_filter_i[1])
        sort_index_list.append(sort_index)
    sorted_id = sorted(range(len(sort_index_list)), key=lambda k: sort_index_list[k])
    unique_index_filter_sort = []
    for index_id in sorted_id:
        unique_index_filter_sort.append(unique_index_filter[index_id])
    position_matrix = np.zeros((len(unique_index_filter_sort), len(unique_index_filter_sort)))
    e_list = []
    for mol_phco_i in range(len(unique_index_filter_sort)):
        mol_phco_i_elment = list(unique_index_filter_sort[mol_phco_i][1])
        if type(unique_index_filter_sort[mol_phco_i][0]) == list:
            type_list.append(six_encoding(unique_index_filter_sort[mol_phco_i][0]))
        else:
            type_list.append(six_encoding([unique_index_filter_sort[mol_phco_i][0]]))

        size_.append(len(mol_phco_i_elment))
        e_list.append(mol_phco_i_elment)
        for mol_phco_j in range(len(unique_index_filter_sort)):
            mol_phco_j_elment = list(unique_index_filter_sort[mol_phco_j][1])
            if mol_phco_i_elment == mol_phco_j_elment:
                position_matrix[mol_phco_i, mol_phco_j] = 0
            elif str(set(mol_phco_i_elment).intersection(set(mol_phco_j_elment))) == 'set()':
                dist_set = []
                for atom_i in mol_phco_i_elment:
                    for atom_j in mol_phco_j_elment:
                        dist = cal_dist(mol, atom_i, atom_j)
                        dist_set.append(dist)
                min_dist = min(dist_set)
                if max(len(mol_phco_i_elment), len(mol_phco_j_elment)) == 1:
                    position_matrix[mol_phco_i, mol_phco_j] = min_dist
                else:
                    position_matrix[mol_phco_i, mol_phco_j] = min_dist + max(len(mol_phco_i_elment),
                                                                             len(mol_phco_j_elment)) * 0.2
            else:
                for type_elment_i in mol_phco_i_elment:
                    for type_elment_j in mol_phco_j_elment:
                        if type_elment_i == type_elment_j:
                            position_matrix[mol_phco_i, mol_phco_j] = max(len(mol_phco_i_elment),
                                                                          len(mol_phco_j_elment)) * 0.2
                        ##The above is a summary of the cases where the two pharmacophores have direct elemental intersection.
    weights = []
    u_list = []
    v_list = []
    phco_single = []

    for u in range(position_matrix.shape[0]):
        for v in range(position_matrix.shape[1]):
            if u != v:
                u_list.append(u)
                v_list.append(v)
                if position_matrix[u, v] >= position_matrix[v, u]:
                    weights.append(position_matrix[v, u])
                else:
                    weights.append(position_matrix[u, v])
    pharmacophore_count = len(type_list)
    atom_count = mol.GetNumAtoms()
    src = list(u_list)
    dst = list(v_list)
    edge_weights = list(weights)
    edge_types = [EDGE_TYPE_PHARMACOPHORE] * len(weights)

    if include_atoms:
        for bond in mol.GetBonds():
            atom_start = pharmacophore_count + bond.GetBeginAtomIdx()
            atom_end = pharmacophore_count + bond.GetEndAtomIdx()
            bond_weight = _bond_weight(bond.GetBondType().name)
            src.extend((atom_start, atom_end))
            dst.extend((atom_end, atom_start))
            edge_weights.extend((bond_weight, bond_weight))
            edge_types.extend((EDGE_TYPE_BOND, EDGE_TYPE_BOND))

        for pharmacophore_id, atom_ids in enumerate(e_list):
            for atom_id in atom_ids:
                src.append(pharmacophore_count + atom_id)
                dst.append(pharmacophore_id)
                edge_weights.append(0.0)
                edge_types.append(EDGE_TYPE_DEPENDENCY)

    graph_node_count = pharmacophore_count + atom_count if include_atoms else pharmacophore_count
    g = dgl.graph((torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)),
                  num_nodes=graph_node_count)
    g.edata['dist'] = torch.HalfTensor(edge_weights)
    g.edata['edge_type'] = torch.tensor(edge_types, dtype=torch.long)

    type_list_tensor = torch.stack(type_list)
    phar_size_tensor = torch.HalfTensor(size_)
    phar_features = torch.cat((type_list_tensor.float(), phar_size_tensor.float().unsqueeze(1)), dim=1)
    g.ndata['type'] = type_list_tensor
    g.ndata['size'] = phar_size_tensor
    g.ndata['is_pharmacophore'] = torch.ones(pharmacophore_count, dtype=torch.bool)
    g.ndata['h'] = phar_features

    if include_atoms:
        assert atom_features is not None
        atom_size_tensor = -torch.ones(atom_count, dtype=torch.float16)
        atom_type_tensor = torch.zeros((atom_count, 7), dtype=type_list_tensor.dtype)
        g.ndata['type'] = torch.cat((type_list_tensor, atom_type_tensor), dim=0)
        g.ndata['size'] = torch.cat((phar_size_tensor, atom_size_tensor), dim=0)
        g.ndata['is_pharmacophore'] = torch.cat((
            torch.ones(pharmacophore_count, dtype=torch.bool),
            torch.zeros(atom_count, dtype=torch.bool)
        ))
        atom_features = torch.cat((atom_features, atom_size_tensor.float().unsqueeze(1)), dim=1)
        g.ndata['h'] = torch.cat((phar_features, atom_features), dim=0)
    smiles_code_res = smiles_code_(smiles, g, e_list)

    return g, smiles_code_res


if __name__ == '__main__':
    smiles = 'CS(=O)(=O)Nc1ccc(-c2ccnc(Nc3ccc(S(N)(=O)=O)cc3)n2)cc1'
    # g_, smiles_code_res = smiles2ppgraph(smiles)  ##g_药效团；smiles_code_res：smiles编码（矩阵格式）
    g_2, smiles_code_res2 = smiles2ppgraph(smiles)
    print()
