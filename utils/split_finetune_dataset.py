import os
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures


PHAR_MAPPING = {
    "Aromatic": 1,
    "Hydrophobe": 2,
    "PosIonizable": 3,
    "Acceptor": 4,
    "Donor": 5,
    "LumpedHydrophobe": 6,
}


def get_pharmacophore_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)

    fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    feats = factory.GetFeaturesForMol(mol)

    features = []
    for feat in feats:
        family = feat.GetFamily()
        if family not in PHAR_MAPPING:
            continue

        atom_indices = tuple(sorted(feat.GetAtomIds()))
        features.append((PHAR_MAPPING[family], atom_indices))

    return mol, features


def get_pharmacophore_signature(smiles):
    mol, features = get_pharmacophore_features(smiles)

    if mol is None or not features:
        return None

    conformer = mol.GetConformer()
    feature_info = []

    for phar_type, atom_indices in features:
        coords = []

        for atom_idx in atom_indices:
            pos = conformer.GetAtomPosition(atom_idx)
            coords.append([pos.x, pos.y, pos.z])

        center = np.mean(coords, axis=0)

        feature_info.append({
            "type": phar_type,
            "size": len(atom_indices),
            "center": center,
        })

    feature_info.sort(
        key=lambda x: (
            x["type"],
            x["size"],
            tuple(np.round(x["center"], 3))
        )
    )

    node_features = tuple((feature["type"], feature["size"]) for feature in feature_info)
    pairwise_distances = []

    for i in range(len(feature_info)):
        for j in range(i + 1, len(feature_info)):
            type_i = feature_info[i]["type"]
            type_j = feature_info[j]["type"]

            distance = np.linalg.norm(feature_info[i]["center"] - feature_info[j]["center"])
            pairwise_distances.append(
                (
                    min(type_i, type_j),
                    max(type_i, type_j),
                    round(float(distance), 1)
                )
            )

    pairwise_distances = tuple(sorted(pairwise_distances))

    return node_features, pairwise_distances


def build_pharmacophore_groups(df, smiles_column):
    groups = defaultdict(list)
    invalid_indices = []

    for idx, smiles in df[smiles_column].items():
        signature = get_pharmacophore_signature(smiles)
        if signature is None:
            invalid_indices.append(idx)
            continue
        groups[signature].append(idx)

    return groups, invalid_indices


def split_groups(df, groups, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.")

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda x: len(x[1]), reverse=True)

    total = len(df)
    targets = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * test_ratio,
    }

    indices = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for _, group_indices in group_items:
        group_size = len(group_indices)
        deficits = {split: targets[split] - counts[split] for split in targets}
        split = max(deficits, key=deficits.get)

        indices[split].extend(group_indices)
        counts[split] += group_size

    train_df = df.loc[indices["train"]].copy()
    val_df = df.loc[indices["val"]].copy()
    test_df = df.loc[indices["test"]].copy()

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def get_signatures(df, smiles_column):
    signatures = set()
    for smiles in df[smiles_column]:
        signature = get_pharmacophore_signature(smiles)
        if signature is not None:
            signatures.add(signature)

    return signatures


def process_target(csv_path, smiles_column, seed, train_ratio, val_ratio, test_ratio):
    df = pd.read_csv(csv_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in {csv_path}.")

    df = df.dropna(
        subset=[smiles_column]
    ).drop_duplicates(
        subset=[smiles_column]
    ).reset_index(drop=True)

    groups, invalid_indices = build_pharmacophore_groups(df, smiles_column)

    if invalid_indices:
        df = df.drop(index=invalid_indices).reset_index(drop=True)
        groups, _ = build_pharmacophore_groups(df, smiles_column)

    train_df, val_df, test_df = split_groups(
        df,
        groups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    print(
        f"{os.path.basename(csv_path)}: "
        f"total={len(df)}, "
        f"train={len(train_df)}, "
        f"val={len(val_df)}, "
        f"test={len(test_df)}, "
        f"groups={len(groups)}"
    )

    return train_df, val_df, test_df


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    train1, val1, test1 = process_target(
        args.target1,
        args.smiles_column,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    train2, val2, test2 = process_target(
        args.target2,
        args.smiles_column,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    train = pd.concat([train1, train2], ignore_index=True)
    val = pd.concat([val1, val2], ignore_index=True)
    test = pd.concat([test1, test2], ignore_index=True)

    train = train.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val = val.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test = test.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    total = len(train) + len(val) + len(test)

    print(
        f"Final dataset: "
        f"train={len(train)} ({len(train) / total:.2%}), "
        f"val={len(val)} ({len(val) / total:.2%}), "
        f"test={len(test)} ({len(test) / total:.2%})"
    )

    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune dataset splitting.")
    parser.add_argument("--ligands_set1", type=str, required=True, help="Path to the first target ligand CSV.")
    parser.add_argument("--ligands_set2", type=str, required=True, help="Path to the second target ligand CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--smiles_column", type=str, default="SMILES", help="Name of the SMILES column.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)
