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

FDEF_NAME = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(FDEF_NAME)


def get_pharmacophore_signature(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = FEATURE_FACTORY.GetFeaturesForMol(mol)
    phar_types = [
        PHAR_MAPPING[feat.GetFamily()]
        for feat in features
        if feat.GetFamily() in PHAR_MAPPING
    ]

    if not phar_types:
        return None

    return tuple(sorted(phar_types))


def build_pharmacophore_groups(df, smiles_column):
    groups = defaultdict(list)
    invalid_indices = []

    for idx, smiles in df[smiles_column].items():
        signature = get_pharmacophore_signature(smiles)
        if signature is None:
            invalid_indices.append(idx)
        else:
            groups[signature].append(idx)

    return groups, invalid_indices


def split_groups(
    df,
    groups,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    seed=42,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.")

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda x: len(x[1]), reverse=True)

    targets = {
        "train": len(df) * train_ratio,
        "val": len(df) * val_ratio,
        "test": len(df) * test_ratio,
    }
    indices = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for _, group_indices in group_items:
        split = max(
            targets,
            key=lambda x: targets[x] - counts[x]
        )
        indices[split].extend(group_indices)
        counts[split] += len(group_indices)

    result = {}
    for split in ["train", "val", "test"]:
        result[split] = (
            df.loc[indices[split]]
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

    return result["train"], result["val"], result["test"]


def process_target(
    csv_path,
    smiles_column,
    seed,
    train_ratio,
    val_ratio,
    test_ratio,
):
    df = pd.read_csv(csv_path)

    if smiles_column not in df.columns:
        raise ValueError(
            f"Column '{smiles_column}' not found in {csv_path}."
        )

    df = (
        df.dropna(subset=[smiles_column])
        .drop_duplicates(subset=[smiles_column])
        .reset_index(drop=True)
    )

    groups, invalid_indices = build_pharmacophore_groups(
        df, smiles_column
    )

    if invalid_indices:
        df = df.drop(index=invalid_indices).reset_index(drop=True)
        groups, _ = build_pharmacophore_groups(
            df, smiles_column
        )

    return split_groups(
        df,
        groups,
        train_ratio,
        val_ratio,
        test_ratio,
        seed,
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    train1, val1, test1 = process_target(
        args.ligands_set1,
        args.smiles_column,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    train2, val2, test2 = process_target(
        args.ligands_set2,
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

    train.to_csv(
        os.path.join(args.output_dir, "train.csv"),
        index=False,
    )
    val.to_csv(
        os.path.join(args.output_dir, "val.csv"),
        index=False,
    )
    test.to_csv(
        os.path.join(args.output_dir, "test.csv"),
        index=False,
    )

    print(
        f"Train: {len(train)}, "
        f"Validation: {len(val)}, "
        f"Test: {len(test)}"
    )
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune dataset splitting.") 
    parser.add_argument("--ligands_set1", type=str, required=True, help="Path to the first target ligand CSV.") 
    parser.add_argument("--ligands_set2", type=str, required=True, help="Path to the second target ligand CSV.") 
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.") 
    parser.add_argument("--smiles_column", type=str, default="SMILES", help="Name of the SMILES column.") 
    parser.add_argument("--train_ratio", type=float, default=0.7) 
    parser.add_argument("--val_ratio", type=float, default=0.1) 
    parser.add_argument("--test_ratio", type=float, default=0.2) 
    parser.add_argument("--seed", type=int, default=42) 
    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)
