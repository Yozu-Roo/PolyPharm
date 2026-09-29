# 🎯 PolyPharm

PolyPharm is a deep learning-based framework for multi-target drug design, capable of generating molecules with potential activities against multiple targets. 

## 1️⃣ Directory Structure & Key Files

```text
/                         ← Root directory
├── data                  ← Dataset, interaction scoring models, and multi-target scoring weights
│   ├── GSK3B+JNK3 
│   └── ROR_gamma+DHODH
├── results               ← Generated molecules from different methods on benchmark tasks
│   ├── gsk3β_jnk3 
│   └── rorγt_dhodh
├── model                 ← Model code
├── score_modules         ← Scoring utilities
│   ├── ESOL_Score
│   └── SA_Score
├── utils                     ← Utilities (multi-target scoring, data preprocessing, etc.)
├── environment.yml           ← Conda environment
├── train_chembl_baseline.py  ← Pre-training script
├── RL_generate.py            ← Fine-tuning script
├── generate.py               ← Molecule generation script
├── hyperparameter_search.py  ← Fine-tuning parameter selection script
├── generate.py               ← Molecule generation script
```

## 2️⃣ Results

✅ Pre-generated molecules are provided for download (including `QED`, `SA`, `Docking score`, `LogP`, `Weight`) along with comparison methods (some from [AIxFuse](https://github.com/biomed-AI/AIxFuse?tab=readme-ov-file) open-source data):

- **GSK3β|JNK3 benchmark task**: `results/gsk3β_jnk3/POLYGEN.csv`
- **RORγt|DHODH benchmark task**: `results/rorγt_dhodh/POLYGEN.csv`

💡 To train from scratch, follow the steps below.

## 3️⃣ Quick Start

### 3.1 📥 Clone Repository & Download Dataset

```bash
git clone https://github.com/Yozu-Roo/POLYGEN.git
```

- Dataset (~1.69GB) download via [Google Drive](https://drive.google.com/file/d/1meevHkArxd2SAeNh4tqCSEE8sJPH64ld/view?usp=drive_link)
- Or download via [Baidu Drive](https://pan.baidu.com/s/1aNK7QzjTPmnSIv1LBGKzlA?pwd=9yr9)
- Or contact via email: niuziru@stu.xmu.edu.cn

After download, extract and place the `data/` folder at the root directory.

------

### 3.2 ⚙️ Install Conda Environment

Recommended **Python 3.8**:

```bash
conda env create -f environment.yml
conda activate polypharm
```

------

### 3.3 🏋️‍♂️ Pre-training

```bash
python train_chembl_baseline.py
```

- For multiple GPUs, adjust `CUDA_VISIBLE_DEVICES` in the script
- Model weights are saved in `pretrain_output/` during training

------

### 3.4 🔧 Fine-tuning

**Splitting the fine-tuning dataset**:

```bash
python utils/split_finetune_dataset.py \
    --ligands_set1 ./data/GSK3B+JNK3/GSK3B.csv \
    --ligands_set2 ./data/GSK3B+JNK3/JNK3.csv \
    --output_dir ./data/GSK3B+JNK3/ \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --test_ratio 0.2
```

```bash
python utils/split_finetune_dataset.py \
    --ligands_set1 ./data/ROR_gamma+DHODH/ROR_gamma.csv \
    --ligands_set2 ./data/ROR_gamma+DHODH/DHODH.csv \
    --output_dir ./data/ROR_gamma+DHODH/ \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --test_ratio 0.2
```

The `train.csv`, `val.csv`, and `test.csv`—will be saved in the `output_dir`.

**Fine-tune on GSK3β|JNK3 benchmark task**:

```bash
python RL_generate.py \
    --target_name GSK3B JNK3 \
    --output_dir ./finetune_output_GJ \
    --data_path ./data/GSK3B+JNK3/train.csv \
    --model_path ./pretrain_output/rs_mapping/fold0_epoch32.pth \
    --tokenizer_path ./pretrain_output/rs_mapping/tokenizer.pkl \
    --n_mol 10000 \
    --device cuda \
    --batch_size 512 \
    --seed 42 \
    --threshold 0.60 \
    --n_epochs 20 \
    --optimize_n_epochs 5 \
    --save_frequency 10 \
    --save_payloads \
    --keep_top 5000
```

**Fine-tune on RORγt|DHODH benchmark task**:

```bash
python RL_generate.py \
    --target_name ROR_gamma DHODH \
    --data_path ./data/ROR_gamma+DHODH/train.csv \
    --output_dir ./finetune_output_RD \
    --model_path ./pretrain_output/fold0_epoch32.pth \
    --tokenizer_path ./pretrain_output/tokenizer.pkl \
    --n_mol 10000 \
    --device cuda \
    --batch_size 512 \
    --seed 42 \
    --threshold 0.60 \
    --n_epochs 15 \
    --optimize_n_epochs 5 \
    --save_frequency 10 \
    --save_payloads \
    --keep_top 5000
```

- `--tokenizer_path ` is your pre-trained model path
- `--threshold ` is the threshold for screening elite molecules 
- `--n_epochs ` is the fine-tuning epochs
- `--optimize_n_epochs ` is the optimization epochs
- `--n_mol` is the number of molecules sampled each epoch
- Generated results and model weights are saved in `finetune_output_*/`
- Multi-GPU users may modify `CUDA_VISIBLE_DEVICES`

> [!NOTE]
> **Hyperparameter selection**
>
> Hyperparameters were selected separately for the two benchmark tasks. We evaluated the number of fine-tuning epochs (`n_epochs`) over the range of **5–25 with a step size of 5**, while fixing `threshold` at **0.60**. We then evaluated the elite-molecule screening threshold (`threshold`) over the range of **0.50–0.70 with a step size of 0.05**, while fixing `n_epochs` at **20**. The remaining fine-tuning settings were kept unchanged.
>
> The entire hyperparameter search can be performed automatically using:
>
> ```bash
> python hyperparameter_search.py \
>     --target_name ROR_gamma DHODH \
>     --data_path ./data/ROR_gamma+DHODH/train.csv \
>     --val_data_path ./data/ROR_gamma+DHODH/val.csv \
>     --model_path ./pretrain_output/fold0_epoch32.pth \
>     --tokenizer_path ./pretrain_output/tokenizer.pkl \
>     --output_prefix ./hyperparam_RD \
>     --n_mol 30000 \
>     --generate_n_mol 10000 \
>     --device cuda \
>     --batch_size 512 \
>     --seed 42 \
>     --optimize_n_epochs 5 \
>     --save_frequency 10 \
>     --keep_top 5000
> ```
>
> For the GSK3β|JNK3 benchmark task:
>
> ```bash
> python hyperparameter_search.py \
>     --target_name GSK3B JNK3 \
>     --data_path ./data/GSK3B+JNK3/train.csv \
>     --val_data_path ./data/GSK3B+JNK3/val.csv \
>     --model_path ./pretrain_output/rs_mapping/fold0_epoch32.pth \
>     --tokenizer_path ./pretrain_output/rs_mapping/tokenizer.pkl \
>     --output_prefix ./hyperparam_GJ \
>     --n_mol 10000 \
>     --generate_n_mol 10000 \
>     --device cuda \
>     --batch_size 512 \
>     --seed 42 \
>     --optimize_n_epochs 5 \
>     --save_frequency 10 \
>     --keep_top 5000
> ```
>
> After the script finishes, CSV files containing the generated molecules for different parameter settings will be produced in the following directories:
> `hyperparam_**_threshold_**_generation/` and `hyperparam_**_epoch_**_generation/`.
>
> Next, perform batch docking on the generated molecules using [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina/releases) and calculate their `QED` and `SA` properties. Finally, calculate the `USR docking` scores and `SR` values according to the evaluation metrics described in the manuscript. Compare these results and select the parameter combination with the highest `SR` and `USR docking` scores as the final hyperparameter setting.

------

### 3.5 🧪 Molecule Generation

```bash
python generate.py \
    --target_name ROR_gamma DHODH \  
    --output_dir ./generate_output_RD \
    --save_file RD_gen.csv
    --model_path ./finetune_output_RD/epoch_15_finetuned_model.pth \  
    --tokenizer_path ./pretrain_output/tokenizer.pkl \
    --init_smi_path ./data/ROR_gamma+DHODH/test.csv \
    --n_mol 10000 \
    --device cuda \
    --filter \
    --batch_size 512 \
    --seed 42
```

- `--target_name` is the benchmark task and can be replaced with `GSK3B JNK3`
- `--init_smi_path` is the file path for the test set.
- `--model_path` is your fine-tuned model path
- `--n_mol` is the number of generated molecules 
- Generated molecules are saved in `generate_output_*/`

------

## 4️⃣  Tips🌟

- Ensure all paths are correct to avoid file-not-found errors
- GPU significantly speeds up training and generation
- Docking tool: [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina/releases) or using [Vina-GPU](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1) speeds up
- Retrosynthesis tool: [AiZynthFinder](https://github.com/MolecularAI/AiZynthFinder)

