import argparse
import subprocess
from pathlib import Path


def run_command(cmd):
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(str(x) for x in cmd))
    print("=" * 80 + "\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}.")


def find_model(output_dir, n_epochs):
    output_dir = Path(output_dir)

    candidates = [
        output_dir / f"epoch_{n_epochs}_finetuned_model.pth",
        output_dir / f"epoch_{n_epochs}_finetuned_model.pt",
    ]

    for model_path in candidates:
        if model_path.exists():
            return str(model_path)

    raise FileNotFoundError(
        f"Cannot find fine-tuned model for epoch {n_epochs} "
        f"in {output_dir}"
    )


def run_finetuning(args, n_epochs, threshold, output_dir):
    cmd = [
        "python",
        "RL_generate.py",
        "--target_name",
        *args.target_name,
        "--data_path",
        args.data_path,
        "--output_dir",
        output_dir,
        "--model_path",
        args.model_path,
        "--tokenizer_path",
        args.tokenizer_path,
        "--n_mol",
        str(args.n_mol),
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--threshold",
        f"{threshold:.2f}",
        "--n_epochs",
        str(n_epochs),
        "--optimize_n_epochs",
        str(args.optimize_n_epochs),
        "--save_frequency",
        str(args.save_frequency),
        "--save_payloads",
        "--keep_top",
        str(args.keep_top),
    ]

    run_command(cmd)


def run_generation(args, model_path, output_dir, save_file):
    cmd = [
        "python",
        "generate.py",
        "--target_name",
        *args.target_name,
        "--output_dir",
        output_dir,
        "--model_path",
        model_path,
        "--tokenizer_path",
        args.tokenizer_path,
        "--n_mol",
        str(args.generate_n_mol),
        "--device",
        args.device,
        "--filter",
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--save_file",
        save_file,
    ]

    run_command(cmd)


def main():
    parser = argparse.ArgumentParser(description="Automated hyperparameter search for PolyPharm.")
    parser.add_argument("--target_name", nargs=2, required=True, help="Two target names, e.g. GSK3B JNK3 or ROR_gamma DHODH")
    parser.add_argument("--data_path", required=True, help="Path to the fine-tuning training dataset.")
    parser.add_argument("--model_path", required=True, help="Path to the pre-trained model.")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the pre-trained tokenizer.")
    parser.add_argument("--output_prefix", required=True, help="Prefix for hyperparameter-search output directories.")

    parser.add_argument("--n_mol", type=int, default=30000)
    parser.add_argument("--generate_n_mol", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimize_n_epochs", type=int, default=5)
    parser.add_argument("--save_frequency", type=int, default=10)
    parser.add_argument("--keep_top", type=int, default=10000)

    args = parser.parse_args()

    # ============================================================
    # Stage 1: Search n_epochs
    # threshold = 0.60
    # ============================================================

    print("\n" + "#" * 80)
    print("# Stage 1: Search n_epochs")
    print("# n_epochs = 5, 10, 15, 20, 25")
    print("# threshold = 0.60")
    print("#" * 80)

    for n_epochs in [5, 10, 15, 20, 25]:
        ft_output = (f"{args.output_prefix}_epoch_{n_epochs}")
        gen_output = (f"{args.output_prefix}_epoch_{n_epochs}_generation")
        save_file = (f"{args.output_prefix}_epoch_{n_epochs}.csv")
        print(f"\n>>> Finetuning n_epochs={n_epochs}, " f"threshold=0.60")

        # Fine-tuning
        run_finetuning(args=args, n_epochs=n_epochs, threshold=0.60, output_dir=ft_output)

        # Find fine-tuned model
        model_path = find_model(ft_output, n_epochs)

        # Molecule generation
        run_generation(args=args, model_path=model_path, output_dir=gen_output, save_file=save_file)

        print(f"\nFinished n_epochs={n_epochs}." f"\nModel: {model_path}" f"\nGenerated molecules: {gen_output}/{save_file}")

    # ============================================================
    # Stage 2: Search threshold
    # n_epochs = 20
    # ============================================================

    print("\n" + "#" * 80)
    print("# Stage 2: Search threshold")
    print("# n_epochs = 20")
    print("# threshold = 0.50, 0.55, 0.60, 0.65, 0.70")
    print("#" * 80)

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        threshold_tag = f"{threshold:.2f}".replace(".", "")
        ft_output = (f"{args.output_prefix}_threshold_{threshold_tag}")
        gen_output = (f"{args.output_prefix}_threshold_{threshold_tag}_generation")
        save_file = (f"{args.output_prefix}_threshold_{threshold_tag}.csv")

        print(f"\n>>> Testing n_epochs=20, " f"threshold={threshold:.2f}")

        # Fine-tuning
        run_finetuning(args=args, n_epochs=20, threshold=threshold, output_dir=ft_output)

        # Find fine-tuned model
        model_path = find_model(ft_output, 20)

        # Molecule generation
        run_generation(args=args, model_path=model_path, output_dir=gen_output, save_file=save_file)

        print(f"\nFinished threshold={threshold:.2f}." f"\nModel: {model_path}" f"\nGenerated molecules: {gen_output}/{save_file}")

    print("\n" + "=" * 80)
    print("Hyperparameter search completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
