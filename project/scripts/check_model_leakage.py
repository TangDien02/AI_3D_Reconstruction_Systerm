from pathlib import Path
import sys
import argparse
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


def normalize_model_path(path: object) -> str:
    return str(path).replace("\\", "/").strip()


def cad_uid_from_model(path: object) -> str:
    normalized = normalize_model_path(path)
    return Path(normalized).parent.as_posix() or normalized


def load_split(split_dir: Path, name: str) -> pd.DataFrame:
    path = split_dir / f"{name}.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    df["split"] = name

    if "cad_uid" in df.columns:
        df["_leak_key"] = df["cad_uid"].astype(str)
        key_name = "cad_uid"
    elif "model" in df.columns:
        df["_leak_key"] = df["model"].apply(cad_uid_from_model)
        key_name = "cad_uid(derived)"
    elif "model_uid" in df.columns:
        df["_leak_key"] = df["model_uid"].astype(str)
        key_name = "model_uid"
    else:
        raise KeyError("Split CSV must contain 'cad_uid', 'model', or 'model_uid' column.")

    return df, key_name


def report_overlap(left_name: str, left_df: pd.DataFrame, right_name: str, right_df: pd.DataFrame) -> int:
    left_keys = set(left_df["_leak_key"])
    right_keys = set(right_df["_leak_key"])

    overlap = left_keys & right_keys

    print(f"{left_name}-{right_name} overlap: {len(overlap)}")

    if overlap:
        print(f"\nLeak examples between {left_name} and {right_name}:")
        leak_rows = pd.concat([
            left_df[left_df["_leak_key"].isin(overlap)],
            right_df[right_df["_leak_key"].isin(overlap)],
        ])

        show_cols = [
            col for col in
            ["split", "sample_id", "category", "cad_uid", "model_uid", "model", "img", "pointcloud", "_leak_key"]
            if col in leak_rows.columns
        ]

        print(leak_rows[show_cols].head(20).to_string(index=False))
        print()

    return len(overlap)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Pix3D split leakage by CAD folder.")
    parser.add_argument("--processed-dir", default=str(PROJECT_DIR / "data" / "processed_2048"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    split_dir = processed_dir / "splits"
    train, key_name_train = load_split(split_dir, "train")
    val, key_name_val = load_split(split_dir, "val")
    test, key_name_test = load_split(split_dir, "test")

    print("Checking CAD model leakage...")
    print(f"Leakage key used: {key_name_train}")
    print(f"Train samples: {len(train)}")
    print(f"Val samples:   {len(val)}")
    print(f"Test samples:  {len(test)}")
    print()

    n_train_val = report_overlap("train", train, "val", val)
    n_train_test = report_overlap("train", train, "test", test)
    n_val_test = report_overlap("val", val, "test", test)

    total_leaks = n_train_val + n_train_test + n_val_test

    print("Summary")
    print("-------")

    if total_leaks == 0:
        print("OK: no CAD model leakage between train/val/test.")
        sys.exit(0)

    print("LEAKAGE DETECTED: the same CAD folder appears in multiple splits.")
    print("Regenerate processed splits with cad_uid grouped splitting.")
    sys.exit(1)


if __name__ == "__main__":
    main()
