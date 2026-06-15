from pathlib import Path
import sys
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data" / "processed" / "splits"


def normalize_model_path(path: object) -> str:
    return str(path).replace("\\", "/").strip()


def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    df["split"] = name

    if "model_uid" in df.columns:
        df["_leak_key"] = df["model_uid"].astype(str)
        key_name = "model_uid"
    elif "model" in df.columns:
        df["_leak_key"] = df["model"].apply(normalize_model_path)
        key_name = "model"
    else:
        raise KeyError("Split CSV must contain either 'model_uid' or 'model' column.")

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
            ["split", "sample_id", "category", "model_uid", "model", "img", "pointcloud", "_leak_key"]
            if col in leak_rows.columns
        ]

        print(leak_rows[show_cols].head(20).to_string(index=False))
        print()

    return len(overlap)


def main() -> None:
    train, key_name_train = load_split("train")
    val, key_name_val = load_split("val")
    test, key_name_test = load_split("test")

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

    print("LEAKAGE DETECTED: the same CAD model appears in multiple splits.")
    print("Regenerate data/processed/splits with model-level grouped splitting.")
    sys.exit(1)


if __name__ == "__main__":
    main()
