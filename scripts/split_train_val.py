from pathlib import Path


def main() -> None:
    data_dir = Path("data")
    source = data_dir / "train.txt"

    if not source.exists():
        raise FileNotFoundError(f"Missing file: {source}")

    text = source.read_text(encoding="utf-8", errors="ignore")

    split_idx = int(len(text) * 0.995)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    source.write_text(train_text, encoding="utf-8")
    (data_dir / "val.txt").write_text(val_text, encoding="utf-8")

    print("Wrote:")
    print(f"  {source} -> {source.stat().st_size / (1024**2):.2f} MB")
    print(f"  {data_dir / 'val.txt'} -> {(data_dir / 'val.txt').stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()