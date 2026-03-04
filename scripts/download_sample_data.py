from pathlib import Path
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    out_file = data_dir / "train.txt"

    if out_file.exists():
        print("Dataset already exists:", out_file)
        return

    print("Downloading Tiny Shakespeare dataset...")

    urllib.request.urlretrieve(DATA_URL, out_file)

    size = out_file.stat().st_size
    print(f"Saved dataset to {out_file}")
    print(f"Dataset size: {size/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
