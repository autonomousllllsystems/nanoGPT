"""
Prepare the enwik8 dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.

Source: 
How to split? 
https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
https://github.com/facebookresearch/adaptive-span
"""
import os
import requests
import numpy as np
import zipfile
from pathlib import Path
import pickle
import zipfile
import requests

# download the enwik8 dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'enwik8')
if not os.path.exists(input_file_path):
    data_url = 'http://mattmahoney.net/dc/enwik8.zip'
    """Prepare the enwik8 dataset for character-level language modeling.
    """

    ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
    TEST_CHARS = 5_000_000


    def download_enwik8(dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        raw_file = dest_dir / "enwik8"
        zip_file = dest_dir / "enwik8.zip"

        if raw_file.exists():
            return raw_file

        resp = requests.get(ENWIK8_URL)
        resp.raise_for_status()
        zip_file.write_bytes(resp.content)

        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(dest_dir)
        return raw_file


    def build_vocab(text: str):
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return chars, stoi, itos


    def encode(text: str, stoi):
        return [stoi[c] for c in text]


    def split_dataset(text: str, test_chars: int):
        tail = 2 * test_chars
        train = text[:-tail]
        val = text[-tail:-test_chars]
        test = text[-test_chars:]
        return train, val, test


    def write_bins(train_ids, val_ids, test_ids, dest: Path):
        dest.mkdir(parents=True, exist_ok=True)
        np.array(train_ids, dtype=np.uint16).tofile(dest / "train.bin")
        np.array(val_ids, dtype=np.uint16).tofile(dest / "val.bin")
        np.array(test_ids, dtype=np.uint16).tofile(dest / "test.bin")


    def write_meta(meta, dest: Path):
        with open(dest / "meta.pkl", "wb") as fh:
            pickle.dump(meta, fh)


    def main():
        data_dir = Path(__file__).resolve().parent
        raw_path = download_enwik8(data_dir)

        text = raw_path.read_text(encoding="latin-1")
        print(f"length of dataset in characters: {len(text):,}")

        chars, stoi, itos = build_vocab(text)
        print("all the unique characters:", "".join(chars))
        print(f"vocab size: {len(chars):,}")

        train_text, val_text, test_text = split_dataset(text, TEST_CHARS)
        train_ids = encode(train_text, stoi)
        val_ids = encode(val_text, stoi)
        test_ids = encode(test_text, stoi)

        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")
        print(f"test has {len(test_ids):,} tokens")

        write_bins(train_ids, val_ids, test_ids, data_dir)

        meta = {
            "vocab_size": len(chars),
            "itos": itos,
            "stoi": stoi,
        }
        write_meta(meta, data_dir)


    if __name__ == "__main__":
        main()