"""
Prepare the enwik8 dataset for character-level language modeling.
Will save train.bin, val.bin, test.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.

Source: 
- For download link: https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/getdata.sh#L34
- For script skeleton: https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py
"""
import os
import pickle
import zipfile
import numpy as np
import urllib.request

# download the enwik8 dataset if not present
input_file_path = os.path.join(os.path.dirname(__file__), 'enwik8')
zip_file_path = os.path.join(os.path.dirname(__file__), 'enwik8.zip')

if not os.path.exists(input_file_path):
    if not os.path.exists(zip_file_path):
        print("Downloading enwik8 dataset...")
        data_url = 'http://mattmahoney.net/dc/enwik8.zip'
        urllib.request.urlretrieve(data_url, zip_file_path)
        print("Download complete!")
    
    print("Extracting enwik8...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(__file__))
    print("Extraction complete!")

# read the dataset
with open(input_file_path, 'rb') as f:
    data = f.read()

# decode bytes to string
data = data.decode('utf-8', errors='ignore')
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars[:100]))  # print first 100 for readability
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and validation splits
# enwik8 standard split: 90M train, 5M val, 5M test
n = len(data)
num_test_chars = 5000000
train_data = data[:n - 2 * num_test_chars]
val_data = data[n - 2 * num_test_chars: n - num_test_chars]
test_data = data[n - num_test_chars:]

# encode all to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Done! Created train.bin, val.bin, test.bin, and meta.pkl")
