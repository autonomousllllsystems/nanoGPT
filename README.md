# Character-Level Language Model (a modiefied nanoGPT)

This project implements and compares two character-level language models trained on the enwik8 dataset: a baseline transformer model and an enhanced version using Rotary Position Embeddings (RoPE).

## Project Overview

- **Dataset**: enwik8 (~100M characters from Wikipedia)
- **Task**: Character-level language modeling
- **Baseline**: Standard GPT-style transformer
- **Novel Approach**: GPT with Rotary Position Embeddings (RoPE)


## Setup

1. Clone the nanoGPT repository:
   ```
   git clone https://github.com/RiddhiRaj/RotaryCharTransformer.git

   cd RotaryCharTransformer
   ```

2. Install required packages:
   ```
   pip install transformers datasets tiktoken wandb tqdm numpy torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
   ```

## Data Download & Preparation

Download the dataset (press "y" when asked to run custom code)
```python
python data/enwik8/prep_enwik8.py
```

## Model Implementation

- Baseline model: Implemented in `model_baseline.py`
- RoPE model: Implemented in `model_rope.py`

## Training

1. Train the baseline model:
   ```
   python train.py --config config/enwik8_char_baseline.py
   ```

2. Train the RoPE model:
   ```
   python train.py --config config/enwik8_char_rope.py
   ```

## Evaluation

Evaluate both models:
```
python evaluate.py --model_type gpt --checkpoint out-enwik8-char/ckpt.pt
python evaluate.py --model_type rope --checkpoint out-enwik8-char-rope/ckpt.pt
```

## Results

The GPTWithRoPE model significantly outperforms the baseline, demonstrating the effectiveness of Rotary Position Embeddings in character-level language modeling.

For more detailed information on the implementation and analysis, please refer to the Jupyter notebook and individual Python files in this repository.


Implement

[] add enwik8 dataset
[] comparison between nanoGPT and the new one
[] change positional encoding.


Transformer XL rotate positional embeddings

[] memory mechanism
[] change positional encdoding



Transformer-XL:

Adds a segment-level recurrence mechanism (memory) to standard Transformers, allowing it to model much longer contexts efficiently.
Uses relative positional encodings for better generalization to longer sequences.
Designed for both word-level and character-level language modeling.
Handles very long sequences by caching hidden states from previous segments.


NanoGPT:

Implements a standard GPT (decoder-only Transformer) architecture, similar to OpenAI’s GPT-2.
Uses absolute positional encodings (no recurrence or memory).
Simpler, minimal codebase focused on ease of use and training small/medium GPT models.
No built-in support for segment-level recurrence or relative positions.


Summary:
Transformer-XL can model longer contexts and uses recurrence and relative positions, while NanoGPT is a straightforward GPT-style Transformer with absolute positions and no memory.


Transformer-XL uses relative positional encoding to understand token order and maintain context across distant tokens in the input sequence. The model's ability to handle longer contexts and maintain information from previous segments results in more coherent and contextually




https://github.com/RiddhiRaj/RotaryCharTransformer/blob/master/prepare_enwik8.py

https://github.com/RiddhiRaj/lunar-lander/blob/main/lunar_lander_neat.py
https://github.com/RiddhiRaj/kanji-generator
https://github.com/RiddhiRaj/multiagent-debate/blob/main/multiagent_debate_tinyllama.ipynb
