"""SPECTER embeddings through ONNX Runtime, matching the labeling scripts.

Reproduces `embed_with_sliding()` from the stage-04 scripts step by step:
wordpiece-tokenize the text, cut it into 100-token windows at stride 50, turn
each window back into a string, re-encode it with [CLS]/[SEP] (truncated at
512 tokens), take the [CLS] vector (SPECTER's pooling), and average the
windows into one 768-dimensional vector per document.

The model is the ONNX export that sentence-transformers publishes for
`allenai-specter`, so this needs neither torch nor CUDA. On Windows it uses
DirectML (any GPU) if `onnxruntime-directml` is installed, else the CPU.
"""

import re

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL_REPO = "sentence-transformers/allenai-specter"
WINDOW, STRIDE = 100, 50

_tok = Tokenizer.from_file(hf_hub_download(MODEL_REPO, "tokenizer.json"))
_tok.no_truncation()
_tok.no_padding()
_sess = ort.InferenceSession(
    hf_hub_download(MODEL_REPO, "onnx/model.onnx"),
    providers=[p for p in ("DmlExecutionProvider", "CUDAExecutionProvider",
                           "CPUExecutionProvider") if p in ort.get_available_providers()],
)
_inputs = {i.name for i in _sess.get_inputs()}


def preprocess(text):
    """Document-side cleaning used by every labeling script."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return " ".join(w for w in text.split() if len(w) > 2)


def _windows(text):
    toks = _tok.encode(text, add_special_tokens=False).tokens
    # BertTokenizer.convert_tokens_to_string
    chunks = [" ".join(toks[i:i + WINDOW]).replace(" ##", "").strip()
              for i in range(0, len(toks), STRIDE)]
    return [c for c in chunks if c] or [""]


def _encode(strings, batch=128):
    out = np.zeros((len(strings), 768), np.float32)
    ids = []
    for e in (_tok.encode(s) for s in strings):
        ids.append(e.ids if len(e.ids) <= 512 else e.ids[:511] + [e.ids[-1]])
    order = np.argsort([len(i) for i in ids])
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        width = -(-max(len(ids[i]) for i in idx) // 16) * 16  # few distinct shapes
        input_ids = np.zeros((len(idx), width), np.int64)
        mask = np.zeros_like(input_ids)
        for r, i in enumerate(idx):
            input_ids[r, :len(ids[i])] = ids[i]
            mask[r, :len(ids[i])] = 1
        feed = {"input_ids": input_ids, "attention_mask": mask}
        if "token_type_ids" in _inputs:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        out[idx] = _sess.run(None, feed)[0][:, 0, :]  # CLS pooling
    return out


def embed(texts, chunk_docs=4000, progress=True):
    """One averaged sliding-window SPECTER vector per text."""
    res = np.zeros((len(texts), 768), np.float32)
    for start in range(0, len(texts), chunk_docs):
        part = texts[start:start + chunk_docs]
        wins, owner = [], []
        for j, t in enumerate(part):
            w = _windows(t)
            wins += w
            owner += [j] * len(w)
        owner = np.array(owner)
        sums = np.zeros((len(part), 768), np.float64)
        np.add.at(sums, owner, _encode(wins))
        res[start:start + len(part)] = sums / np.bincount(owner, minlength=len(part))[:, None]
        if progress:
            print(f"  embedded {start + len(part):,}/{len(texts):,}", flush=True)
    return res
