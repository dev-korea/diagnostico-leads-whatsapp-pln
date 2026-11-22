"""Ajuste fino do BERTimbau (neuralmind/bert-base-portuguese-cased) — laço PyTorch explícito.

- Vários seeds (padrão 0 1 2 3 4): reporte média ± desvio (ajuste fino é sensível à semente).
- Seleção de época e de limiar (multirrótulo) na VALIDAÇÃO; teste só é tocado no final.
- Truncamento cabeça+cauda: conversas longas mantêm início (queixa/pedido) e fim (desfecho).
- `--smoke` usa um BERT minúsculo aleatório offline: valida o encanamento, NÃO gera resultado científico.
GPU: rode no Colab/Kaggle/Runpod; 5 seeds × ~300 conversas levam poucos minutos numa T4.
"""
from __future__ import annotations

import argparse
import copy
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from ..dataset import TASKS, load_task_frame
from ..ingest import connect
from ..metrics import to_matrix
from .common import save_predictions

DEFAULT_MODEL = "neuralmind/bert-base-portuguese-cased"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def head_tail_ids(tok, text: str, max_len: int, head_frac: float = 0.25) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False)
    budget = max_len - 2
    if len(ids) > budget:
        h = int(budget * head_frac)
        ids = ids[:h] + ids[-(budget - h):]
    return [tok.cls_token_id] + ids + [tok.sep_token_id]


def collate(batch_ids: list[list[int]], pad_id: int, device):
    L = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), L), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, :len(x)] = torch.tensor(x); mask[i, :len(x)] = 1
    return ids.to(device), mask.to(device)


def build_tiny(texts: list[str], n_labels: int, multilabel: bool):
    """Tokenizador de vocabulário por palavra + BERT minúsculo (apenas para o modo --smoke)."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    words = sorted({w for t in texts for w in t.lower().split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    d = Path(tempfile.mkdtemp()); (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=512, num_labels=n_labels,
                     problem_type="multi_label_classification" if multilabel else "single_label_classification")
    return tok, (lambda: BertForSequenceClassification(cfg))


def build_pretrained(model_name: str, n_labels: int, multilabel: bool):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    pt = "multi_label_classification" if multilabel else "single_label_classification"
    return tok, (lambda: AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels, problem_type=pt))


@torch.no_grad()
def predict_logits(model, enc, pad_id, device, bs=32):
    model.eval(); out = []
    for i in range(0, len(enc), bs):
        ids, mask = collate(enc[i:i + bs], pad_id, device)
        out.append(model(input_ids=ids, attention_mask=mask).logits.float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 1))


def decode(logits: np.ndarray, labels: list[str], multilabel: bool, thr: float = 0.5):
    if multilabel:
        P = 1 / (1 + np.exp(-logits))
        return [{labels[j] for j in np.where(row >= thr)[0]} for row in P]
    return [labels[j] for j in logits.argmax(1)]


def fit_predict(df, task: str, tok, model_factory, seed: int, epochs: int = 6, lr: float = 3e-5,
                batch_size: int = 16, max_len: int = 256, patience: int = 2, device=None):
    spec = TASKS[task]; labels = spec["labels"]; ml = spec["multilabel"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = (df[df.part == p] for p in ("train", "val", "test"))
    enc = lambda d: [head_tail_ids(tok, t, max_len) for t in d.text]
    Etr, Eva, Ete = enc(tr), enc(va), enc(te)
    if ml:
        Ytr = torch.tensor(to_matrix(list(tr.y), labels), dtype=torch.float)
        pos = Ytr.sum(0).clamp(min=1); neg = (len(Ytr) - Ytr.sum(0)).clamp(min=1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=(neg / pos).clamp(max=10).to(device))
    else:
        Ytr = torch.tensor([labels.index(y) for y in tr.y], dtype=torch.long)
        cnt = torch.bincount(Ytr, minlength=len(labels)).float().clamp(min=1)
        loss_fn = nn.CrossEntropyLoss(weight=(len(Ytr) / (len(labels) * cnt)).to(device))
    set_seed(seed)
    model = model_factory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * max(1, (len(Etr) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / (0.1 * steps), max(0.0, (steps - s) / (0.9 * steps))))
    pad = tok.pad_token_id
    val_score = lambda lg, thr=0.5: f1_score(
        to_matrix(list(va.y), labels) if ml else [labels.index(y) for y in va.y],
        (to_matrix(decode(lg, labels, True, thr), labels) if ml else lg.argmax(1)),
        average="macro", zero_division=0)
    best, best_state, bad, g = -1.0, None, 0, torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(Etr), generator=g).tolist()
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            ids, mask = collate([Etr[j] for j in idx], pad, device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, Ytr[idx].to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        sc = val_score(predict_logits(model, Eva, pad, device)) if len(Eva) else float(ep)
        if sc > best:
            best, bad, best_state = sc, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    thr = 0.5
    if ml and len(Eva):
        lg = predict_logits(model, Eva, pad, device)
        thr = max(np.arange(0.2, 0.76, 0.05), key=lambda t: val_score(lg, float(t)))
    preds = decode(predict_logits(model, Ete, pad, device), labels, ml, float(thr))
    return te.conversation_id.tolist(), preds, {"best_val_macro_f1": best, "thr": float(thr)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model-name", default=DEFAULT_MODEL); ap.add_argument("--split", default="v1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", default="results"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    df = load_task_frame(connect(a.db), a.task, a.split)
    spec = TASKS[a.task]; rows = []
    if a.smoke:
        tok, factory = build_tiny(list(df.text), len(spec["labels"]), spec["multilabel"])
        method, ver = "bertimbau_smoke", "tiny-random"
    else:
        tok, factory = build_pretrained(a.model_name, len(spec["labels"]), spec["multilabel"])
        method, ver = "bertimbau", a.model_name
    for s in a.seeds:
        ids, preds, info = fit_predict(df, a.task, tok, factory, s, a.epochs, a.lr, a.batch_size, a.max_len)
        print(f"seed={s} val_macro_f1={info['best_val_macro_f1']:.3f}")
        rows += [{"conversation_id": i, "task": a.task, "method": method, "model_version": ver, "seed": s, "pred": p}
                 for i, p in zip(ids, preds)]
    save_predictions(Path(a.out) / f"preds_{method}_{a.task}.csv", rows)
    print(f"{method}/{a.task}: {len(rows)} linhas")


if __name__ == "__main__":
    main()
