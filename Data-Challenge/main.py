import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np


def sinkhorn_log_domain(x, y, eps=0.05, n_iters=40, p=1):
    """
    Entropic Sinkhorn OT between two 1D empirical measures.
    x: tensor (n,)
    y: tensor (m,)
    """
    x = x.view(-1, 1)
    y = y.view(1, -1)
    if p == 1:
        C = torch.abs(x - y)
    else:
        C = (x - y) ** 2

    logK = -C / eps
    n, m = C.shape
    device = x.device

    loga = torch.log(torch.full((n,), 1.0 / n, device=device))
    logb = torch.log(torch.full((m,), 1.0 / m, device=device))

    logu = torch.zeros_like(loga)
    logv = torch.zeros_like(logb)

    for _ in range(n_iters):
        logu = loga - torch.logsumexp(logK + logv.view(1,-1), dim=1)
        logv = logb - torch.logsumexp(logK.t() + logu.view(1,-1), dim=1)

    logT = logu.view(-1,1) + logK + logv.view(1,-1)
    T = torch.exp(logT)
    return torch.sum(T * C)


class ScoreDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=256, train=True):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        text = (
            f"metric_name: {row.get('metric_name','')}\n"
            f"system_prompt: {row.get('system_prompt','')}\n"
            f"user_prompt: {row.get('user_prompt','')}\n"
            f"response: {row.get('response','')}"
        )

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}

        if self.train:
            item["score"] = torch.tensor(float(row["score"]), dtype=torch.float32)

        item["id"] = row.get("id", idx)
        return item


class Scorer(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.reg = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        raw = self.reg(pooled).squeeze(-1)
        return torch.sigmoid(raw) * 10.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train(
        train_path="train_data.json",
        test_path="test_data.json",
        out_csv="submission.csv",
        model_name="bert-base-uncased",
        lr=2e-5,
        batch_size=8,
        epochs=3,
        lambda_ot=0.1,
        eps=0.05,
        target_pool_size=30000
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_data = load_json(train_path)
    test_data = load_json(test_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = ScoreDataset(train_data, tokenizer, train=True)
    test_ds = ScoreDataset(test_data, tokenizer, train=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = Scorer(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()


    print("Using target distribution: Uniform[2,8]")
    target_pool = (torch.rand(target_pool_size) * 6 + 2).to(device)
    print("target_pool | mean:", float(target_pool.mean()),
          "std:", float(target_pool.std()))


    model.train()
    for ep in range(epochs):
        task_vals = []
        ot_vals = []
        print(f"\nEpoch {ep+1}/{epochs}")

        for batch in tqdm(train_dl):

            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            scores = batch["score"].to(device)

            preds = model(input_ids, attn)

            task_loss = loss_fn(preds, scores)

            idx = torch.randint(0, len(target_pool), (preds.shape[0],), device=device)
            tgt = target_pool[idx]

            ot_loss = sinkhorn_log_domain(preds, tgt, eps=eps, n_iters=40)

            loss = task_loss + lambda_ot * ot_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            task_vals.append(task_loss.item())
            ot_vals.append(ot_loss.item())

        print(f"Epoch {ep+1}: task_loss={np.mean(task_vals):.4f}, "
              f"ot_loss={np.mean(ot_vals):.4f}")

    model.eval()
    preds_list = []
    ids_list = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            preds = model(input_ids, attn)

            preds_list.extend(preds.cpu().tolist())
            ids_list.extend(batch["id"])

    df = pd.DataFrame({"id": ids_list, "score": preds_list})
    df.to_csv(out_csv, index=False)
    print("Saved submission to:", out_csv)


if __name__ == "__main__":
    train(
        train_path="train_data.json",
        test_path="test_data.json",
        out_csv="submission.csv",
        model_name="bert-base-uncased",
        lr=2e-5,
        batch_size=8,
        epochs=100,
        lambda_ot=0.1,
        eps=0.05,
        target_pool_size=30000
    )