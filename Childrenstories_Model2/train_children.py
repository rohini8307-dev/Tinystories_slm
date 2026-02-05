import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# Model config (MUST match base model)
# -------------------------------------------------
class GPTConfig:
    vocab_size = 50257
    block_size = 256
    n_layer = 8
    n_head = 8
    n_embd = 336
    dropout = 0.1

# -------------------------------------------------
# Training hyperparameters
# -------------------------------------------------
batch_size = 32
learning_rate = 1e-4
weight_decay = 0.1
betas = (0.9, 0.95)
grad_clip = 1.0

max_steps = 50_000
warmup_steps = 1_000
eval_interval = 1_000
eval_iters = 200

# 🔹 Early stopping settings
patience = 5                  # number of evals to wait
best_val_loss = float("inf")
patience_counter = 0

train_bin = "train_ch.bin"
val_bin = "val_ch.bin"

# -------------------------------------------------
# Dataset
# -------------------------------------------------
class BinDataset:
    def __init__(self, path, block_size):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data) - self.block_size - 1, (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i+1:i+self.block_size+1].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)

# -------------------------------------------------
# Model
# -------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0)

        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

# -------------------------------------------------
# Load base model
# -------------------------------------------------
checkpoint = torch.load("tinystories_28M_final.pt", map_location=device)
model = GPT(GPTConfig()).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
print("Loaded TinyStories base model")

# -------------------------------------------------
# Optimizer
# -------------------------------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=betas,
    weight_decay=weight_decay
)

train_data = BinDataset(train_bin, 256)
val_data = BinDataset(val_bin, 256)

# -------------------------------------------------
# LR schedule
# -------------------------------------------------
def get_lr(step):
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = val_data.get_batch(batch_size)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# -------------------------------------------------
# Training loop with Early Stopping
# -------------------------------------------------
model.train()
for step in range(max_steps):

    for pg in optimizer.param_groups:
        pg["lr"] = get_lr(step)

    xb, yb = train_data.get_batch(batch_size)
    optimizer.zero_grad(set_to_none=True)

    _, loss = model(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step} | train loss {loss.item():.4f}")

    if step % eval_interval == 0 and step > 0:
        val_loss = estimate_loss()
        print(f"🔍 validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(GPTConfig()),
                },
                "children_stories_fin.pt"
            )
            print("New best model saved")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

print("Training finished")
