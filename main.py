"""
Türkçe Haber Metinleri Kategorizasyonu
TTC-4900 Veri Seti ile 1D-CNN ve Bi-LSTM (Masked Attention) Karşılaştırması

Veri Seti  : TTC-4900 (savasy/ttc4900 - HuggingFace)
Çerçeve    : PyTorch
Donanım    : MPS (Apple Silicon) / CUDA / CPU otomatik algılama
"""

# ─────────────────────────────────────────────
# Kütüphaneler
# ─────────────────────────────────────────────
import os
import re
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────
# Tekrarlanabilirlik
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ─────────────────────────────────────────────
# 1. HİPERPARAMETRELER
# ─────────────────────────────────────────────
MAX_LEN        = 256      # 150'den 256'ya çıkarıldı (Daha geniş bağlam)
BATCH_SIZE     = 64
EPOCHS         = 25
EMBEDDING_DIM  = 200      
NUM_CLASSES    = 7
DROPOUT_RATE   = 0.5      # Overfitting'i önlemek için 0.4'ten 0.5'e çıkarıldı
LEARNING_RATE  = 1e-3
MIN_FREQ       = 2        
MAX_VOCAB_SIZE = 15000    
WEIGHT_DECAY   = 1e-4     

LR_GRID     = [1e-2, 1e-3, 1e-4]
LABEL_NAMES = ["Siyaset", "Dünya", "Ekonomi", "Kültür", "Sağlık", "Spor", "Teknoloji"]
OUTPUT_DIR  = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 2. CİHAZ
# ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"\n{'='*50}")
print(f"  Aktif hesaplama birimi : {device}")
print(f"{'='*50}\n")


# ─────────────────────────────────────────────
# 3. ÖZEL TOKENIZER & STOPWORDS
# ─────────────────────────────────────────────
# Gereksiz kelime havuzu gürültüyü azaltmak için genişletildi
_TR_STOPS = {
    "bir", "bu", "ve", "da", "de", "ile", "için", "olan", "olarak", "oldu", 
    "gibi", "çok", "daha", "en", "ise", "o", "bu", "şu", "ne", "kadar",
    "mi", "mu", "mü", "mı", "ki", "ya", "hem", "ama", "veya", "göre",
    "ancak", "fakat", "lakin", "üzere", "kadar", "sonra", "önce", "diye",
    "her", "hiç", "biz", "siz", "ben", "sen", "onlar", "bunu", "gibi",
    "buna", "bunun", "şu", "şunu", "şuna", "ya", "yani", "ise", "zaman",
    "vardı", "yok", "var", "gibi", "kendi", "tarafından", "neden", "niçin"
}

def temizle(metin: str) -> list[str]:
    metin = metin.lower()
    metin = re.sub(r"[^\w\s]", " ", metin)   
    metin = re.sub(r"\d+", " ", metin)        
    kelimeler = metin.split()
    return [k for k in kelimeler if k not in _TR_STOPS and len(k) > 1]


class Vocab:
    PAD, UNK = 0, 1

    def __init__(self):
        self.w2i = {"<PAD>": 0, "<UNK>": 1}
        self.i2w = {0: "<PAD>", 1: "<UNK>"}

    def build(self, corpus: list[list[str]], min_freq: int = 2, max_size: int = 15000):
        sayac = Counter(k for cumle in corpus for k in cumle)
        for kelime, frek in sayac.most_common():
            if frek < min_freq:
                break
            if len(self.w2i) >= max_size:
                break
            idx = len(self.w2i)
            self.w2i[kelime] = idx
            self.i2w[idx]    = kelime
        print(f"  Vocab boyutu: {len(self.w2i):,} kelime (Limitli)")

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        ids = [self.w2i.get(t, self.UNK) for t in tokens[:max_len]]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.w2i)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.w2i, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        v = cls()
        with open(path, encoding="utf-8") as f:
            v.w2i = json.load(f)
        v.i2w = {i: w for w, i in v.w2i.items()}
        return v


# ─────────────────────────────────────────────
# 4. VERİ SETİ & EDA
# ─────────────────────────────────────────────
class HaberDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.samples = [
            (vocab.encode(temizle(t), max_len), l)
            for t, l in zip(texts, labels)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, label = self.samples[idx]
        return {
            "input_ids": torch.tensor(ids,   dtype=torch.long),
            "targets"  : torch.tensor(label, dtype=torch.long)
        }

def perform_eda(raw_data):
    """ Kılavuzdaki 'Veri Analizi Görselleştirmeleri' gereksinimi """
    print("EDA (Keşifsel Veri Analizi) grafikleri oluşturuluyor...")
    
    cats = raw_data["category"]
    counts = Counter(cats)
    names = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
    vals = [counts[i] for i in range(NUM_CLASSES)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=names, y=vals, ax=ax, hue=names, palette="viridis", legend=False)
    ax.set_title("Veri Seti Kategori Dağılımı (TTC-4900)")
    ax.set_ylabel("Örnek Sayısı")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_class_distribution.png"), dpi=150)
    plt.close(fig)

    lengths = [len(t.split()) for t in raw_data["text"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(lengths, bins=50, ax=ax, color="purple", kde=True)
    ax.set_xlim(0, 1000)
    ax.set_title("Haber Metinleri Kelime Sayısı Dağılımı")
    ax.set_xlabel("Kelime Sayısı")
    ax.set_ylabel("Frekans")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_length_distribution.png"), dpi=150)
    plt.close(fig)
    print("  -> Grafikler plots/ klasörüne kaydedildi.\n")

def load_and_split(test_ratio=0.15, val_ratio=0.15):
    print("Veri seti indiriliyor: savasy/ttc4900 ...")
    raw = load_dataset("savasy/ttc4900")["train"]
    
    perform_eda(raw)

    tmp       = raw.train_test_split(test_size=test_ratio, seed=SEED)
    test_raw  = tmp["test"]
    val_size  = val_ratio / (1.0 - test_ratio)
    tmp2      = tmp["train"].train_test_split(test_size=val_size, seed=SEED)
    train_raw = tmp2["train"]
    val_raw   = tmp2["test"]

    print(f"  Eğitim  : {len(train_raw):,} örnek")
    print(f"  Doğrulama: {len(val_raw):,} örnek")
    print(f"  Test    : {len(test_raw):,} örnek\n")

    print("Vocab inşa ediliyor...")
    vocab = Vocab()
    vocab.build([temizle(t) for t in train_raw["text"]], min_freq=MIN_FREQ, max_size=MAX_VOCAB_SIZE)
    vocab.save("vocab.json")

    def make_loader(ds, shuffle):
        dataset = HaberDataset(ds["text"], ds["category"], vocab, MAX_LEN)
        return DataLoader(dataset, batch_size=BATCH_SIZE,
                          shuffle=shuffle, num_workers=0)

    return (
        make_loader(train_raw, True),
        make_loader(val_raw,   False),
        make_loader(test_raw,  False),
        vocab
    )


# ─────────────────────────────────────────────
# 5. MODEL MİMARİLERİ
# ─────────────────────────────────────────────

class TextCNN(nn.Module):
    """ Paralel çok-kernel 1D-CNN (TextCNN) """
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=DROPOUT_RATE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 128, kernel_size=k, padding=k // 2)
            for k in [2, 3, 4, 5]
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(4)])
        self.relu    = nn.ReLU()
        self.pool    = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(128 * 4, num_classes)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)        
        emb = self.dropout(emb)                         
        
        feats = []
        for conv, bn in zip(self.convs, self.bns):
            f = conv(emb)
            f = bn(f)
            f = self.relu(f)
            f = self.pool(f).squeeze(2)
            feats.append(f)
            
        out = self.fc(self.dropout(torch.cat(feats, dim=1)))
        return out


class MaskedAttention(nn.Module):
    """ Gelişmiş Dikkat (Self-Attention) Mekanizması - Padding Maskeleme ile """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs, mask):
        attn_scores = self.attention(rnn_outputs).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context


class BiLSTMAttention(nn.Module):
    """ Çift Yönlü LSTM + Masked Attention """
    def __init__(self, vocab_size, embed_dim, num_classes,
                 hidden=128, dropout=DROPOUT_RATE): # hidden boyutu 64'ten 128'e çıkarıldı
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout if dropout > 0 else 0
        )
        self.attention = MaskedAttention(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        mask = (x != 0)
        emb = self.dropout(self.embedding(x))
        output, _ = self.lstm(emb)           
        attn_out = self.attention(output, mask)    
        out = self.fc(self.dropout(attn_out))
        return out


# ─────────────────────────────────────────────
# 6. EĞİTİM & DEĞERLENDİRME
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        tgt  = batch["targets"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(ids), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * ids.size(0)
        n += ids.size(0)
    return total_loss / n


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            tgt = batch["targets"].to(device)
            logits = model(ids)
            total_loss += criterion(logits, tgt).item() * ids.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(tgt.cpu().tolist())
    return total_loss / len(targets), accuracy_score(targets, preds), preds, targets


def full_training_run(model, name, train_loader, val_loader, test_loader,
                      lr=LEARNING_RATE, epochs=EPOCHS):
    model = model.to(device)
    # Modelin aşırı emin olup ezberlemesini engellemek için Label Smoothing eklendi (0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tr_losses, vl_losses, vl_accs = [], [], []
    best_loss, patience_ctr = float("inf"), 0
    PATIENCE = 7

    print(f"\n{'─'*50}\n  {name}  |  lr={lr}  |  epochs={epochs}\n{'─'*50}")

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        tr_losses.append(tr_loss)
        vl_losses.append(vl_loss)
        vl_accs.append(vl_acc)

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        if vl_loss < best_loss:
            best_loss = vl_loss
            patience_ctr = 0
            torch.save(model.state_dict(), f"best_{name.replace(' ','_')}.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  [Erken Durdurma] {PATIENCE} tur iyileşme yok.")
                break

    model.load_state_dict(torch.load(f"best_{name.replace(' ','_')}.pt",
                                     map_location=device, weights_only=True))

    _plot_curves(tr_losses, vl_losses, vl_accs, name)

    _, test_acc, preds, tgts = evaluate(model, test_loader, criterion)
    print(f"\n  Test Doğruluğu: {test_acc:.4f}")
    print(classification_report(tgts, preds, target_names=LABEL_NAMES, digits=4))

    _plot_cm(tgts, preds, name)

    tb = label_binarize(tgts,  classes=list(range(NUM_CLASSES)))
    pb = label_binarize(preds, classes=list(range(NUM_CLASSES)))
    try:
        roc = roc_auc_score(tb, pb, average="macro")
        print(f"  ROC-AUC (macro): {roc:.4f}")
    except ValueError:
        roc = None

    return dict(model_name=name, test_acc=test_acc, roc_auc=roc,
                preds=preds, targets=tgts,
                tr_losses=tr_losses, vl_losses=vl_losses)


# ─────────────────────────────────────────────
# 7. HİPERPARAMETRE ANALİZİ
# ─────────────────────────────────────────────

def lr_analysis(ModelClass, vocab_size, train_loader, val_loader, epochs=5):
    print(f"\n{'='*50}\n  Öğrenme Hızı Duyarlılık Analizi\n{'='*50}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{ModelClass.__name__} — Öğrenme Hızı Karşılaştırması")

    for lr in LR_GRID:
        m = ModelClass(vocab_size, EMBEDDING_DIM, NUM_CLASSES).to(device)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1) # Analizde de Label Smoothing uygulandı
        opt  = optim.AdamW(m.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        vl, va = [], []
        for _ in range(epochs):
            train_one_epoch(m, train_loader, crit, opt)
            l, a, _, _ = evaluate(m, val_loader, crit)
            vl.append(l); va.append(a)
        lbl = f"lr={lr:.0e}"
        axes[0].plot(range(1, epochs+1), vl, marker="o", label=lbl)
        axes[1].plot(range(1, epochs+1), va, marker="o", label=lbl)
        print(f"  lr={lr:.0e} | Val Loss: {vl[-1]:.4f} | Val Acc: {va[-1]:.4f}")

    for ax, yl in zip(axes, ["Doğrulama Kaybı", "Doğrulama Doğruluğu"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"lr_analysis_{ModelClass.__name__}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Grafik: {path}")


# ─────────────────────────────────────────────
# 8. GRAFİK YARDIMCILARI
# ─────────────────────────────────────────────

def _plot_curves(tr_losses, vl_losses, vl_accs, name):
    ep = range(1, len(tr_losses) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{name} — Öğrenme Eğrileri")
    a1.plot(ep, tr_losses, label="Eğitim Kaybı",    marker="o")
    a1.plot(ep, vl_losses, label="Doğrulama Kaybı", marker="s", linestyle="--")
    a1.set(xlabel="Epoch", ylabel="Loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(ep, vl_accs, label="Doğrulama Doğruluğu", marker="s",
            linestyle="--", color="orange")
    a2.set(xlabel="Epoch", ylabel="Accuracy"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"learning_curves_{name.replace(' ','_')}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Öğrenme eğrisi: {p}")


def _plot_cm(targets, preds, name):
    cm = confusion_matrix(targets, preds)
    cm_n = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_n, annot=True, fmt=".2f",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                cmap="Blues", ax=ax)
    ax.set_title(f"{name} — Karışıklık Matrisi"); ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"confusion_matrix_{name.replace(' ','_')}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Karışıklık matrisi: {p}")


def plot_comparison(results):
    names = [r["model_name"] for r in results]
    accs  = [r["test_acc"]   for r in results]
    rocs  = [r["roc_auc"] or 0 for r in results]
    x, w  = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, accs, w, label="Test Doğruluğu")
    b2 = ax.bar(x + w/2, rocs, w, label="ROC-AUC (macro)")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1)
    ax.set_title("Model Karşılaştırması"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, "model_comparison.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\n  Karşılaştırma: {p}")


# ─────────────────────────────────────────────
# 9. ANA AKIŞ
# ─────────────────────────────────────────────
if __name__ == "__main__":

    train_loader, val_loader, test_loader, vocab = load_and_split()
    VOCAB_SIZE = len(vocab)

    # LR analizi
    lr_analysis(TextCNN, VOCAB_SIZE, train_loader, val_loader, epochs=5)

    # TextCNN
    cnn  = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES)
    r_cnn = full_training_run(cnn, "TextCNN", train_loader, val_loader, test_loader)

    # Bi-LSTM + Masked Attention
    lstm  = BiLSTMAttention(VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES)
    r_lstm = full_training_run(lstm, "Bi-LSTM_Attention", train_loader, val_loader, test_loader)

    plot_comparison([r_cnn, r_lstm])

    print(f"\n{'='*50}\n  ÖZET\n{'='*50}")
    print(f"  {'Model':<20} | {'Test Acc':>10} | {'ROC-AUC':>10}")
    print(f"  {'-'*46}")
    for r in [r_cnn, r_lstm]:
        roc = f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A"
        print(f"  {r['model_name']:<20} | {r['test_acc']:>10.4f} | {roc:>10}")
    print(f"{'='*50}")
    print("\nTüm grafikler 'plots/' klasörüne kaydedildi.")