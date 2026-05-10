"""
Türkçe Haber Kategori Tahmini (Masked Bi-LSTM + Attention Sürümü)
─────────────────────────────
Kullanım:
    python predict.py                  → interaktif mod
    python predict.py "Metin buraya"   → tek seferlik tahmin
"""

import os
import sys
import re
import json
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# Ayarlar (main.py ile senkronize edildi)
# ─────────────────────────────────────────────
MAX_LEN       = 256       # 150'den 256'ya çıkarıldı
EMBEDDING_DIM = 200
NUM_CLASSES   = 7
DROPOUT_RATE  = 0.5       # 0.4'ten 0.5'e çıkarıldı
LABEL_NAMES   = ["Siyaset", "Dünya", "Ekonomi", "Kültür", "Sağlık", "Spor", "Teknoloji"]
VOCAB_PATH    = "vocab.json"
MODEL_PATH    = "best_Bi-LSTM_Attention.pt"  

# ─────────────────────────────────────────────
# Cihaz
# ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ─────────────────────────────────────────────
# Metin temizleme (main.py ile genişletilmiş versiyon)
# ─────────────────────────────────────────────
_TR_STOPS = {
    "bir", "bu", "ve", "da", "de", "ile", "için", "olan", "olarak", "oldu", 
    "gibi", "çok", "daha", "en", "ise", "o", "bu", "şu", "ne", "kadar",
    "mi", "mu", "mü", "mı", "ki", "ya", "hem", "ama", "veya", "göre",
    "ancak", "fakat", "lakin", "üzere", "kadar", "sonra", "önce", "diye",
    "her", "hiç", "biz", "siz", "ben", "sen", "onlar", "bunu", "gibi",
    "buna", "bunun", "şu", "şunu", "şuna", "ya", "yani", "ise", "zaman",
    "vardı", "yok", "var", "gibi", "kendi", "tarafından", "neden", "niçin"
}

def temizle(metin: str) -> list:
    metin = metin.lower()
    metin = re.sub(r"[^\w\s]", " ", metin)
    metin = re.sub(r"\d+", " ", metin)
    return [k for k in metin.split() if k not in _TR_STOPS and len(k) > 1]

# ─────────────────────────────────────────────
# Vocab Yükleme ve Güvenlik Kontrolü
# ─────────────────────────────────────────────
if not os.path.exists(VOCAB_PATH):
    print(f"[HATA] {VOCAB_PATH} bulunamadı! Önce main.py dosyasını çalıştırarak veriyi eğitin.")
    sys.exit(1)

with open(VOCAB_PATH, encoding="utf-8") as f:
    w2i = json.load(f)
VOCAB_SIZE = len(w2i)

def encode(tokens):
    ids = [w2i.get(t, 1) for t in tokens[:MAX_LEN]]   # 1 = <UNK>
    ids += [0] * (MAX_LEN - len(ids))                  # 0 = <PAD>
    return ids

# ─────────────────────────────────────────────
# Model (main.py ile AYNISI)
# ─────────────────────────────────────────────
class MaskedAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs, mask):
        attn_scores = self.attention(rnn_outputs).squeeze(-1) 
        # Padding olan yerleri maskele
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context

class BiLSTMAttention(nn.Module):
    # hidden parametresi model kapasitesi arttığı için 128'e eşitlendi
    def __init__(self, vocab_size, embed_dim, num_classes, hidden=128, dropout=DROPOUT_RATE):
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
        mask = (x != 0) # Sıfırları maskeliyoruz
        emb = self.dropout(self.embedding(x))
        output, _ = self.lstm(emb)           
        attn_out = self.attention(output, mask)    
        out = self.fc(self.dropout(attn_out))
        return out

# ─────────────────────────────────────────────
# Yükleme
# ─────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[HATA] {MODEL_PATH} bulunamadı! Önce eğitimi tamamladığınızdan emin olun.")
    sys.exit(1)

print("Model ağırlıkları yükleniyor...")
model = BiLSTMAttention(VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"Hazır — {MODEL_PATH} | Vocab: {VOCAB_SIZE:,} kelime | Cihaz: {device}\n")

# ─────────────────────────────────────────────
# Tahmin
# ─────────────────────────────────────────────
def tahmin_et(metin: str):
    tokens = temizle(metin)
    
    if len(tokens) == 0:
        print("\n[Uyarı] Girilen metin sadece boşluk/noktalama veya stopwords içeriyor!")
        return

    ids = torch.tensor([encode(tokens)], dtype=torch.long).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(ids), dim=1).squeeze().cpu().tolist()

    sirali = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    tahmin = LABEL_NAMES[sirali[0][0]]
    guven  = sirali[0][1] * 100

    print(f"\n{'─'*45}")
    print(f"  Tahmin   : {tahmin}")
    print(f"  Güven    : %{guven:.1f}")
    print(f"{'─'*45}")
    for idx, prob in sirali:
        bar = "█" * int(prob * 30)
        print(f"    {LABEL_NAMES[idx]:<12} {bar:<30} %{prob*100:5.1f}")
    print()

# ─────────────────────────────────────────────
# Ana akış
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        tahmin_et(" ".join(sys.argv[1:]))
    else:
        print("Türkçe bir haber metni girin (çıkmak için 'q'):\n")
        while True:
            try:
                metin = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nÇıkılıyor..."); break
            
            if metin.lower() in ("q", "quit", "exit", "çıkış"):
                print("Çıkılıyor..."); break
            if not metin:
                continue
                
            tahmin_et(metin)