# 🧠 NextWordD: Context-Based Word Generator with MLP

**NextWordMLP** is a simple neural language model that predicts the next word in a sentence using only a **Multi-Layer Perceptron (MLP)**. It is trained on the classic text *The Adventures of Sherlock Holmes* and supports user-adjustable parameters like **block size** and **embedding dimension**.

This project demonstrates that even without RNNs or Transformers, meaningful language modeling is possible with careful design and sufficient context.

---

## 🎯 Try it Live

👉 **[Streamlit App](https://nextwordd.streamlit.app/)**

The app lets you:
- Select a **block size** (context length)
- Tune the **embedding dimension**
- Generate one word at a time from a live MLP model
- Observe how the model behaves with different hyperparameters

---

## ⚙️ Features

- 🔸 Trainable word **embeddings** (`nn.Embedding`)
- 🔸 Fully connected **MLP** with ReLU and Dropout
- 🔸 Tokenization and vocabulary built from Sherlock Holmes corpus
- 🔸 Streamlit interface for real-time interaction

---

## 🧠 Methodology

### 1. Preprocessing
- Text cleaned and tokenized into words
- Vocabulary indexed as `{word: id}` and `{id: word}`
- Inputs: sliding window of `block_size` words → target: next word

### 2. Model Architecture
```text
[ context (N word indices) ]
           ↓
   Embedding Layer
           ↓
   Flattened Embeddings
           ↓
   Linear → ReLU → Dropout → Linear
           ↓
 [ logits over vocabulary ]

### 3. PyTorch Skeleton:

```python
class NextWordMLP(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(block_size * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x):
        emb = self.embedding(x)            # (B, block_size, emb_dim)
        flat = emb.view(x.size(0), -1)     # (B, block_size * emb_dim)
        return self.net(flat)              # (B, vocab_size)

## 📁 Repository Structure

NextWordMLP/<br>
├── text_model_state.pth # model saved using torch.save<br>
├── notebook/ # Jupyter notebooks for training & analysis<br>
├── requirements.txt # Python dependencies<br>
└── README.md # This file

## 🛠 Tools & Libraries

- PyTorch – Model training & embedding
- Streamlit – Interactive web interface
- Matplotlib – Visualizations

## 👥 Contributors

- **A.V.S Manoj** (23110025) – [manoj.anaparthi@iitgn.ac.in](mailto:manoj.anaparthi@iitgn.ac.in)  
- **N. Eshwar** (23110215) – [eshwar.nakka@iitgn.ac.in](mailto:eshwar.nakka@iitgn.ac.in)  
- **O. Akash** (23110225) – [23110225@iitgn.ac.in](mailto:23110225@iitgn.ac.in)
- **P. Praneeth** (23110226) – [pabbathi.praneeth@iitgn.ac.in](mailto:pabbathi.praneeth@iitgn.ac.in)

---
