"""
Example: run dynamic BI-based AdaLoRA fine-tuning on a tiny dummy dataset.

This demo:
 - builds small dataset using HuggingFace datasets or dummy data,
 - computes BI scores each epoch,
 - reallocates LoRA ranks each epoch,
 - trains adapters for that epoch.

Run: python adalora_bi/examples/finetune_bi_demo.py
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from cosine import fine_tune_lora_dynamic

# Simple dummy dataset for demo
class DummyTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx],
                             truncation=True,
                             padding="max_length",
                             max_length=self.max_length,
                             return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def collate_fn(batch):
    # batch already dict tensors
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Dummy texts
    texts = [
        "I love programming in Python",
        "The weather is sunny today",
        "Transformers are great",
        "I had pizza",
        "PyTorch is awesome",
        "OpenAI builds models",
        "I went for a run",
        "I enjoy reading books"
    ]
    labels = [0,1,0,1,0,0,1,1]

    ds = DummyTextDataset(texts, labels, tokenizer)
    train_loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Run dynamic training: BI recomputed at start of each epoch (so 2 epochs -> computed twice)
    fine_tune_lora_dynamic(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        total_R=16,
        tau=0.5,
        epochs=2,
        lr=5e-4,
        weight_decay=0.0,
        alpha=16,
        dropout=0.0,
        max_batches_for_bi=4,
        log_every=10,
        save_path=None
    )

if __name__ == "__main__":
    main()
