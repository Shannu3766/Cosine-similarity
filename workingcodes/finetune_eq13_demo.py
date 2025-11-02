"""Demo: fine-tune DistilBERT using Eq.1.3 importance within adalora_bi_eq13 package."""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator
from adalora_bi_eq13 import fine_tune_lora_dynamic

class DummyDataset(Dataset):
    def __init__(self, tokenizer):
        texts = ["I love AI", "Hate bugs", "Nice weather", "Bad code"]
        labels = [1,0,1,0]
        self.data = [
            {
                **tokenizer(t, truncation=True, padding='max_length', max_length=16, return_tensors='pt'),
                'labels': torch.tensor(l)
            }
            for t,l in zip(texts, labels)
        ]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return {k: v.squeeze(0) for k,v in self.data[i].items()}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    ds = DummyDataset(tokenizer)
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=default_data_collator)

    fine_tune_lora_dynamic(
        model=model,
        train_loader=dl,
        val_loader=dl,
        device=device,
        total_R=16,
        tau=0.5,
        epochs=2,
        lr=5e-4,
        weight_decay=0.0,
        max_batches_for_bi=2,
        recompute_every=1,
        fast_mode=False,
    )

if __name__ == '__main__':
    main()
