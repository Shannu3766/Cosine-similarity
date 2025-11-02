"""
Fine-tune a DeepSeek model for Question Answering using Cosine-based Adaptive LoRA.
Designed for Kaggle GPU runtimes.

Run inside Kaggle:
    !pip install -q transformers datasets accelerate bitsandbytes
    !pip install -e .
    python -m cosine.examples.finetune_deepseek_qa_kaggle
"""

import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    default_data_collator,
)
# ✅ Import the dynamic fine-tuning method from your new package
from cosine import fine_tune_lora_dynamic


def prepare_squad(tokenizer, max_length=384, doc_stride=128):
    """Tokenize the SQuAD-v2 dataset for QA."""
    dataset = load_dataset("squad_v2")

    def preprocess(batch):
        questions = [q.strip() for q in batch["question"]]
        inputs = tokenizer(
            questions,
            batch["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs.pop("offset_mapping")
        start_positions, end_positions = [], []

        for i, offsets in enumerate(offset_mapping):
            input_ids = inputs["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = inputs.sequence_ids(i)
            sample_idx = sample_mapping[i]
            answers = batch["answers"][sample_idx]

            if len(answers["answer_start"]) == 0:
                # No answer provided
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing SQuAD-v2",
    )
    return tokenized["train"].select(range(2000)), tokenized["validation"].select(range(500))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Preparing SQuAD-v2 dataset...")
    train_ds, val_ds = prepare_squad(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, collate_fn=default_data_collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False, collate_fn=default_data_collator
    )

    print("Loading DeepSeek model...")
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",   # spread across GPUs if available
    )

    print("Starting Cosine-based Adaptive LoRA fine-tuning...")
    fine_tune_lora_dynamic(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        total_R=32,
        tau=0.5,
        epochs=2,
        lr=1e-5,
        weight_decay=0.01,
        max_batches_for_bi=2,
        recompute_every=1,
        fast_mode=True,
    )

    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "deepseek_cosine_qa.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved successfully to {save_path}\n")

    # quick inference test
    model.eval()
    context = "The Eiffel Tower is located in Paris and was completed in 1889."
    question = "When was the Eiffel Tower completed?"
    inputs = tokenizer(question, context, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][start:end])
    print(f"Predicted answer: {answer}")


if __name__ == "__main__":
    main()
