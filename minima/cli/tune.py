from __future__ import annotations

import argparse

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from minima.modeling import MinimaModel
from minima.tuning import MinimaForSequenceClassification, enable_recovery_training


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Tune Minima recovery adapters for text classification")
    parser.add_argument("model")
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--dataset-config")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-pair-column")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--num-labels", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--pooling", choices=("mean", "first"), default="mean")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    encoder = MinimaModel.from_pretrained(args.model, device="cuda")
    trainable = enable_recovery_training(encoder)
    model = MinimaForSequenceClassification(encoder, args.num_labels, args.pooling).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = load_dataset(args.dataset, args.dataset_config)

    def tokenize(batch):
        second = batch[args.text_pair_column] if args.text_pair_column else None
        encoded = tokenizer(batch[args.text_column], second, truncation=True, max_length=args.max_length)
        encoded["labels"] = batch[args.label_column]
        return encoded

    remove = dataset[args.train_split].column_names
    prepared = dataset.map(tokenize, batched=True, remove_columns=remove)

    def metrics(prediction):
        logits, labels = prediction
        if args.num_labels == 1:
            from scipy.stats import spearmanr
            return {"spearman": float(spearmanr(logits.squeeze(-1), labels).statistic)}
        predictions = np.asarray(logits).argmax(-1)
        return {"accuracy": float((predictions == labels).mean())}

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=prepared[args.train_split],
                      eval_dataset=prepared[args.eval_split], compute_metrics=metrics,
                      processing_class=tokenizer)
    print(f"trainable recovery parameters: {trainable:,}")
    trainer.train()
    print(trainer.evaluate())
    model.save_adapter(args.output, args.model)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()

