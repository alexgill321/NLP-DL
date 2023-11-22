import torch
import transformers
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import send_example_telemetry
from datasets import load_dataset, load_metric
import argparse
import os
import numpy as np
import pandas as pd

if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")

print(transformers.__version__)

def load_splits(task):
    if task == "rte":
        splits = pd.read_csv("hidden_rte.csv")

def finetune_model(model_name, dataset_name, batch_size=32, lr=0.0001):
    torch.random.manual_seed(42)
    dataset = load_dataset(dataset_name)

    task = dataset_name.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda")

    def rte_preprocess_function(examples):
        return tokenizer(examples["text1"], examples["text2"], truncation=True, padding=True, return_tensors="pt").to("cuda")
    
    def sst2_preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt").to("cuda")
    
    if task == "rte":
        preprocess_function = rte_preprocess_function
    elif task == "sst2":
        preprocess_function = sst2_preprocess_function

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    model_name = model_name.split("/")[-1]

    metric_name = "accuracy"
    metric = load_metric(metric_name)

    args = TrainingArguments(
        os.path.join(os.getcwd() + f"/models/lr{str(lr).replace('.', '_')}", model_name + "-finetuned-" + task),
        # os.path.join(output_dir, model_name + "-finetuned-" + task),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )


    all_preds = []
    all_labels = []
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        all_preds.extend(predictions.tolist())
        all_labels.extend(labels.tolist())
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    all_preds.clear()
    all_labels.clear()
    metrics = trainer.evaluate(encoded_dataset["test"])

    return metrics, all_preds, all_labels
    
def no_finetune(model_name, dataset_name, batch_size=32):
    torch.random.manual_seed(42)
    dataset = load_dataset(dataset_name)

    task = dataset_name.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to("cuda")

    def rte_preprocess_function(examples):
        return tokenizer(examples["text1"], examples["text2"], truncation=True, padding=True, return_tensors="pt").to("cuda")
    
    def sst2_preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt").to("cuda")
    
    if task == "rte":
        preprocess_function = rte_preprocess_function
    elif task == "sst2":
        preprocess_function = sst2_preprocess_function

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    model_name = model_name.split("/")[-1]

    metric_name = "accuracy"
    metric = load_metric(metric_name)

    args = TrainingArguments(
        os.path.join(os.getcwd() + f"/models", model_name + "-finetuned-" + task),
        # os.path.join(output_dir, model_name + "-finetuned-" + task),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )


    all_preds = []
    all_labels = []
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        all_preds.extend(predictions.tolist())
        all_labels.extend(labels.tolist())
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    metrics = trainer.evaluate(encoded_dataset["test"])

    return metrics, all_preds, all_labels
    
    