from re import A
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def eval_csv(filepath, checkpoint_path):
    df = pd.read_csv(filepath)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    if "text1" in df.columns:
        task = "rte"
    elif "text" in df.columns:
        task = "sst2"
    
    preds = []
    prob_0 = []
    prob_1 = []
    for index, row in df.iterrows():
        if task == "rte":
            inputs = tokenizer(row["text1"], row["text2"], truncation=True, padding=True, return_tensors="pt")
        elif task == "sst2":
            inputs = tokenizer(row["text"], truncation=True, padding=True, return_tensors="pt")
        
        outputs = model(**inputs)
        probs = outputs.logits.softmax(-1).detach().numpy()
        prob_0.append(probs[0][0])
        prob_1.append(probs[0][1])
        preds.append(outputs.logits.argmax(-1).item())

    df["preds"] = preds
    df["prob_0"] = prob_0
    df["prob_1"] = prob_1

    df.to_csv(filepath.split(".")[0] + "_eval.csv", index=False)

def eval_rte(text1, text2, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    inputs = tokenizer(text1, text2, truncation=True, padding=True, return_tensors="pt")

    outputs = model(**inputs)
    print(outputs.logits.softmax(-1).detach().numpy())

    return(outputs.logits.argmax(-1).item())

def eval_sst2(text, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

    outputs = model(**inputs)

    print(outputs.logits.softmax(-1).detach().numpy())

    return(outputs.logits.argmax(-1).item())

    
    