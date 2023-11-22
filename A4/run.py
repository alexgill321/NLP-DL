from finetune import finetune_model, no_finetune
from process import eval_csv, eval_rte, eval_sst2

def main():
    bert_tiny = "prajjwal1/bert-tiny"
    bert_mini = "prajjwal1/bert-mini"
    rte_dataset = "yangwang825/rte"
    sst2_dataset = "gpt3mix/sst2"

    tiny_rte_metrics = []
    tiny_sst2_metrics = []
    mini_rte_metrics = []
    mini_sst2_metrics = []
    for lr in [0.0001, 0.00001, 0.000001]:
        metrics, _, _ = finetune_model(bert_tiny, rte_dataset, lr=lr)
        tiny_rte_metrics.append(metrics)
        metrics, _, _ = finetune_model(bert_tiny, sst2_dataset, lr=lr)
        tiny_sst2_metrics.append(metrics)
        metrics, _, _ = finetune_model(bert_mini, rte_dataset, lr=lr)
        mini_rte_metrics.append(metrics)
        metrics, _, _ = finetune_model(bert_mini, sst2_dataset, lr=lr)
        mini_sst2_metrics.append(metrics)

    print("BERT-Tiny RTE Metrics")
    print(tiny_rte_metrics)
    print("BERT-Tiny SST2 Metrics")
    print(tiny_sst2_metrics)
    print("BERT-Mini RTE Metrics")
    print(mini_rte_metrics)
    print("BERT-Mini SST2 Metrics")
    print(mini_sst2_metrics)

    acc, _, _ = no_finetune(bert_tiny, rte_dataset)
    print("BERT-Tiny RTE Accuracy")
    print(acc)
    acc, _, _ = no_finetune(bert_tiny, sst2_dataset)
    print("BERT-Tiny SST2 Accuracy")
    print(acc)
    acc, _, _ = no_finetune(bert_mini, rte_dataset)
    print("BERT-Mini RTE Accuracy")
    print(acc)
    acc, _, _ = no_finetune(bert_mini, sst2_dataset)
    print("BERT-Mini SST2 Accuracy")
    print(acc)

    eval_csv("hidden_rte.csv", "models/lr1e-05/bert-mini-finetuned-rte/checkpoint-780")
    eval_csv("hidden_sst2.csv", "models/lr0_0001/bert-mini-finetuned-sst2/checkpoint-2170")

    print(eval_rte("The doctor is prescribing medicine.", "She is prescribing medicine.", "models/lr1e-05/bert-mini-finetuned-rte/checkpoint-780"))
    print(eval_rte("The doctor is prescribing medicine.", "He is prescribing medicine.", "models/lr1e-05/bert-mini-finetuned-rte/checkpoint-780"))
    print(eval_rte("The nurse is tending to the patient.", "She is tending to the patient.", "models/lr1e-05/bert-mini-finetuned-rte/checkpoint-780"))
    print(eval_rte("The nurse is tending to the patient.", "He is tending to the patient.", "models/lr1e-05/bert-mini-finetuned-rte/checkpoint-780"))

    print(eval_sst2("Kate should get promoted, she is an amazing employee.", "models/lr0_0001/bert-mini-finetuned-sst2/checkpoint-2170"))
    print(eval_sst2("Bob should get promoted, he is an amazing employee.", "models/lr0_0001/bert-mini-finetuned-sst2/checkpoint-2170"))
    print(eval_sst2("Kate should get promoted, he is an amazing employee.", "models/lr0_0001/bert-mini-finetuned-sst2/checkpoint-2170"))
    print(eval_sst2("Bob should get promoted, they are an amazing employee.", "models/lr0_0001/bert-mini-finetuned-sst2/checkpoint-2170"))

if __name__ == "__main__":
    main()