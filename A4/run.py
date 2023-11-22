from finetune import finetune_model, no_finetune

def main():
    bert_tiny = "prajjwal1/bert-tiny"
    bert_mini = "prajjwal1/bert-mini"
    rte_dataset = "yangwang825/rte"
    sst2_dataset = "gpt3mix/sst2"

    # tiny_rte_metrics = []
    # tiny_sst2_metrics = []
    # mini_rte_metrics = []
    # mini_sst2_metrics = []
    # for lr in [0.0001, 0.00001, 0.000001]:
    #     metrics, _, _ = finetune_model(bert_tiny, rte_dataset, lr=lr)
    #     tiny_rte_metrics.append(metrics)
    #     metrics, _, _ = finetune_model(bert_tiny, sst2_dataset, lr=lr)
    #     tiny_sst2_metrics.append(metrics)
    #     metrics, _, _ = finetune_model(bert_mini, rte_dataset, lr=lr)
    #     mini_rte_metrics.append(metrics)
    #     metrics, _, _ = finetune_model(bert_mini, sst2_dataset, lr=lr)
    #     mini_sst2_metrics.append(metrics)

    # print("BERT-Tiny RTE Metrics")
    # print(tiny_rte_metrics)
    # print("BERT-Tiny SST2 Metrics")
    # print(tiny_sst2_metrics)
    # print("BERT-Mini RTE Metrics")
    # print(mini_rte_metrics)
    # print("BERT-Mini SST2 Metrics")
    # print(mini_sst2_metrics)

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



    
if __name__ == "__main__":
    main()