import torch
from transformers import BertTokenizer, BertConfig
# from transformers import Trainer, TrainingArguments
import json
from models.bert_for_ner import BertCrfForNer
# import numpy as np
# from sklearn.model_selection import KFold
from tools.finetuning_argparse import get_argparse

def predict(model, predict_encodings):
    model.eval()
    with torch.no_grad():
        outputs = model(**predict_encodings)
        logits = outputs[0]
        tags = model.crf.decode(logits, torch.tensor(predict_encodings['attention_mask'], dtype=torch.bool))
        return tags



def data_process(args, tokenizer):
    # 所有关于输入的数据处理都在这里进行
    ids = []
    texts = []
    labels_idx = []

    # loading the data and appending ids texts and labels to lists
    with open(args.data_dir, 'r', encoding='utf-8') as f:
        for i in f.readlines():
            i = json.loads(i)
            ids.append(i['id'])
            texts.append(i['text'])
            labels_idx.append(i['label'])

    # tokennize
    train_encodings = tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=125)
    return train_encodings, texts


def load_model1(path, id2label):
    """ from transformers from pertrained"""

    bert_config = BertConfig.from_pretrained(path, num_labels=len(id2label.keys()))
    model = BertCrfForNer.from_pretrained(path, config=bert_config)
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)

    return model, tokenizer


def main():
    args = get_argparse().parse_args(['--do_predict', '--data_dir', '/home/ner-clear/datasets/company-kg/entities_test.json',
                                      "--model_name_or_path", "/home/ner-clear/outputs/robertav2", "--output_dir",
                                      "/home/ner-clear/outputs/robertav2", '--max_length', '125', '--epoch', '15', '--batch_size', '64'])
    id2label = {0: '[PAD]', 1: 'B-INVESTOR', 2: 'I-INVESTOR', 3: 'B-COMPANY', 4: 'I-COMPANY', 5: '[CLS]', 6: '[SEP]', 7: 'O'}
    label2id = {v: k for k, v in id2label.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # n_gpu = torch.cuda.device_count()
    model, tokenizer = load_model1(args.model_name_or_path, label2id)
    predict_encodings, texts = data_process(args, tokenizer)
    tags = predict(model, predict_encodings)
    tages = [[id2label[tt] for tt in t] for t in tags]

    for t1, t2 in zip(texts,tages):
        print('text: ', t1, ' label: ', t2)

   # transfer to label








if __name__ == "__main__":
    main()