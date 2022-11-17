import torch
from transformers import BertTokenizer, BertConfig
from transformers import Trainer, TrainingArguments
from transformers import get_scheduler
import json
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from models.bert_for_ner import BertCrfForNer
import numpy as np
# from sklearn.model_selection import KFold
from tools.finetuning_argparse import get_argparse
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train1(args, train_dataset, valid_dataset, model):
    """基于transformers trainarguments 和 trainer 来完成训练,简单快捷
    但是 Traingqrguments 里面的参数很多，文档里面也没有写全 所以后期调试很麻烦
    """
    training_args = TrainingArguments(
        output_dir='./outputs/robertav1',  # output directory
        num_train_epochs=args.epoch,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=30,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        # no_cuda=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset  # evaluation dataset
    )
    return trainer


def train2(args, train_dataset, model, device):

    train_data_load = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    # 设置损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # 设置一共有几步 epoch数量乘以每一个epoch里面的batch数量
    num_training_steps = args.epoch * len(train_data_load)
    # 设置学习策略
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=(num_training_steps*0.1),
        num_training_steps=num_training_steps)

    # 把模型加入cuda
    model.to(device=device)
    max_f1 = 0
    #训练循环
    for e in range(args.epoch):
        print("Number of EPOCH: ", e, "/", args.epoch)
        p_list = list()
        r_list = list()
        f1_list = list()
        for batch in tqdm(train_data_load):
            batch = {k: v.to(device) for k, v in batch.items()}
            model.train()
            outputs = model(**batch)
            logits = outputs[1]
            # 模型输出的参数进入crf解码得到结果
            crf_logits = model.crf.decode(logits, torch.tensor(batch['attention_mask'], dtype=torch.bool))
            # 模型预测结果
            predictions_np = np.asarray([np.asarray(i) for i in crf_logits])
            # 数据自己的结果
            labels_np = batch["labels"].cpu().numpy()
            # 计算precision 和 recall
            precision = np.asarray([precision_score(x[:len(y)], y, average='macro', zero_division=1) for x, y in zip(labels_np, predictions_np)])
            recall = np.asarray([recall_score(x[:len(y)], y, average='macro', zero_division=1) for x, y in zip(labels_np, predictions_np)])
            p_list.extend(precision)
            r_list.extend(recall)
            f1_list.extend((2 * precision * recall)/(precision + recall))
            if len(f1_list) > 0:
                result_f1 = np.mean(f1_list)
                result_p = np.mean(p_list)
                result_r = np.mean(r_list)
                print("result_f1: ", result_f1, " result_p: ", result_p, " result_r: ", result_r)
                loss = outputs[0] # 计算当前损失
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            """后面这里加valuation"""
        if max_f1 < result_f1:
            model.save_pretrained(args.output_dir)
            max_f1 = result_f1
            return None



# def evaluate():
#
# def test():
#

class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# def load_data2():
""" 用torch load的方法加载"""
    # loader = torch.utils.data.DataLoader(
    #     dataset=tokenized_tensor_data_input,  # torch TensorDataset format
    #     batch_size=args.batch_size,  # mini batch size
    #     shuffle=False,  # 要不要打乱数据 (打乱比较好)
    #     num_workers=2,  # 多线程来读数据
    # )
    #   kfold
    # kf = KFold()
    # kf.get_n_splits(ids)
    # fold5_index = [{'train': x, 'val': y} for x, y in kf.split(ids)]
    # return loader


def load_model1(path, id2label):
    """ from transformers from pertrained"""
    print(len(id2label.keys()))
    bert_config = BertConfig.from_pretrained(path, num_labels=len(id2label.keys()))
    model = BertCrfForNer.from_pretrained(path, config=bert_config)
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)

    return model, tokenizer


# def load_model2():
""" 用torch的方法 加载"""


def process_label(encodings, texts_long, labels, label2id, max_length):
    """
        处理labels，使所有label和经过padding和truncation的
    """
    labels_list_idx = []

    for index, (i, l) in enumerate(zip(encodings['input_ids'], labels)):
        labels_1 = [label2id['O']] * len(i)
        labels_1[0] = label2id['[CLS]']

        if l['investor']:
            for ii in l['investor']:
                labels_1[ii[1]+1] = label2id['B-INVESTOR']
                labels_1[ii[1]+2:ii[2]+1] = [label2id['I-INVESTOR']] * (ii[2] - ii[1] - 1)

        elif l['company']:
            for ii in l['company']:
                labels_1[ii[1]+1] = label2id['B-COMPANY']
                labels_1[ii[1]+2:ii[2]+1] = [label2id['I-COMPANY']] * (ii[2] - ii[1] - 1)
        labels_1[texts_long[index]+1] = label2id['[SEP]']
        labels_1[texts_long[index]+2:] = [0]*(max_length-2-texts_long[index])
        labels_list_idx.append(labels_1)
    return labels_list_idx


def data_process(args, tokenizer, label2id):
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
    # 制作label
    # entity = ['investor', 'company']
    # tokennize
    train_encodings = tokenizer(texts[:10], padding='max_length', truncation=True, max_length=125)
    texts_long = [len(i) for i in texts[:10]]
    labels = process_label(train_encodings, texts_long, labels_idx[:10],  label2id, args.max_length)
    return train_encodings, labels

    # transfer the label to [O,O,B_Person,xxx] 这种


def main():
    args = get_argparse().parse_args(['--do_train', '--data_dir', '/home/ner-clear/datasets/company-kg/entities_train.json',
                                      "--model_name_or_path", "/home/ner-clear/modelweights/roberta/", "--output_dir",
                                      "/home/ner-clear/outputs/robertav1", '--max_length', '125', '--epoch', '15', '--batch_size', '64'])
    id2label = {0: '[PAD]', 1: 'B-INVESTOR', 2: 'I-INVESTOR', 3: 'B-COMPANY', 4: 'I-COMPANY', 5: '[CLS]', 6: '[SEP]', 7: 'O'}
    label2id = {v: k for k, v in id2label.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # n_gpu = torch.cuda.device_count()
    model, tokenizer = load_model1(args.model_name_or_path, label2id)
    train_encodings, labels = data_process(args, tokenizer, label2id)

    # 用训练数据的前1000个条数据进行验证
    valid_encodings = dict()
    valid_encodings['input_ids'] = train_encodings['input_ids'][:1000]
    valid_encodings['token_type_ids'] = train_encodings['token_type_ids'][:1000]
    valid_encodings['attention_mask'] = train_encodings['attention_mask'][:1000]
    vali_labels = labels[:1000]

    # 转成tensor
    train_dataset = TaskDataset(train_encodings, labels)
    valid_dataset = TaskDataset(valid_encodings, vali_labels)

    # training1
    # trainer = train1(args, train_dataset, valid_dataset, model)
    # trainer.train()
    # trainer.save_model(args.output_dir)

    # training2
    train2(args, train_dataset, model, device)




if __name__ == "__main__":
    main()