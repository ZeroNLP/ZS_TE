import sys
import os
from pathlib import Path
import torch.nn
import pickle

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())

sys.path.append(CURR_DIR)
P = PATH.parent
print("current dir: ", CURR_DIR)
for i in range(1):  # add parent path, height = 3
    P = P.parent
    PROJECT_PATH = str(P.absolute())
    sys.path.append(str(P.absolute()))

from torch.utils.data import Dataset
import torch.utils.data as util_data
import torch
from collections import Counter
import random
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DebertaV2ForSequenceClassification
import transformers
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class MNLIInputFeatures:
    premise: str
    hypothesis: str
    label: int
    # {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}

labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}

def set_global_random_seed(seed):
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def load(path_name: object) -> object:
    with open(path_name,'rb') as file:
        return pickle.load(file)
def save(obj,path_name):
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)
def multi_acc(y_pred, y_true):
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_true).sum().float() / float(y_true.size(0))
    return acc

class mnli_data(Dataset):
    def __init__(self, texts, labels, probs) -> object:
        self.texts = texts
        self.labels = labels
        self.probs = probs
        print("各类数量:", Counter(self.labels))
        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'texts': self.texts[idx],
            'labels': self.labels[idx],
            'probs': self.probs[idx]}

import torch.nn.functional as F


def fine_tune_v3(args):
    print(args)
    set_global_random_seed(args.seed)
    device = torch.device("cuda:{}".format(args.cuda_index))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=3).to(device)
    # if args.load_weight:
    #     print("load weight ", args.model_weight_path)
    #     model.load_state_dict(torch.load(args.model_weight_path))
    # model = torch.nn.DataParallel(model,device_ids=[0,1,2]).cuda()
    # model = model.module
    args.check_point_path = "/data/rywu/prob_te_model/raw_data_model.pt"

    # 加载数据
    data = load(args.train_path)
    args.data_num = len(data) // 3
    random.shuffle(data)  # 打乱
    texts = [f"{item.premise} {tokenizer.sep_token} {item.hypothesis}." for item in data]
    labels = [item.label for item in data]
    probs  = [item.prob for item in data]

    n_dev = int(len(data) * 0.2)  # dev数量

    # 划分dev, train, 创建data_loader
    dev_dataset = mnli_data(texts[:n_dev], labels[:n_dev], probs[:n_dev])
    train_dataset = mnli_data(texts[n_dev:], labels[n_dev:], probs[n_dev:])

    dev_loader = util_data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # create optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)  # 学习率一定要小 4e-7
    print("warm up steps:", int(args.ratio * 100 * 40))
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=int(args.ratio * 100 * 40),num_training_steps=args.epoch * (len(train_loader)))

    # train
    model.train()
    # saved_paths = ["/data/rywu/2022_5_24_mnli_fine_tune"]
    base_acc = -1

    for epoch in range(args.epoch):
        total_step = len(train_loader)
        accs = []
        losses = []
        for batch_idx, batch in enumerate(train_loader):
            text = batch['texts']
            label = batch['labels']
            prob = batch['probs']

            f_prob = []
            for value in prob:
                f_prob.append(float(value))
            ver_prob = []
            for value in prob:
                ver_prob.append((1-float(value))/2)

            input_ids = tokenizer.batch_encode_plus(text, padding=True, truncation=True, max_length=128)
            # print(len(text))
            input_ids = torch.tensor(input_ids["input_ids"]).to(device)
            out = model(input_ids, labels=label.to(device))
            # loss = out[0]
            # print(out)
            # print(loss)
            loss_fct = torch.nn.CrossEntropyLoss()

            prediction = out[1]
            # loss1 = loss_fct(prediction, label.to(device))
            target = F.one_hot(label, 3)  # 转换成one-hot
            _w_tensor =  torch.Tensor(ver_prob)
            old_tensor = target.float()
            new_tensor = torch.Tensor(f_prob)
            # 找到旧张量中为1的位置索引
            indices = torch.where(old_tensor == 1)
            # 使用索引操作和赋值操作将新值替换旧张量中的1
            old_tensor[indices] = new_tensor

            for i in range(old_tensor.size(0)):
                # 找到当前行中为0的位置索引
                indices = torch.where(old_tensor[i] == 0)
                # 使用索引操作和赋值操作将新值替换当前行中的0
                old_tensor[i][indices] = _w_tensor[i]

            te_label = old_tensor.to(device)

            # loss = loss_fct(prediction, label.to(device))
            te_loss = loss_fct(prediction, te_label.to(device))
            loss = te_loss
            # loss_fct1 = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            # loss3 = loss_fct1(prediction, label.to(device))

            print(loss)
            print(epoch)
            print()

            loss.backward()

            lr = optimizer.state_dict()['param_groups'][0]['lr']
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # 记录
            acc = multi_acc(prediction.detach().cpu(), label)
            accs.append(acc.item())
            losses.append(loss.detach().cpu().item())
            print('\r'," step {}/{} ,  loss_{:.4f} acc_{:.4f}  lr:{}".format(batch_idx + 1, total_step, np.mean(losses),np.mean(accs), lr), end='', flush=True)

        train_acc = np.mean(accs)
        train_loss = np.mean(losses)

        # evaluation
        with torch.no_grad():
            dev_accs = []
            for batch_idx, batch in enumerate(dev_loader):
                text = batch['texts']
                label = batch['labels']
                input_ids = tokenizer.batch_encode_plus(text, padding=True, truncation=True, max_length=128)
                # print(len(text))
                input_ids = torch.tensor(input_ids["input_ids"]).to(device)
                prediction = model(input_ids, labels=label.to(device))[-1]
                acc = multi_acc(prediction.detach().cpu(), label)
                dev_accs.append(acc.detach().cpu().item())
            dev_acc = np.mean(dev_accs)

        # save by acc (or f1)
        if dev_acc > base_acc:
            base_acc = dev_acc
            torch.save(model.state_dict(), args.check_point_path)

        print(f'Epoch {epoch + 1}: train_loss: {train_loss:.5f} train_acc: {train_acc:.5f}  dev_acc: {dev_acc:.5f}')

if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='as named')
    parser.add_argument('--cuda_index', type=int, default=0, help='as named')
    parser.add_argument('--epoch', type=int, default=10, help='as named')
    parser.add_argument('--seed', type=int, default=16, help='as named')
    parser.add_argument('--ratio', type=float, default=0.07, help='as named')
    parser.add_argument('--dataset', type=str, help='as named')
    parser.add_argument('--lr', type=float, default=4e-7,help='learning rate')
    parser.add_argument('--train_path', type=str,default="/data/rywu/raw_data_prob_plus_value.pickle")
    parser.add_argument('--model_path', type=str, default="/data/transformers/microsoft_deberta-v2-xlarge-mnli")
    # parser.add_argument('--load_weight', type=str2bool, default=False)  # few-shot 模式需要加载之前finetune后的模型
    # parser.add_argument('--model_weight_path', type=str, default="",help="this is for few-shot")

    args = parser.parse_args()
    fine_tune_v3(args)
    debug_stop = 1