import json
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from transformers import BertForMaskedLM
from transformers import logging
logging.set_verbosity_error()

from dataset import MyDataset
from torch.utils.data import Dataset

class IEDataset(Dataset):
    def __init__(self, path, max_length=128, gpu=False, ignore_title=False):
        self.path = path
        self.data = []
        self.gpu = gpu
        self.max_length = max_length
        self.ignore_title = ignore_title
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    def load_data(self):
        """Load data from file."""
        overlength_num = title_num = 0
        with open(self.path, 'r', encoding='utf-8') as r:
            for line in r:
                inst = json.loads(line)
                is_title = inst['sent_id'].endswith('-3') and inst['tokens'][-1] != '.'
                if self.ignore_title and is_title:
                    title_num += 1
                    continue

                # TODO: add back coarse type
                for event in inst['event_mentions']:
                    event_type = event['event_type']
                    if ':' in event_type:
                        event['event_type'] = event_type.split(':')[1].upper()
                self.data.append(inst)

        if title_num:
            print('Discarded {} titles'.format(title_num))
        print('Loaded {} instances from {}'.format(len(self), self.path))
dev_dataset = IEDataset("/data/rywu/Dev_TC_2022_10_30_model_0.99.json")
test_dataset = IEDataset("/data/rywu/Test_TC_2022_10_30_model_0.99.json")

def type_format_transfer(input):
    if input == "BE-BORN":
        return "BE_BORN"
    if input == "TRANSFER-OWNERSHIP":
        return "TRANSFER_OWNERSHIP"
    if input == "TRANSFER-MONEY":
        return "TRANSFER_MONEY"
    if input == "START-ORG":
        return "START_ORG"
    if input == "MERGE-ORG":
        return "MERGE_ORG"
    if input == "DECLARE-BANKRUPTCY":
        return "DECLARE_BANKRUPTCY"
    if input == "END-ORG":
        return "END_ORG"
    if input == "PHONE-WRITE":
        return "PHONE_WRITE"
    if input == "START-POSITION":
        return "START_POSITION"
    if input == "END-POSITION":
        return "END_POSITION"
    if input == "ARREST-JAIL":
        return "ARREST_JAIL"
    if input == "RELEASE-PAROLE":
        return "RELEASE_PAROLE"
    if input == "TRIAL-HEARING":
        return "TRIAL_HEARING"
    if input == "CHARGE-INDICT":
        return "CHARGE_INDICT"
    else:
        return input
def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
  file = open(filename,'w')
  for i in range(len(data)):
    s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
    s = s.replace("'",'').replace(',','') +'\n'  #去除单引号，逗号，每行末尾追加换行符
    file.write(s)
  file.close()
def safe_div(num, denom):
    if denom > 0:
        if num / denom <= 1:
            return num / denom
        else:
            return 1
    else:
        return 0
def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

def evaluate(model, device, dataloader, dataset):
    model_path = "/data/transformers/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_path)

    all_tigger_pred_list = []
    all_tigger_second_pred_list = []
    all_tigger_third_pred_list = []

    prob_value1 = []
    prob_value2 = []
    prob_value3 = []

    near_word_list = []

    data_analysis = []

    new_tokens = ["BE_BORN", "TRANSFER_OWNERSHIP", "TRANSFER_MONEY", "START_ORG", "MERGE_ORG",
                  "DECLARE_BANKRUPTCY", "END_ORG", "PHONE_WRITE", "START_POSITION", "END_POSITION",
                  "ARREST_JAIL", "RELEASE_PAROLE","TRIAL_HEARING","CHARGE_INDICT", "EXTRADITE",
                  "NOT_MENTION", "crumble", "injure", "acquit", "acquittal", "indict", "extradition",
                  "no_trigger", "trigger", "[event]"] # "re-election", "by-elections"

    lower_new_tokens = []
    for word in new_tokens:
        lower_new_tokens.append(word.lower())
    all_trigger_list = []

    with open("/data/rywu/Train_TC.pickle", 'rb') as file:
        dic = pickle.load(file)
        for line in dic:
            if line["sentence_type"] != "NOT_MENTION":
                if "real_sentence_keyword" not in line:
                    if line["kind_number"] == 1:
                        if line["sentence_keyword"] not in all_trigger_list:
                            all_trigger_list.append(line["sentence_keyword"])
                    else:
                        for keyword in line["sentence_keyword"]:
                            if keyword not in all_trigger_list:
                                all_trigger_list.append(keyword)
        all_trigger_list.append("no_trigger")

    all_list = all_trigger_list + lower_new_tokens

    num_added_tokens = tokenizer.add_tokens(all_list)
    print('We have added', num_added_tokens, 'tokens')
    model.resize_token_embeddings(len(tokenizer))

    gold_trigger_num = 0
    pred_trigger_num = 0
    match_tigger_num = 0

    gold_type_num = 0
    pred_type_num = 0
    match_type_num = 0

    for batch in dataloader:
        token_ids = batch['input_ids'].to(device)
        masked_index = batch['masked_index'].to(device)
        original_text = batch['original_text']
        model.eval()

        with torch.no_grad():
            outputs = model(token_ids)
            predictions = outputs[0]
            predictions1 = predictions[:, masked_index[0], :]  # (batch_size,vocab_size)

            for i, a in enumerate(token_ids) :
                index = [i for i, val in enumerate(a.tolist()) if val == 102]
                end = index[1]-1
                word_list = a[index[0] : end]
                text = original_text[i]
                sentence_word_list = []
                for j, part in enumerate(word_list.tolist()):
                    if j > 0:
                        sentence_word_list.append(part)

                all_word_space = sentence_word_list + [tokenizer.convert_tokens_to_ids("no_trigger")]
                word_list = torch.tensor(all_word_space)

                # sentence_token_id= []
                # for word in sentence_word_list:
                #     if word != 0:
                #         sentence_token_id.append(tokenizer.convert_ids_to_tokens(word))

                logits = predictions1[i, torch.max(torch.zeros_like(word_list), word_list)]  # (batch_size, n_classes, 1)
                n = word_list.size()[0]
                logits = logits.view(-1, n)
                probs = F.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).detach()

                all_tigger_pred_list.append(tokenizer.convert_ids_to_tokens(word_list[pred.item()].item()))
                prob_value1.append(round(probs.tolist()[0][pred.item()], 3))

                if len(word_list) > 1:
                    probs_ls = probs.tolist()[0]
                    del(probs_ls[pred.item()])
                    del(all_word_space[pred.item()])
                    second_prob_pred = probs_ls.index(max(probs_ls))

                    # new_probs1 = new_probs.remove(second_prob_pred.item())
                    # new_word2 = []
                    # third_prob_pred =  torch.argmax(torch.tensor(new_probs1), dim=-1).detach()

                    all_tigger_second_pred_list.append(tokenizer.convert_ids_to_tokens(all_word_space[second_prob_pred]))
                    prob_value2.append(round(probs_ls[second_prob_pred], 3))

                    # all_tigger_third_pred_list.append(tokenizer.convert_ids_to_tokens(word_list[pred.item()].item()))
                    # prob_value3.append(round(new_probs1[pred.item()], 3))
                else:
                    all_tigger_second_pred_list.append([])
                    prob_value2.append([])

                if len(word_list) > 1:
                    if pred.item() == 0:
                        near_word_list.append(tokenizer.convert_ids_to_tokens(word_list[pred.item() + 1].item()))
                    elif pred.item() == len(word_list)-1:
                        near_word_list.append(tokenizer.convert_ids_to_tokens(word_list[pred.item() - 1].item()))
                    else:
                        near_word = []
                        near_word.append(tokenizer.convert_ids_to_tokens(word_list[pred.item() - 1].item()))
                        near_word.append(tokenizer.convert_ids_to_tokens(word_list[pred.item() + 1].item()))
                        near_word_list.append(near_word)
                else:
                    near_word_list.append([])

    pred_tigger_list = []
    a = 0
    for line in tqdm(dataset):
        if line["pred_event_type"] == []:
            line["pred_trigger"] = all_tigger_pred_list[a : a + 1]
            line["pred_trigger_prob"] = prob_value1[a: a + 1]
            line["near_word"] = near_word_list[a : a + 1]
            line["second_pred_trigger"] = all_tigger_second_pred_list[a : a + 1]
            line["second_pred_trigger_prob"] = prob_value2[a: a + 1]
            pred_tigger_list.append(line)
            a += 1
        else:
            line["pred_trigger"] = all_tigger_pred_list[a : a + len(line["pred_event_type"])]
            line["pred_trigger_prob"] = prob_value1[a: a + len(line["pred_event_type"])]
            line["near_word"] = near_word_list[a : a + len(line["pred_event_type"])]
            line["second_pred_trigger"] = all_tigger_second_pred_list[a : a + len(line["pred_event_type"])]
            line["second_pred_trigger_prob"] = prob_value2[a: a + len(line["pred_event_type"])]
            pred_tigger_list.append(line)
            a += len(line["pred_event_type"])

    all_gold_ls = []
    all_pred_ls = []

    all_type_list = ["BE_BORN", "MARRY", "DIVORCE", "INJURE", "DIE", "TRANSPORT", "TRANSFER_OWNERSHIP",
                     "TRANSFER_MONEY", "START_ORG", "MERGE_ORG", "DECLARE_BANKRUPTCY", "END_ORG", "ATTACK",
                     "DEMONSTRATE", "MEET", "PHONE_WRITE", "START_POSITION", "END_POSITION", "NOMINATE",
                     "ELECT", "ARREST_JAIL", "RELEASE_PAROLE", "TRIAL_HEARING", "CHARGE_INDICT", "SUE",
                     "CONVICT", "SENTENCE", "FINE", "EXECUTE", "EXTRADITE", "ACQUIT", "PARDON", "APPEAL"]

    for t in all_type_list:
        locals()['number_' + str(t)] = 0
        all_pred_ls.append(locals()['number_' + str(t)])

    for t in all_type_list:
        locals()['gold_number_' + str(t)] = 0
        all_gold_ls.append(locals()['gold_number_' + str(t)])

    part_match_tigger_num = 0
    for number, line in enumerate(dataset):
        dic = {}
        dic["number"] = number

        gold_type = list()
        pred_type = list()

        gold_trigger = list()
        pred_trigger = list()
        near_word = []
        pred_trigger_prob = []
        second_pred_trigger = []
        second_pred_trigger_prob = []

        for event in line["event_mentions"]:
            gold_type.append(event["event_type"])
            gold_trigger.append(event["trigger"]["text"].lower())

        error_trigger_analysis = ["no_trigger", ",", ".", "!", "\"", "-", "'", "$", "in", "on", "with", "by", "for", "at", "about", "under", "of", "to", "into", ":", "the", '`', ';']

        if line["pred_event_type"] != []:
            for i,j,k,w,s,z in zip(line["pred_event_type"], line["pred_trigger"], line["near_word"], line["pred_trigger_prob"], line["second_pred_trigger"], line["second_pred_trigger_prob"]):
                if j not in error_trigger_analysis:
                    pred_type.append(i["event_type"])
                    pred_trigger.append(j)
                    near_word.append(k)
                    pred_trigger_prob.append(w)
                    second_pred_trigger.append(s)
                    second_pred_trigger_prob.append(z)
                dic["pred_type"] = pred_type

        else:
            dic["pred_type"] = []
            pred_trigger = []

        dic["sentence"] = line["sentence"]
        dic["gold_type"] = gold_type
        dic["gold_trigger"] = gold_trigger
        dic["pred_trigger"] = pred_trigger
        data_analysis.append(dic)

        gold_type_num += len(gold_type)
        pred_type_num += len(pred_type)

        gold_trigger_num += len(gold_trigger)
        pred_trigger_num += len(pred_trigger)

        for i in gold_type:
            all_gold_ls[all_type_list.index(type_format_transfer(i))] += 1
            if i in pred_type:
                match_type_num += 1

        diff_trigger = []

        phrase_ls = ['head over', 'pushed onward', 'pushed forward', 'punched through', 'air strikes', 'desert storm', 'walked out', 'head out',
                     'going down', 'raping and drugging', 'took on', 'bring in', 'steps down', 'blowing up', 'chopping off', 'taking money',
                     'raising money', 'steps down', 'took office', 'stepped down', 'set up', 'took office', 'take over', 'take over', 'aid package',
                     'swept out of power', 'buying out', 'dip into', 'be out', 'step down', 'step aside', 'wiped out', 'smash through']

        for i in gold_type:
            if i not in diff_trigger:
                index = [j for j, val in enumerate(gold_type) if val == i]
                diff_trigger.append(i)
                if i in pred_type:
                    if len(index) == 1:
                        if pred_trigger[pred_type.index(i)] in gold_trigger[gold_type.index(i)]:
                            if gold_trigger[gold_type.index(i)] not in phrase_ls:
                                match_tigger_num += 1
                                all_pred_ls[all_type_list.index(type_format_transfer(i))] += 1

                    else:
                        for dex in index:
                            if pred_trigger[pred_type.index(i)] in gold_trigger[dex]:
                                if gold_trigger[dex] not in phrase_ls:
                                    match_tigger_num += 1
                                    all_pred_ls[all_type_list.index(type_format_transfer(i))] += 1


    with open("/data/rywu/2022_10_18/" + "{}".format(dataset) + "_json_data.json", 'w') as f:
        for data in data_analysis:
            back_json = json.dumps(data)
            f.write(back_json + '\n')
            f.flush()

    type_precision, type_recall, type_f1 = compute_f1(pred_type_num, gold_type_num, match_type_num)
    trigger_precision, trigger_recall, trigger_f1 = compute_f1(pred_trigger_num, gold_trigger_num, match_tigger_num)

    return  pred_type_num, gold_type_num, match_type_num, pred_trigger_num, gold_trigger_num, match_tigger_num, \
            trigger_precision, trigger_recall, trigger_f1, pred_tigger_list, type_precision, type_recall, type_f1, all_type_list, all_gold_ls, all_pred_ls

def train():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_path = "/data/transformers/bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)
    model.to(device)

    new_tokens = ["BE_BORN", "TRANSFER_OWNERSHIP", "TRANSFER_MONEY", "START_ORG", "MERGE_ORG",
                  "DECLARE_BANKRUPTCY", "END_ORG", "PHONE_WRITE", "START_POSITION", "END_POSITION",
                  "ARREST_JAIL", "RELEASE_PAROLE","TRIAL_HEARING","CHARGE_INDICT", "EXTRADITE",
                  "NOT_MENTION", "crumble", "injure", "acquit", "acquittal", "indict", "extradition",
                  "no_trigger", "trigger", "[event]"] # "re-election", "by-elections"

    lower_new_tokens = []
    for word in new_tokens:
        lower_new_tokens.append(word.lower())
    all_trigger_list = []

    with open("/data/rywu/2022_12_15_proportion_data.pickle", 'rb') as file:
        dic = pickle.load(file)
        for line in dic:
            if line["sentence_type"] != "NOT_MENTION":
                if "real_sentence_keyword" not in line:
                    if line["kind_number"] == 1:
                        if line["sentence_keyword"] not in all_trigger_list:
                            all_trigger_list.append(line["sentence_keyword"])
                    else:
                        for keyword in line["sentence_keyword"]:
                            if keyword not in all_trigger_list:
                                all_trigger_list.append(keyword)
        all_trigger_list.append("no_trigger")

    all_list = all_trigger_list + lower_new_tokens

    num_added_tokens = tokenizer.add_tokens(all_list)
    print('We have added', num_added_tokens, 'tokens')
    model.resize_token_embeddings(len(tokenizer))

    label2idx_tensor = torch.ones([len(all_trigger_list), 1], dtype=torch.long) * -1

    for idx, token in enumerate(all_trigger_list):
        index = tokenizer.convert_tokens_to_ids(token)
        label2idx_tensor[idx, 0] = index

    data_dir = "/data/rywu/"
    save_dir = "/data/rywu/2022_bert_model_Mask_part_output"

    log_dir = "/data/rywu/2022_bert_model_Mask_part_output/log/prompt-learning"
    log_file = "agnews_pretrain_hyper_params.txt"

    train_dir = os.path.join("/data/rywu/Train_TC.pickle")
    test_dir = os.path.join("/data/rywu/Test_TC.pickle")
    dev_dir = os.path.join("/data/rywu/Dev_TC.pickle")

    batch_size = 16
    learning_rate = 1e-5
    epochs = 1
    eval_period = 10

    train_set = MyDataset(train_dir, tokenizer)
    test_set = MyDataset(test_dir, tokenizer)
    dev_set = MyDataset(dev_dir, tokenizer)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_set, batch_size=batch_size)

    print('device', device)
    print('Epoch', epochs)
    print('model', model_path)

    best_dev_precision, best_dev_recall, best_dev_f1 = 0., 0., 0.
    best_epoch = 0
    best_global_step = 0
    global_step = 0

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    weight_decay = 0.01
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    t_total = len(train_dataloader) * epochs
    warmup_steps = 0

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    for epoch in range(1, epochs + 1):

        for batch in tqdm(train_dataloader):
            model.train()
            global_step += 1

            token_ids = batch['input_ids'].to(device)
            masked_index = batch['masked_index'].to(device)
            trigger = batch['trigger'].to(device)
            label2idx_tensor = label2idx_tensor.to(device)

            # for i, a in enumerate(token_ids) :
            #     print(tokenizer.convert_ids_to_tokens(a.tolist()[6]))

            outputs = model(token_ids)
            predictions = outputs[0]
            predictions1 = predictions[:, masked_index[0], :]  # (batch_size,vocab_size)

            logits = predictions1[:, torch.max(torch.zeros_like(label2idx_tensor), label2idx_tensor)] # (batch_size, n_classes, 1)
            n = label2idx_tensor.size()[0]
            logits = logits.view(-1, n)

            loss = nn.CrossEntropyLoss()(logits, trigger)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % eval_period == 0:
                print('Start Evaluating ...')
                dev_pred_type_num, dev_gold_type_num, dev_match_type_num, dev_pred_trigger_num, dev_gold_trigger_num, dev_match_tigger_num, dev_trigger_precision, dev_trigger_recall, dev_trigger_f1, pred_trigger_list, type_precision, type_recall, \
                type_f1, dev_all_type_list, dev_all_gold_ls, dev_all_pred_ls = evaluate(model, device, dev_dataloader, dev_dataset)

                test_pred_type_num, test_gold_type_num, test_match_type_num, test_pred_trigger_num, test_gold_trigger_num, test_match_tigger_num, test_trigger_precision, test_trigger_recall, test_trigger_f1, test_pred_trigger_list, test_type_precision, test_type_recall,\
                test_type_f1, test_all_type_list, test_all_gold_ls, test_all_pred_ls = evaluate(model, device, test_dataloader, test_dataset)

                print('Test\n')
                print('test_type_Precision {:.5f}, test_type_Recall {:.5f}, test_type_F1 {:.5f}'.format(test_type_precision, test_type_recall, test_type_f1))
                print('test_trigger_Precision {:.5f}, test_trigger_Recall {:.5f}, test_trigger_F1 {:.5f}'.format(test_trigger_precision, test_trigger_recall, test_trigger_f1))
                print(test_pred_type_num, test_gold_type_num, test_match_type_num, test_pred_trigger_num, test_gold_trigger_num, test_match_tigger_num)
                print()

                print('Epoch {} Step {}\n'.format(epoch, global_step))

                print('Dev\n')
                print('dev_type_Precision {:.5f}, dev_type_Recall {:.5f}, dev_type_F1 {:.5f}'.format(type_precision, type_recall, type_f1))

                print('dev_trigger_Precision {:.5f}, dev_trigger_Recall {:.5f}, dev_trigger_F1 {:.5f}'.format(dev_trigger_precision, dev_trigger_recall, dev_trigger_f1))
                print(dev_pred_type_num, dev_gold_type_num, dev_match_type_num, dev_pred_trigger_num, dev_gold_trigger_num, dev_match_tigger_num)
                print()

                if dev_trigger_f1 > best_dev_f1:
                    best_epoch = epoch
                    best_global_step = global_step
                    best_dev_f1 = dev_trigger_f1

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    filename = os.path.join(save_dir, "model_pretrain_{}.pt".format(global_step))
                    torch.save(model.state_dict(), filename)
                    with open(os.path.join(save_dir, "record"), 'w') as fo:
                        fo.write("{}\n".format(global_step))

    model.load_state_dict(torch.load(filename))
    model.eval()

    dev_pred_type_num, dev_gold_type_num, dev_match_type_num, dev_pred_trigger_num, dev_gold_trigger_num, dev_match_tigger_num, dev_trigger_precision, dev_trigger_recall, dev_trigger_f1, dev_pred_trigger_list, \
    dev_type_precision, dev_type_recall, dev_type_f1, dev_all_type_list, dev_all_gold_ls, dev_all_pred_ls = evaluate(model, device, dev_dataloader, dev_dataset)

    test_pred_type_num, test_gold_type_num, test_match_type_num, test_pred_trigger_num, test_gold_trigger_num, test_match_tigger_num, test_trigger_precision, test_trigger_recall, test_trigger_f1, test_pred_trigger_list,\
    test_type_precision, test_type_recall, test_type_f1, test_all_type_list, test_all_gold_ls, test_all_pred_ls = evaluate(model, device, test_dataloader, test_dataset)

    print('Best Epoch {} Step {}\n'.format(best_epoch, best_global_step))

    print('Dev\n')
    print('best_dev_type_Precision {:.5f}, best_dev_type_Recall {:.5f}, best_dev_type_F1 {:.5f}'.format(
        dev_type_precision, dev_type_recall, dev_type_f1))

    print('best_dev_trigger_Precision {:.5f}, best_dev_trigger_Recall {:.5f}, best_dev_trigger_F1 {:.5f}'.format(
        dev_trigger_precision, dev_trigger_recall, dev_trigger_f1))
    print(dev_pred_type_num, dev_gold_type_num, dev_match_type_num, dev_pred_trigger_num, dev_gold_trigger_num,
          dev_match_tigger_num)
    print()

    print('Test\n')
    print('best_test_type_Precision {:.5f}, best_test_type_Recall {:.5f}, best_test_type_F1 {:.5f}'.format(
        test_type_precision, test_type_recall, test_type_f1))

    print('best_test_trigger_Precision {:.5f}, best_test_trigger_Recall {:.5f}, best_test_trigger_F1 {:.5f}'.format(
        test_trigger_precision, test_trigger_recall, test_trigger_f1))
    print(test_pred_type_num, test_gold_type_num, test_match_type_num, test_pred_trigger_num, test_gold_trigger_num,
          test_match_tigger_num)
    print()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, log_file),"a") as fo:
        fo.write('language_model {}\n'.format(model_path))
        fo.write('learing_rate {:.6f}\n'.format(learning_rate))
        fo.write('batch_size {}\n'.format(batch_size))
        fo.write('dataset {}\n'.format(data_dir))
        fo.write('num_epochs {}\n'.format(epochs))
        fo.write('Best Epoch as {}\n'.format(best_epoch))
        fo.write('Best Global Step at {}\n'.format(best_global_step))

        fo.write('Dev\n')
        fo.write('best_dev_trigger_Precision {:.5f}, best_dev_trigger_Recall {:.5f}, best_dev_trigger_F1 {:.5f}\n'.format(
            dev_trigger_precision, dev_trigger_recall, dev_trigger_f1))

        fo.write('Test\n')
        fo.write('best_test_trigger_Precision {:.5f}, best_test_trigger_Recall {:.5f}, best_test_trigger_F1 {:.5f}'.format(
            test_trigger_precision, test_trigger_recall, test_trigger_f1))
        fo.write('='*100 + '\n')

if __name__ == '__main__':
    train()