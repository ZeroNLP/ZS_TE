from copy import deepcopy
import pickle
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

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

class MyDataset(Dataset):
    def __init__(self, data_dir, tokenizer: BertTokenizer):
        sentence_list = []
        trigger_list = []
        pred_type_list = []

        all_trigger_list = []

        if data_dir == "/data/rywu/Train_TC.pickle":
            with open(data_dir, 'rb') as file:
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

            with open(data_dir, 'rb') as file:
                dic = pickle.load(file)
            for line in dic:
                if line["sentence_type"] != "NOT_MENTION":
                    if "real_sentence_keyword" in line:
                        input_text = line["sentence"]
                        input_label = line["fake_substitute_type"]
                        trigger_index = all_trigger_list.index(line["fake_trigger"])

                        sentence_list.append(input_text)
                        trigger_list.append(trigger_index)
                        pred_type_list.append(input_label)
                    else:
                        if line["kind_number"] != 1:
                            for sentence_type, sentence_keyword in zip(line["sentence_type"], line["sentence_keyword"]):
                                input_text = line["sentence"]
                                input_label = sentence_type
                                trigger_index = all_trigger_list.index(sentence_keyword)

                                sentence_list.append(input_text)
                                trigger_list.append(trigger_index)
                                pred_type_list.append(input_label)

                        else:
                            input_text = line["sentence"]
                            input_label = line["sentence_type"]
                            trigger_index = all_trigger_list.index(line["sentence_keyword"])

                            sentence_list.append(input_text)
                            trigger_list.append(trigger_index)
                            pred_type_list.append(input_label)
                else:
                    input_text = line["sentence"]
                    input_label = line["fake_substitute_type"]
                    trigger_index = all_trigger_list.index(line["fake_trigger"])

                    sentence_list.append(input_text)
                    trigger_list.append(trigger_index)
                    pred_type_list.append(input_label)

            self.pred_type = pred_type_list
            self.trigger = trigger_list
            self.text = sentence_list
            self.tokenizer = tokenizer

        else:
            with open(data_dir, 'rb') as fin:
                dic = pickle.load(fin)
            for line in dic:
                if line["pred_event_type"] != []:
                    for event in line["pred_event_type"]:
                        input_text = line["sentence"]
                        sentence_list.append(input_text)
                        pred_type_list.append(type_format_transfer(event))
                        trigger_list.append([])

                else:
                    input_text = line["sentence"]
                    sentence_list.append(input_text)
                    pred_type_list.append("NOT_MENTION")
                    trigger_list.append([])

            self.pred_type = pred_type_list
            self.trigger = trigger_list
            self.text = sentence_list
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.trigger)

    def __getitem__(self, idx):
        text = self.text[idx]
        trigger = self.trigger[idx]
        pred_type = self.pred_type[idx]

        if text.strip()[-1] not in ['.', '!', '?']:
            text = text + '.'
        text = "[event] {}".format(pred_type) + "[event]" + "The trigger is [MASK] " + "[SEP]" + text
        original_text = deepcopy(text)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding["input_ids"].flatten()
        masked_index = 7

        return dict(
            original_text = original_text,
            input_ids = input_ids,
            masked_index = masked_index,
            trigger = trigger,
        )