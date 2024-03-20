import json
from tqdm.auto import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

def load_trg_probe_lexicon():
    """hypothesis"""
    with open("/data/rywu/trg_te_probes_5.txt", "r", encoding="utf8") as fr:
        lexicon = {}
        for line in fr:
            line = line.strip()
            if line:
                if line.isupper():
                    event_type = line
                    lexicon[event_type] = []
                else:
                    lexicon[event_type].append(line)

    return lexicon
hypothesis_dict = load_trg_probe_lexicon()

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(2)

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

dev_dataset = IEDataset("/home/rywu/scut/dev.event.json")
test_dataset = IEDataset("/home/rywu/scut/test.event.json")

class Config(object):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"当前设备: {torch.cuda.get_device_name()}")
        self.threshold = 0.98
        self.path = "/data/rywu/raw_wordn_data_te_model"
        self.id2label = {
            "0": "CONTRADICTION",
            "1": "NEUTRAL",
            "2": "ENTAILMENT"
        }
        self.output_file = self.path + "/Dev_0.1_smoothing_w_r" + ".json"
        self.log_file = self.path + "/Dev_0.1_smoothing_w_r" + ".log"
        self.test_output_file = self.path + "/Test_0.1_smoothing_w_r" + ".json"
        self.test_log_file = self.path + "/Test_0.1_smoothing_w_r" + ".log"

config = Config()

TE_model = AutoModelForSequenceClassification.from_pretrained("/data/transformers/microsoft_deberta-v2-xlarge-mnli").to(config.device)
# /data/zzengae/transformers/microsoft_deberta-v2-xlarge-mnli
TE_model.load_state_dict(torch.load("/data/rywu/wordn_raw_model/wordn_raw_model_0.1_smoothing.pt"))
TE_model.eval()

TE_tokenizer = AutoTokenizer.from_pretrained("/data/transformers/microsoft_deberta-v2-xlarge-mnli")

def predict(config, model, tokenizer, hypothesis_dict, dataset, output_file):
    all_pred_data = []
    for i, data in enumerate(tqdm(dataset)):
        print()
        print(i, data["sentence"])
        premise = data["sentence"]

        pred_event_types = []
        gold_help_pred_arguments = []
        gold_event_types = []

        for event in data["event_mentions"]:
            for argument in event["arguments"]:
                gold_help_pred_arguments.append(argument["text"])

        for event_type in hypothesis_dict.keys():

            probs_list = []
            for hypothesis in hypothesis_dict[event_type]:
                x = tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation='only_first', max_length=512).to(config.device)
                logits = model(x)[0]
                probs = logits.softmax(1)[:, 2]  # entailment
                probs_dict = {"event_type": event_type, "top_1_hypothesis": hypothesis, "probs": probs.item()}
                probs_list.append(probs_dict)

            max_probs_dict = sorted(probs_list, key=lambda x: x["probs"], reverse=True)[0]

            # if max_probs_dict["probs"] >= config.threshold:
            pred_event_types.append(max_probs_dict)

        pred_event_types = sorted(pred_event_types, key=lambda x: x["probs"], reverse=True)
        for event in data["event_mentions"]:
            gold_event_types.append(event["event_type"])

        print(f"gold_event_types = {gold_event_types}")

        if len(pred_event_types) == 0:
            print("pred_event_types = []\n")
        else:
            print(r"pred_event_types = [")
            for each in pred_event_types:
                print("\t" + str(each))
            print("]\n")

        data["pred_event_type"] = pred_event_types
        all_pred_data.append(data)

    with open(output_file, "w") as output_file:
        for pred in all_pred_data:
            output_file.write(json.dumps(pred) + "\n")
            output_file.flush()

predict(config, TE_model, TE_tokenizer, hypothesis_dict, test_dataset, config.test_output_file)
predict(config, TE_model, TE_tokenizer, hypothesis_dict, dev_dataset, config.output_file)