import pickle
from dataclasses import dataclass
import random
import collections
import numpy as np

@dataclass
class MNLIInputFeatures:
    premise: str
    hypothesis: str
    label: int
    # prob: float
@dataclass
class MNLIInputFeatures1:
    premise: str
    hypothesis: str
    label: int
    prob: float

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

def neutral(input_type):
    label_name = ["BE_BORN", "MARRY", "DIVORCE", "INJURE", "DIE", "TRANSPORT", "TRANSFER_OWNERSHIP",
                  "TRANSFER_MONEY", "START_ORG", "MERGE_ORG", "DECLARE_BANKRUPTCY", "END_ORG", "ATTACK",
                  "DEMONSTRATE", "MEET", "PHONE_WRITE", "START_POSITION", "END_POSITION", "NOMINATE",
                  "ELECT", "ARREST_JAIL", "RELEASE_PAROLE", "TRIAL_HEARING", "CHARGE_INDICT", "SUE",
                  "CONVICT", "SENTENCE", "FINE", "EXECUTE", "EXTRADITE", "ACQUIT", "PARDON", "APPEAL"]
    for type in input_type:
        if type in label_name:
            label_name.remove(type)
    neutral_label = type_format_transfer_reverse(label_name[random.randint(0, len(label_name) - 1)])
    return neutral_label

def type_format_transfer_reverse(input):
    if "_" in input:
        return input.replace("_", "-")
    else:
        return input

hypothesis_dict = load_trg_probe_lexicon()
def load(path_name: object) -> object:
    with open(path_name,'rb') as file:
        return pickle.load(file)

ls = []
ls1 = []
ls2 = []
_data = []

with open("raw_data.pickle", 'rb') as file:
    dic = pickle.load(file)
    for line in dic:
        _data.append(line)

for line in _data:
    if len(line["sentence_type"]) == 1:
        ls1.append(line)
        ls.append(line)
    else:
        ls2.append(line)
        ls.append(line)

label_name = ["BE_BORN", "MARRY", "DIVORCE", "INJURE", "DIE", "TRANSPORT", "TRANSFER_OWNERSHIP",
              "TRANSFER_MONEY", "START_ORG", "MERGE_ORG", "DECLARE_BANKRUPTCY", "END_ORG", "ATTACK",
              "DEMONSTRATE", "MEET", "PHONE_WRITE", "START_POSITION", "END_POSITION", "NOMINATE",
              "ELECT", "ARREST_JAIL", "RELEASE_PAROLE", "TRIAL_HEARING", "CHARGE_INDICT", "SUE",
              "CONVICT", "SENTENCE", "FINE", "EXECUTE", "EXTRADITE", "ACQUIT", "PARDON", "APPEAL"]

type_ls = []
for line in ls:
    for ty in line["sentence_type"]:
        type_ls.append(ty)
col = collections.Counter(type_ls)
print(col)

out_ls = []
num1 = 0

for line in ls1:
    if num1 < len(ls1)*1/3:

        index = random.randint(0,len(hypothesis_dict[type_format_transfer_reverse(line["sentence_type"][0])]) - 1)
        new_line = MNLIInputFeatures1(premise=line["sentence"], hypothesis=hypothesis_dict[type_format_transfer_reverse(line["sentence_type"][0])][index], label=2, prob=float(line["line_type_mean_prob"][0]))
        out_ls.append(new_line)

        num1 += 1
    elif num1 < len(ls1) * 2/3:

        neutral_label = neutral(line["sentence_type"])
        index = random.randint(0,len(hypothesis_dict[neutral_label]) - 1)
        new_line = MNLIInputFeatures1(premise=line["sentence"], hypothesis=hypothesis_dict[neutral_label][index], label=1, prob=float(line["line_type_mean_prob"][0]))
        out_ls.append(new_line)
        num1 += 1

    else:

        new_line = MNLIInputFeatures1(premise=line["sentence"], hypothesis="This sentence does not express any events.", label=0, prob=float(line["line_type_mean_prob"][0]))
        out_ls.append(new_line)
        num1 += 1


num2 = 0
aaa = 0
for line in ls2:
    if num2 < len(ls2)*1/3:
        for ty, prob in zip(line["sentence_type"], line["line_type_mean_prob"]):
            index = random.randint(0,len(hypothesis_dict[type_format_transfer_reverse(ty)]) - 1)
            new_line = MNLIInputFeatures1(premise=line["sentence"], hypothesis=hypothesis_dict[type_format_transfer_reverse(ty)][index], label=2, prob=float(prob))
            out_ls.append(new_line)

        num2 += 1

    elif num2 < len(ls2) * 2/3:

        neutral_label = neutral(line["sentence_type"])
        index = random.randint(0,len(hypothesis_dict[neutral_label]) - 1)

        n_ = random.randint(0, len(line["line_type_mean_prob"])-1)
        new_line = MNLIInputFeatures1(premise=line["sentence"], hypothesis=hypothesis_dict[neutral_label][index], label=1, prob=line["line_type_mean_prob"][n_])

        out_ls.append(new_line)
        num2 += 1

    else:

        n_ = random.randint(0, len(line["line_type_mean_prob"])-1)
        new_line = MNLIInputFeatures1(premise=line["sentence"], hypothesis="This sentence does not express any events.", label=0, prob=line["line_type_mean_prob"][n_])

        out_ls.append(new_line)
        num2 += 1


# file = open("fine_tune_data.pickle", "wb")
# pickle.dump(out_ls, file)
# file.close()