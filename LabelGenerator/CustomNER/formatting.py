import re
import sys
from collections import Counter

import pandas as pd

sys.path.append("..")
from utils import data_utils

label_data = data_utils.read_data(path=str(data_utils.root_dir / "prompt_result.json"))
len(label_data["input"])
entity_types = [
    "組織(ORGANIZATION)",
    "公司(COMPANY)",
    "股票(STOCK)",
    "人物(PERSON)",
    "國家(GPE)",
    "地點(LOCATION)",
    "產品(PRODUCT)",
]
# 分隔 OpenAI 輸出結果
openai_output_splitter = ",|\n-|、"
# 清理文章內特殊符號
special_chars = "\r|\n|\u3000|\t|\xa0|\xa07"
# BIOES label
bioes_label_table = {
    "組織(ORGANIZATION)": "ORG",
    "公司(COMPANY)": "COM",
    "股票(STOCK)": "STO",
    "人物(PERSON)": "PER",
    "國家(GPE)": "GPE",
    "地點(LOCATION)": "LOC",
    "產品(PRODUCT)": "PROD",
}


def result_check(result_string):
    for test_term in entity_types:
        if test_term not in result_string:
            return False
    return True


# OpenAI 字串轉陣列
formatted_label_data = {"input_id": [], "input": [], "openai_label": []}
for i in range(len(label_data["openai_output"])):
    _result_string = label_data["openai_output"][i]
    # 如果 key 值符合預期才繼續往下處理
    if result_check(_result_string):
        _formatted_label = {k: [] for k in entity_types}
        for item in _result_string.split("\n\n"):
            # 如果該類別缺少答案則跳過
            if len(item.split(":")) == 2:
                h, t = item.split(":")
                if h.strip() in _formatted_label:
                    _formatted_label.update(
                        {
                            h.strip(): [
                                chr.strip()
                                for chr in re.split(openai_output_splitter, t)
                                if chr.strip() and chr.strip() not in ["無", "无"]
                            ]
                        }
                    )

        formatted_label_data["input_id"].append(label_data["input_id"][i])
        formatted_label_data["input"].append(label_data["input"][i])
        formatted_label_data["openai_label"].append(_formatted_label)


# 清理文章內特殊符號、長度大於 510 則分多段
formatted_label_data["input"] = [
    "，".join(sent.split()) for sent in formatted_label_data["input"]
]
formatted_label_data["input"] = [
    re.sub(special_chars, " ", sent) for sent in formatted_label_data["input"]
]
formatted_label_data = pd.DataFrame(formatted_label_data)
formatted_label_data["input"] = formatted_label_data["input"].apply(
    lambda x: split_sentence(x, flag="zh", mode="paragraph")
)
formatted_label_data = formatted_label_data.explode(["input"]).reset_index(drop=True)
formatted_label_data["input"] = formatted_label_data["input"].apply(
    lambda x: x[1:] if x.startswith("，") else x
)
formatted_label_data = formatted_label_data[
    formatted_label_data["input"].apply(lambda x: len(x)) < 512
].reset_index(drop=True)

# 陣列轉出現位置
_collect_openai_label_offset = []
for i in range(len(formatted_label_data["openai_label"])):
    _input_text, _formatted_label = (
        formatted_label_data["input"][i],
        formatted_label_data["openai_label"][i],
    )
    _formatted_label_offset = {}
    for ent_type, ent_list in _formatted_label.items():
        if ent_list:
            _ent_offsets = [
                item.span() for item in re.finditer("|".join(ent_list), _input_text)
            ]
            if _ent_offsets:
                _formatted_label_offset[ent_type] = _ent_offsets

    _collect_openai_label_offset.append(_formatted_label_offset)

formatted_label_data["openai_label_offset"] = _collect_openai_label_offset


# 出現位置+文本轉 BIOES
_collect_openai_label_bioes = []
for i in range(len(formatted_label_data["openai_label"])):
    _input_text, _label_offset = (
        formatted_label_data["input"][i],
        formatted_label_data["openai_label_offset"][i],
    )
    bioes_tags = ["O"] * len(_input_text)
    for ent_type, ent_offset_list in _label_offset.items():
        ent_bio = bioes_label_table[ent_type]
        for ent_offset in ent_offset_list:
            bioes_tags[ent_offset[0]] = f"B-{ent_bio}"
            bioes_tags[ent_offset[0] + 1 : ent_offset[1]] = [f"I-{ent_bio}"] * len(
                bioes_tags[ent_offset[0] + 1 : ent_offset[1]]
            )
    _collect_openai_label_bioes.append(bioes_tags)

formatted_label_data["openai_label_bioes"] = _collect_openai_label_bioes


data_utils.save_data(formatted_label_data, "formatting_result.ndjson.gz")


# label stats
label_stats = dict(Counter(formatted_label_data["openai_label_bioes"].sum()))
data_utils.save_data(label_stats, path="formatting_result_stats.json")


# split train/val/test
_train, _dev, _test = (
    formatted_label_data.iloc[: len(formatted_label_data) - 100],
    formatted_label_data.iloc[
        len(formatted_label_data) - 100 : len(formatted_label_data) - 50
    ].reset_index(drop=True),
    formatted_label_data.iloc[len(formatted_label_data) - 50 :].reset_index(drop=True),
)
for _d, folder_name in zip([_train, _dev, _test], ["train", "dev", "test"]):
    _output_dir = data_utils.root_dir / folder_name
    _output_dir.mkdir(exist_ok=True, parents=True)
    for i in range(len(_d)):
        assert len(_d["input"][i]) == len(_d["openai_label_bioes"][i])

    _input = [" ".join(_d["input"][i]) for i in range(len(_d))]
    _output = [" ".join(_d["openai_label_bioes"][i]) for i in range(len(_d))]

    data_utils.save_data(data=_input, path=str(_output_dir / "seq.in"))
    data_utils.save_data(data=_output, path=str(_output_dir / "seq.out"))
