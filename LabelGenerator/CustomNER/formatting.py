import os
import re
import sys
from collections import Counter

import pandas as pd
from tqdm import tqdm

sys.path.append("..")
from utils import data_utils

# gather all label files
label_filepaths = os.listdir("prompt_results")
label_data = pd.DataFrame()
for fp in label_filepaths:
    label_data = pd.concat(
        [label_data, data_utils.read_data(path=f"prompt_results/{fp}")], axis=0
    ).reset_index(drop=True)

# label_data = data_utils.read_data(
#     path=str(data_utils.root_dir / "prompt_result-20230407.ndjson.gz")
# )
print(len(label_data["input"]))


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


def chr_startswith_special_character(string: str):
    special_characters = ["?", "*"]
    for chr in special_characters:
        if string.startswith(chr):
            return True
    return False


def add2dicts(dict1: dict, dict2: dict) -> dict:
    dict3 = {**dict1, **dict2}
    for k, v in dict1.items():
        if k in dict2:
            dict3[k] += v
    return dict3


def count_label_stats(formatted_label_data: pd.DataFrame) -> dict:
    label_stats = {}
    for _l in formatted_label_data["openai_label_bioes"]:
        label_stats = add2dicts(label_stats, dict(Counter(_l)))

    print(label_stats)
    return label_stats



# OpenAI 字串轉陣列
formatted_label_data = {"input_id": [], "input": [], "openai_label": []}
for i in tqdm(range(len(label_data["openai_output"]))):
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
    lambda x: data_utils.split_sentence(x, flag="zh", mode="paragraph")
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
for i in tqdm(range(len(formatted_label_data["openai_label"]))):
    _input_text, _formatted_label = (
        formatted_label_data["input"][i],
        formatted_label_data["openai_label"][i],
    )
    _formatted_label_offset = {}
    for ent_type, ent_list in _formatted_label.items():
        if ent_list:
            try:
                _ent_list = [
                    f"\{chr}" if chr_startswith_special_character(chr) else chr
                    for chr in ent_list
                ]
                _ent_offsets = [
                    item.span()
                    for item in re.finditer("|".join(_ent_list), _input_text)
                ]
            except Exception as err:
                print(type(_ent_list), _ent_list)
                print(_input_text)
                raise {err}

            if _ent_offsets:
                _formatted_label_offset[ent_type] = _ent_offsets

    _collect_openai_label_offset.append(_formatted_label_offset)

formatted_label_data["openai_label_offset"] = _collect_openai_label_offset


# 出現位置+文本轉 BIOES
_collect_openai_label_bioes = []
for i in tqdm(range(len(formatted_label_data["openai_label"]))):
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


# data_utils.save_data(formatted_label_data, "formatting_result.ndjson.gz")
# print("Successfully saved formatting_result.ndjson.gz")


# label stats
label_stats = count_label_stats(formatted_label_data)
data_utils.save_data(label_stats, path="formatting_result_stats.json")


# split train/val/test
# 排除沒有任何實體的樣本
filtered_formatted_label_data = formatted_label_data[formatted_label_data["openai_label_offset"].apply(lambda x: True if x else False)].reset_index(drop=True)
print(len(filtered_formatted_label_data))
label_stats = count_label_stats(filtered_formatted_label_data)
data_utils.save_data(label_stats, path="filtered_formatting_result_stats.json")

_train, _dev, _test = (
    filtered_formatted_label_data.iloc[: len(filtered_formatted_label_data) // 10 * 7],
    filtered_formatted_label_data.iloc[
        len(filtered_formatted_label_data) // 10 * 7 : len(filtered_formatted_label_data) // 10 * 9
    ].reset_index(drop=True),
    filtered_formatted_label_data.iloc[len(filtered_formatted_label_data) // 10 * 9 :].reset_index(drop=True),
)
print(_train, _dev, _test)
# for _d, folder_name in zip([_train, _dev, _test], ["train", "dev", "test"]):
#     _output_dir = data_utils.root_dir / folder_name
#     _output_dir.mkdir(exist_ok=True, parents=True)
#     for i in range(len(_d)):
#         assert len(_d["input"][i]) == len(_d["openai_label_bioes"][i])

#     _input = [" ".join(_d["input"][i]) for i in range(len(_d))]
#     _output = [" ".join(_d["openai_label_bioes"][i]) for i in range(len(_d))]

#     data_utils.save_data(data=_input, path=str(_output_dir / "seq.in"))
#     data_utils.save_data(data=_output, path=str(_output_dir / "seq.out"))
