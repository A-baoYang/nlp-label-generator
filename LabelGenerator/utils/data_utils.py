import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List

import pandas as pd
import yaml

# from KGBuilder.data_utils import split_sentence

root_dir = Path(__name__).parent.absolute()


def log_setting(
    log_folder: str = "logs-default", log_level: int = logging.INFO, stream: bool = True
) -> None:
    log_folder = log_folder if log_folder.startswith("logs-") else "logs-" + log_folder
    log_filename = os.path.join(
        Path(__file__).resolve().parent,
        log_folder,
        f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log",
    )
    Path(log_filename).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_filename,
        level=log_level,
        format="[%(asctime)s] [%(name)s | line:%(lineno)s | %(funcName)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if stream:
        console = logging.StreamHandler()
        console_format = logging.Formatter("[%(name)s] [%(levelname)s] - %(message)s")
        console.setFormatter(console_format)
        logging.getLogger().addHandler(console)


def read_data(path: str) -> Any:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        data = data.split("\n")
    elif path.endswith(".csv"):
        data = pd.read_csv(path)
    elif path.endswith(".ndjson"):
        data = pd.read_json(path, lines=True, orient="records")
    elif path.endswith(".ndjson.gz"):
        data = pd.read_json(path, lines=True, orient="records", compression="gzip")
    elif path.endswith(".pickle"):
        data = pd.read_pickle(path)
    elif path.endswith(".parquet"):
        data = pd.read_parquet(path)
    elif path.endswith(".yaml"):
        with open(path, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
    else:
        data = []
    return data


def save_data(data: Any, path: str) -> None:
    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif path.endswith(".txt") and isinstance(data, list):
        with open(path, "w", encoding="utf-8") as f:
            for _d in data:
                f.write(_d)
                f.write("\n")
    elif path.endswith(".csv"):
        data.to_csv(path, index=False)
    elif path.endswith(".ndjson"):
        data.to_json(path, lines=True, orient="records")
    elif path.endswith(".ndjson.gz"):
        data.to_json(path, lines=True, orient="records", compression="gzip")
    elif path.endswith(".pickle"):
        data.to_pickle(path)
    elif path.endswith(".parquet"):
        data.to_parquet(path)
    elif isinstance(data, list):
        with open(path, "w", encoding="utf-8") as f:
            for _d in data:
                f.write(_d)
                f.write("\n")
    else:
        pass


def split_sentence(
    document: str, flag: str = "all", mode: str = "sentence", limit: int = 510
) -> List[str]:
    """將文字段落根據標點符號分句 (@Reference: LTP)

    Args:
        document (str): 要進行分句的文字段落
        flag (:obj:`str`, optional): {"all", "zh", "en"}. "all": 中英文標點分句, "zh": 中文標點分句, "en": 英文標點分句   # noqa E501
        limit (:obj:`int`, optional): 預設單句最大長度為 510 個字符，可以透過這個此參數增減

    Returns:
        list: 分句後的結果列表

    """
    result_list = []
    try:
        if flag == "zh":
            document = re.sub(
                "(?P<quotation_mark>([。？！…](?![”’\"'])))",
                r"\g<quotation_mark>\n",
                document,
            )  # 單字符斷句號
            document = re.sub(
                "(?P<quotation_mark>([。？！]|…{1,2})[”’\"'])",
                r"\g<quotation_mark>\n",
                document,
            )  # 特殊引號
        elif flag == "en":
            document = re.sub(
                "(?P<quotation_mark>([.?!](?![”’\"'])))",
                r"\g<quotation_mark>\n",
                document,
            )  # 英文單字符斷句號
            document = re.sub(
                "(?P<quotation_mark>([?!.][\"']))", r"\g<quotation_mark>\n", document
            )  # 特殊引號
        else:
            document = re.sub(
                "(?P<quotation_mark>([。？！….?!](?![”’\"'])))",
                r"\g<quotation_mark>\n",
                document,
            )  # 單字符斷句號
            document = re.sub(
                "(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’\"']))",
                r"\g<quotation_mark>\n",
                document,
            )  # 特殊引號

        if mode == "sentence":
            sent_list_ori = document.splitlines()
            for sent in sent_list_ori:
                sent = sent.strip()
                if not sent:
                    continue
                else:
                    while len(sent) > limit:
                        temp = sent[0:limit]
                        result_list.append(temp)
                        sent = sent[limit:]
                    result_list.append(sent)

        elif mode == "paragraph":
            sent_list_ori = document.splitlines()[::-1]
            while sent_list_ori:
                paragraph = sent_list_ori.pop().strip()
                len_sent = len(sent_list_ori)
                while sent_list_ori and len(paragraph + sent_list_ori[-1]) < limit:
                    paragraph += sent_list_ori.pop().strip()
                    if len(sent_list_ori) == len_sent:
                        print(sent_list_ori)
                        raise Exception("not reducing")
                    else:
                        len_sent = len(sent_list_ori)
                result_list.append(paragraph)

    except Exception as e:
        print(f"Error: {e}")
        result_list.clear()
        result_list.append(document)

    return result_list
