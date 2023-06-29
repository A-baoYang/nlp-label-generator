import configparser
import json
import logging
import sys
from time import sleep

import pandas as pd

sys.path.append("..")
from utils.data_utils import log_setting, read_data, root_dir, save_data
from utils.openai_utils import OpenAIAPIWrapper

log_setting()
date_str_list = [
    "20230201",
    "20230217",
    # "20230305",
    # "20230328",
    # "20230407",
    # "20230412",
    # "20230504",
    # "20230523",
    # "20230601",
    # "20230617",
]

for date_str in date_str_list:
    news = read_data(
        path=f"gs://dst-largitdata/domestic/merged-data/merged-{date_str}.ndjson.gz"
    )
    news = news.explode("CNYES_INDUSTRY").reset_index(drop=True)
    news = (
        news[
            (news["CNYES_INDUSTRY"].apply(lambda x: x["prob"] > 0.5))
            & (news["content"].apply(lambda x: len(x) < 1500))
        ][["id", "content"]]
        .drop_duplicates(subset=["content"])
        .reset_index(drop=True)
    )
    config = configparser.ConfigParser()
    config.read_file(open(root_dir / "secret.cfg"))
    api = OpenAIAPIWrapper(
        API_KEY=config.get("OPENAI", "API_KEY")
    )
    PROMPT = "文本:{article}。請從文本中列出所有組織(ORGANIZATION)、公司(COMPANY)、股票(STOCK)、人物(PERSON)、國家(GPE)、地點(LOCATION)、產品(PRODUCT)"

    prompt_result = {"input_id": [], "input": [], "openai_output": []}
    num_tokens = 0
    for i in range(len(news)):
        messages = [
            {
                "role": "user",
                "content": PROMPT.format(article=news["content"][i][:1500]),
            }
        ]
        num_tokens += api.num_tokens_from_messages(messages=messages)
        result = None
        while not result:
            try:
                result = api.get_chat_completion(messages, generation_params={"temperature": 0})
            except Exception as err:
                logging.error(err)
                sleep(10)
            sleep(1)

        num_tokens += api.num_tokens_from_messages(messages=[{"output": result}])

        prompt_result["input_id"].append(news["id"][i])
        prompt_result["input"].append(news["content"][i])
        prompt_result["openai_output"].append(result)

        with open(f"prompt_result-{date_str}.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps({"input_id": news["id"][i], "input": news["content"][i], "openai_output": result}))
            f.write("\n")

    save_data(
        data=pd.DataFrame(prompt_result), path=f"prompt_result-{date_str}.ndjson.gz"
    )

    logging.info(
        f'{date_str} : {api.price_counter(num_tokens=num_tokens, currency="TWD")}'
    )
