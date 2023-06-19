import logging
from pathlib import Path
from time import sleep

import openai
import tiktoken

root_dir = Path(__name__).parent.absolute()


class OpenAIAPIWrapper:
    def __init__(self, API_KEY: str):
        """初始化並驗證身份"""
        self._openai = openai
        self._openai.api_key = API_KEY

    def get_embeddings(self, text, model="text-embedding-ada-002"):
        """免費帳戶有時間內之取用上限"""
        self._model = model
        self._text = text.replace("\n", " ")

        success = False
        while not success:
            try:
                res = self._openai.Embedding.create(input=[text], model=model)["data"][
                    0
                ]["embedding"]
                success = True

            except Exception as e:
                logging.error(e)
                sleep(60)
        return res

    def get_text_completion(
        self, prompt, model="text-davinci-003", generation_params=None
    ):
        """generation_params (default)
        - temperature=0.7
        - max_tokens=1024
        """
        self._model = model
        self._prompt = prompt
        self._generation_params = generation_params

        if not generation_params:
            generation_params = {"temperature": 0.7, "max_tokens": 1024}
        logging.info(f"\nPrompt: {prompt}\nWith params: {generation_params}")
        response = self._openai.Completion.create(
            model=model, prompt=prompt, **generation_params
        )
        res = response["choices"][0]["text"].strip()
        logging.info(f"\nGPT-3 Reply: {res}\n")
        return res

    def get_chat_completion(
        self, messages, model="gpt-3.5-turbo", generation_params=None
    ):
        """
        messages:

        Example: [
            {
                "role": "system",
                "content: "你現在是一位金融專家"
            },
            {
                "role": "system",
                "content: f"請根據以下文章段落，使用幽默又不失專業的語氣對各個重要子議題以繁體中文摘要：{article}"
            }
        ]

        generation_params:

        Example: {"temperature": 0.7, "max_tokens": 1024} (default)

        """
        self._model = model
        self._messages = messages
        self._generation_params = generation_params

        if not generation_params:
            generation_params = {"temperature": 0.7, "max_tokens": 1024}
        logging.info(f"\nMessages: {messages}\nWith params: {generation_params}")

        response = self._openai.ChatCompletion.create(
            model=model, messages=messages, **generation_params
        )
        res = response["choices"][0]["message"]["content"].strip()
        logging.info(f"\nChatGPT Reply: {res}\n")
        return res

    def num_tokens_from_messages(self, messages: list, model: str = "gpt-3.5-turbo"):
        """Returns the number of tokens used by a list of messages or texts."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logging.info("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":
            logging.info(
                "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301."
            )
            return self.num_tokens_from_messages(
                messages=messages, model="gpt-3.5-turbo-0301"
            )
        elif model == "gpt-4":
            logging.info(
                "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314."
            )
            return self.num_tokens_from_messages(messages=messages, model="gpt-4-0314")
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}."""
            )

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def price_counter(
        self, num_tokens: int, model: str = "gpt-3.5-turbo", currency: str = "USD"
    ):
        self._price_table = {
            "gpt-4-32k": 0.00006,
            "gpt-4": 0.00003,
            "gpt-3.5-turbo-16k": 0.000003,
            "gpt-3.5-turbo": 0.0000015,
            "text-davinci-003": 0.00002,
        }
        if currency == "TWD":
            return num_tokens * self._price_table[model] * 30.79
        return num_tokens * self._price_table[model]
