import re
import string

from huggingface_hub import HfApi

LANGUAGES = ["ar", "ca", "code", "en", "es", "eu", "fr", "id", "indic", "nigercongo", "pt", "vi", "zh", "zhs", "zht"]
WHITESPACE_COMPATIBLE = [
    "ca",
    "en",
    "es",
    "eu",
    "fr",
    "nigercongo",
    "pt",
]


def get_datasets_with_prefix(prefix, use_auth_token=None, data_org="bigscience-data"):
    datasets = [
        ds_info.id
        for ds_info in HfApi().list_datasets(use_auth_token=use_auth_token)
        if (ds_info.id.startswith(data_org) and prefix in ds_info.id)
    ]
    return datasets


def get_json_path(dataset_name, base_dir, data_org, task):
    dataset_id = dataset_name[len(data_org + "/") :]
    dest_dir = base_dir + data_org + "-" + task + "/"
    return dest_dir + dataset_id + ".jsonl"


def get_langauge(dataset_name):
    tokens = dataset_name.split("_")
    return tokens[1].split("-")[0]


def normalize(s):
    # unused
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def replace_quotes(text):
        text = text.replace("’", "'")
        text = text.replace("‘", "'")
        text = text.replace("”", "'")
        text = text.replace("“", "'")
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if s is None:
        return None

    return white_space_fix(remove_punc(replace_quotes(lower(s))))
