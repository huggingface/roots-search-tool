import argparse
import ast
import math
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pprint import pprint

import numpy as np
import pandas as pd
from datasets import load_from_disk
from utils import get_datasets_with_prefix


def summarize_dataset(dataset_name, base_dir, paragraph_length):
    print("Processing ", dataset_name, "...")
    try:
        dataset = load_from_disk(base_dir + dataset_name)
    except:
        print("Dataset", dataset_name, "not available locally")
        return None

    split = "train"
    meta_fields = []
    if "meta" in dataset[split][-1].keys():
        if isinstance(dataset[split][-1]["meta"], dict):
            meta_fields = list(dataset[split][-1]["meta"].keys())
        elif isinstance(dataset[split][-1]["meta"], str):
            meta_fields = list(ast.literal_eval(dataset[split][-1]["meta"]).keys())
        else:
            meta_fields = dataset[split][-1]["meta"]

    return (
        dataset_name,
        split,
        len(dataset[split]),
        list(dataset[split][-1].keys()),
        meta_fields,
        math.ceil(np.array([len(row["text"]) for row in dataset[split]]).mean()),
        math.ceil(np.array([row["text"].count(" ") for row in dataset[split]]).mean()),
        np.array([math.ceil(row["text"].count(" ") / paragraph_length) for row in dataset[split]]).sum(),
    )


if __name__ == "__main__":
    """
    Export your huggingface token which gives access to the `bigscience-catalogue-lm-data` organization.
    Run the following in the terminal where you execute the code (replace the XXX with your actual token):
    ```
    export HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXX
    ```
    """
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    if HUGGINGFACE_TOKEN is None:
        raise RuntimeError("Hugging Face token not specified.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        default="bigscience-data/roots_",
        help="We will summarize datasets the names of which match this prefix.",
    )
    parser.add_argument("--dir", type=str, default="data/", help="Path to a directory where results will be stored.")
    parser.add_argument("--para_len", type=int, default=240, help="Estimated length of a paragraph in words.")
    args = parser.parse_args()

    roots_datasets = get_datasets_with_prefix(prefix=args.prefix, use_auth_token=HUGGINGFACE_TOKEN)
    pprint(roots_datasets)

    workers = cpu_count()
    print("Number of workers:", workers)

    pool = Pool(workers)
    data = pool.map(partial(summarize_dataset, base_dir=args.dir, paragraph_length=args.para_len), roots_datasets)
    pool.close()
    pool.join()

    data = [d for d in data if d is not None]
    df = pd.DataFrame(
        data,
        columns=[
            "dataset",
            "split",
            "num_doc",
            "fields",
            "meta_fields",
            "avg_doc_len",
            "avg_doc_word_count",
            "total_paragraphs",
        ],
    )
    df["num_doc"] = df["num_doc"].astype("int64")
    df["avg_doc_len"] = df["avg_doc_len"].astype("int64")
    df["avg_doc_word_count"] = df["avg_doc_word_count"].astype("int64")
    df["total_paragraphs"] = df["total_paragraphs"].astype("int64")
    df.to_csv(args.dir + "datasets_summary_roots.csv")
