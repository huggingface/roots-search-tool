import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pprint import pprint

import jsonlines
import pymongo
from datasets import load_from_disk
from tqdm import tqdm
from utils import (WHITESPACE_COMPATIBLE, get_datasets_with_prefix,
                   get_json_path, normalize)


def process_dataset_batch(dataset_name, base_dir, data_org, batch_size):
    client = MongoClient("localhost", 27017)
    roots = client.roots.roots
    mongodb_filename = get_json_path(dataset_name, base_dir, data_org, "mongodb")

    print("Processing", dataset_name)
    with jsonlines.open(mongodb_filename, mode="r") as reader:
        docs_batch = []
        for mongo_doc in tqdm(reader):
            if len(docs_batch) >= batch_size:
                try:
                    roots.insert_many(docs_batch)
                    docs_batch = []
                except BulkWriteError as e:
                    print(e["writeErrors"])
            else:
                docs_batch.append(mongo_doc)
        if len(docs_batch) > 0:
            try:
                roots.insert_many(docs_batch)
            except Exception as e:
                print(e)

    print("Finished processing", dataset_name)


def process_dataset_one(dataset_name, base_dir, data_org, batch_size):
    client = pymongo.MongoClient("localhost", 27017)
    roots = client.roots.roots
    mongodb_filename = get_json_path(dataset_name, base_dir, data_org, "mongodb")

    print("Processing", dataset_name)
    with jsonlines.open(mongodb_filename, mode="r") as reader:
        for mongo_doc in tqdm(reader):
            try:
                roots.insert_one(mongo_doc)
            except pymongo.errors.DuplicateKeyError as e:
                # print("DuplicateKeyError for", mongo_doc["_id"])
                continue
            except pymongo.errors.DocumentTooLarge as e:
                print("DocumentTooLarge error for", mongo_doc["_id"])
                continue
    print("Finished processing", dataset_name)


def process_dataset_batch_looking_glass(batch_size=100000):
    client = pymongo.MongoClient("localhost", 27017)
    # client.drop_database("looking_glass")
    looking_glass = client.looking_glass.prompts

    dataset_path = "/mnt/disks/looking_glass_storage/data/laion/laion2B-en/"
    print("Processing", dataset_path)
    dataset = load_from_disk(dataset_path)
    docs_batch = []
    for datapoint_id, sample in tqdm(enumerate(dataset["train"])):
        if len(docs_batch) >= batch_size:
            try:
                looking_glass.insert_many(docs_batch)
                docs_batch = []
            except BulkWriteError as e:
                print(e["writeErrors"])
        else:
            mongo_doc = {}
            mongo_doc["_id"] = datapoint_id
            mongo_doc["TEXT"] = sample["TEXT"]
            mongo_doc["NORMALIZED_TEXT"] = normalize(sample["TEXT"])
            mongo_doc["URL"] = sample["URL"]
            docs_batch.append(mongo_doc)
    if len(docs_batch) > 0:
        try:
            looking_glass.insert_many(docs_batch)
        except Exception as e:
            print(e)
    print("Finished processing", dataset_path)


def process_dataset_one_looking_glass():
    client = pymongo.MongoClient("localhost", 27017)
    # client.drop_database("looking_glass")
    looking_glass = client.looking_glass.prompts

    dataset_path = "/mnt/disks/looking_glass_storage/data/laion/laion2B-en/"
    print("Processing", dataset_path)
    dataset = load_from_disk(dataset_path)
    for datapoint_id, sample in tqdm(enumerate(dataset["train"])):
        mongo_doc = {}
        mongo_doc["_id"] = datapoint_id
        mongo_doc["TEXT"] = sample["TEXT"]
        mongo_doc["NORMALIZED_TEXT"] = normalize(sample["TEXT"])
        mongo_doc["URL"] = sample["URL"]
        try:
            looking_glass.insert_one(mongo_doc)
        except pymongo.errors.DuplicateKeyError as e:
            print("DuplicateKeyError for", mongo_doc["_id"])
            continue
        except pymongo.errors.DocumentTooLarge as e:
            print("DocumentTooLarge error for", mongo_doc["_id"])
            continue
    print("Finished processing", dataset_name)


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
        "--task",
        type=str,
        required=True,
        choices=["pyserini", "mongodb", "looking_glass"],
        help="Preprocessing task - pyserini or mongodb",
    )
    parser.add_argument("--dir", type=str, default="data/", help="Path to a directory containing source files.")
    parser.add_argument(
        "--data_org", type=str, default="bigscience-data", help="Huggingface hub organization maintaining the data."
    )
    parser.add_argument(
        "--prefix", type=str, default="roots_", help="Process datasets with names matching the prefix."
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Number of documents added to MongoDB per batch.")
    args = parser.parse_args()

    if args.task == "looking_glass":
        process_dataset_batch_looking_glass()
    else:
        data_org = args.data_org
        prefix = data_org + "/" + args.prefix

    print("Processing datasets matching prefix:", prefix)
    datasets = get_datasets_with_prefix(prefix=prefix, use_auth_token=HUGGINGFACE_TOKEN)
    filtered_datasets = []
    for dataset_name in datasets:
        lang = dataset_name.replace("/", "-").replace("_", "-").split("-")[3]
        if lang not in WHITESPACE_COMPATIBLE:
            print("Skipping", dataset_name, "- the language is not whitespace compatible.")
            continue
        filtered_datasets.append(dataset_name)
    print("Processing {} datasets:".format(len(filtered_datasets)))
    pprint(filtered_datasets)

    workers = len(filtered_datasets)
    print("Number of workers:", workers)
    pool = Pool(workers)
    pool.map(
        partial(
            process_dataset_one,
            base_dir=args.dir,
            data_org=data_org,
            batch_size=args.batch_size,
        ),
        filtered_datasets,
    )
    pool.close()
    pool.join()
