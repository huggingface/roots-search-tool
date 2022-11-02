import argparse
import ast
import glob
import os
import pickle
import random
import string
from functools import partial
from multiprocessing import Pool, cpu_count
from pprint import pprint

import jsonlines
import nltk
import pymongo
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from utils import (LANGUAGES, WHITESPACE_COMPATIBLE, get_datasets_with_prefix,
                   get_json_path, normalize)

# TODO(piktus) consider https://github.com/anoopkunchukuttan/indic_nlp_library

nltk.download("punkt")


def constuct_doc_id(dataset_name, datapoint_id):
    """
    Sample doc id format:s
    bigscience-data/roots_zh_wikinews/12345?seg=para_1_2&seg_id=20
    """
    return dataset_name + "/" + str(datapoint_id)


def find_whitespace(text):
    for i, c in enumerate(text):
        if c in string.whitespace:
            yield i


def get_segmentation(text, passage_tokens, overlap_tokens):
    whitespace_idx = [-1] + list(find_whitespace(text))
    unique_tokens = passage_tokens - overlap_tokens
    passages = []
    for i in range(0, len(whitespace_idx), unique_tokens):
        if i + passage_tokens >= len(whitespace_idx):
            passages.append((whitespace_idx[i] + 1, len(text)))
            break
        passages.append((whitespace_idx[i] + 1, whitespace_idx[i + passage_tokens] + 1))
    return passages


def test_reconstuct_segmented(text, segmentation, overlap_tokens):
    assert len(segmentation) > 0
    reconstructed = text[segmentation[0][0] : segmentation[0][1]]
    for segment_start, segment_end in segmentation[1:]:
        whitespace_idx = [-1] + list(find_whitespace(text[segment_start:segment_end]))
        reconstructed += text[segment_start + whitespace_idx[overlap_tokens] + 1 : segment_end]
    return text == reconstructed


def extract_segment(text, segment):
    segment_start, segment_end = segment
    return text[segment_start:segment_end]


def process_dataset_pyserini_looking_glass(
    dataset_path="/mnt/disks/looking_glass_storage/data/laion/laion2B-en/",
    dest_filename="/mnt/disks/looking_glass_storage/data/laion/laion2B-en-pyserini.jsonl",
):
    print("Processing", dataset_path, "to be saved under", dest_filename)

    dataset = load_from_disk(dataset_path)
    with jsonlines.open(dest_filename, mode="w") as writer:
        for datapoint_id, sample in tqdm(enumerate(dataset["train"])):
            pyserini_sample = {"id": datapoint_id, "contents": sample["TEXT"], "meta": {"url": sample["URL"]}}
            try:
                writer.write(pyserini_sample)
            except TypeError as e:
                print("Type error: {}".format(e), pyserini_sample)

    print("Finished processing", dataset_path)


def process_dataset_pyserini_looking_glass_normalized(
    dataset_path="/mnt/disks/looking_glass_storage/data/laion/laion2B-en/",
    dest_filename="/mnt/disks/looking_glass_storage/data/laion/laion2B-en-pyserini-normalized.jsonl",
):
    print("Processing", dataset_path, "to be saved under", dest_filename)

    dataset = load_from_disk(dataset_path)
    with jsonlines.open(dest_filename, mode="w") as writer:
        for datapoint_id, sample in tqdm(enumerate(dataset["train"])):
            if sample["TEXT"] is None:
                print("None text sample for", datapoint_id)
                continue
            pyserini_sample = {
                "id": datapoint_id,
                "contents": normalize(sample["TEXT"]),
                "meta": {"url": sample["URL"], "original_contents": sample["TEXT"]},
            }
            try:
                writer.write(pyserini_sample)
            except TypeError as e:
                print("Type error: {}".format(e), pyserini_sample)

    print("Finished processing", dataset_path)


def process_dataset_pyserini_looking_glass_dedup(
    dest_pattern="/mnt/disks/looking_glass_storage/data/laion/laion2B-en-pyserini-dedup/{}.jsonl",
):

    SPLIT_SIZE = 10000000

    processed = set()
    hits = 0
    hit_rate = 0
    file_id = 51
    line_id = 0
    """
    files = glob.glob("/mnt/disks/looking_glass_storage/data/laion/laion2B-en-pyserini-dedup/*")
    for file in files:
        print("Collecting ids from", file)
        file_id += 1
        with jsonlines.open(file, "r") as reader:
            for i, line in tqdm(enumerate(reader)):
                for doc in line["meta"]["docs"]:
                    processed.add(doc["_id"])

    with open("/mnt/disks/looking_glass_storage/data/laion/laion2B-en-pyserini-dedup/processed.pkl", "wb") as handle:
        pickle.dump(processed, handle)

    print("Saved processed pickle")

    return
    """
    with open("/mnt/disks/looking_glass_storage/data/laion/laion2B-en-pyserini-dedup/processed.pkl", "rb") as handle:
        processed = pickle.load(handle)

    sufix = "{:03d}".format(file_id)
    dest_filename = dest_pattern.format(sufix)
    print("Processing documents to be saved under", dest_filename)
    writer = jsonlines.open(dest_filename, mode="w")
    lines = []

    client = pymongo.MongoClient("localhost", 27017)
    looking_glass = client.looking_glass.prompts

    print("file_id", file_id)
    while True:
        hits = 0
        all_cursor = looking_glass.find()
        pbar = tqdm(enumerate(all_cursor), desc="description")
        try:
            for iter_lines, document in pbar:
                pbar.set_description(
                    "Iter lines {}. Total lines {}. Cache hit rate {}.".format(
                        iter_lines, len(processed), 100 * (hits / (iter_lines + 1))
                    )
                )
                if len(lines) >= SPLIT_SIZE:
                    writer.write_all(lines)
                    file_id += 1
                    sufix = "{:03d}".format(file_id)
                    dest_filename = dest_pattern.format(sufix)
                    writer = jsonlines.open(dest_filename, mode="w")
                    print("Processing documents to be saved under", dest_filename)
                    lines = []

                _id = document["_id"]
                if _id in processed:
                    hits += 1
                    continue
                processed.add(_id)

                normalized_text = document["NORMALIZED_TEXT"]
                line = {}
                line["id"] = line_id
                line["contents"] = normalized_text
                line_id += 1

                dup_cursor = looking_glass.find({"NORMALIZED_TEXT": normalized_text})
                dup_list = []
                for dup in dup_cursor:
                    dup.pop("NORMALIZED_TEXT")
                    dup_list.append(dup)
                    processed.add(dup["_id"])

                line["meta"] = {"docs": dup_list}
                lines.append(line)
            break
        except pymongo.errors.CursorNotFound as err:
            print(err)

    if len(lines) > 0:
        writer.write_all(lines)

    print("Finished processing")


def process_dataset_pyserini(dataset_name, base_dir, data_org, segmentation):
    mongodb_filename = get_json_path(dataset_name, base_dir, data_org, "mongodb")
    pyserini_filename = get_json_path(dataset_name, base_dir, data_org, "pyserini")
    print("Processing", dataset_name, "to be saved under", pyserini_filename)

    with jsonlines.open(mongodb_filename, mode="r") as reader, jsonlines.open(pyserini_filename, mode="w") as writer:
        for document in tqdm(reader):
            for i, segment in enumerate(document["segmentations"][segmentation]):
                pyserini_sample = {
                    "id": document["_id"] + "?seg={}&seg_id={}".format(segmentation, i),
                    "contents": extract_segment(document["text"], segment),
                    "meta": document["meta"],
                }
                try:
                    writer.write(pyserini_sample)
                except TypeError as e:
                    print("Type error: {}".format(e), pyserini_sample)

    print("Finished processing", dataset_name)


def process_dataset_pyserini_normalized(dataset_name, base_dir, data_org, segmentation):
    mongodb_filename = get_json_path(dataset_name, base_dir, data_org, "mongodb")
    pyserini_filename = get_json_path(dataset_name, base_dir, data_org, "pyserini-normalized")
    print("Processing", dataset_name, "to be saved under", pyserini_filename)

    with jsonlines.open(mongodb_filename, mode="r") as reader, jsonlines.open(pyserini_filename, mode="w") as writer:
        for document in tqdm(reader):
            for i, segment in enumerate(document["segmentations"][segmentation]):
                pyserini_sample = {
                    "id": document["_id"] + "?seg={}&seg_id={}".format(segmentation, i),
                    "contents": normalize(extract_segment(document["text"], segment)),
                }
                try:
                    writer.write(pyserini_sample)
                except TypeError as e:
                    print("Type error: {}".format(e), pyserini_sample)

    print("Finished processing", dataset_name)


def process_dataset_mongodb(dataset_name, base_dir, data_org, passage_tokens, overlap_tokens):
    jsonl_filename = get_json_path(dataset_name, base_dir, data_org, "mongodb")
    print("Processing", dataset_name, "to be saved under", jsonl_filename)

    dataset = load_from_disk(base_dir + dataset_name)
    with jsonlines.open(jsonl_filename, mode="w") as writer:
        for datapoint_id, sample in tqdm(enumerate(dataset["train"])):
            meta = None
            title = None
            if "meta" in sample:
                meta = ast.literal_eval(sample["meta"]) if isinstance(sample["meta"], str) else sample["meta"]
                for k, v in meta.items():
                    if isinstance(v, bytes):
                        meta[k] = meta[k].decode("utf-8")
                if "title" in meta:
                    title = meta["title"]

            # generate segmentations
            segmentation = get_segmentation(sample["text"], passage_tokens, overlap_tokens)
            assert test_reconstuct_segmented(sample["text"], segmentation, overlap_tokens)

            mongo_doc = {
                "_id": constuct_doc_id(dataset_name, datapoint_id),
                "text": sample["text"],
                "title": title,
                "language": "en",
                "meta": meta,
                "segmentations": {"para_{}_{}".format(passage_tokens, overlap_tokens): segmentation},
            }

            try:
                writer.write(mongo_doc)
            except TypeError as e:
                print("Type error: {}".format(e), mongo_doc)

    print("Finished processing", dataset_name)


def process_dataset_ict(dataset_name, base_dir, data_org, segmentation, sampling_rate, target_count=1500):
    mongodb_filename = get_json_path(dataset_name, base_dir, data_org, "mongodb")
    ict_filename = get_json_path(dataset_name, base_dir, data_org, "ict-1500")
    print("Processing", dataset_name, "to be saved under", ict_filename)

    num_lines = 0
    with jsonlines.open(mongodb_filename, mode="r") as reader:
        for document in tqdm(reader):
            for _ in document["segmentations"][segmentation]:
                num_lines += 1
    sampling_rate = target_count / num_lines
    total_count = 0

    with jsonlines.open(mongodb_filename, mode="r") as reader, jsonlines.open(ict_filename, mode="w") as writer:
        for document in tqdm(reader):
            if total_count >= target_count:
                break
            for i, segment in enumerate(document["segmentations"][segmentation]):
                if total_count >= target_count:
                    break
                if random.random() > sampling_rate:
                    continue
                segment = extract_segment(document["text"], segment)
                sentences = nltk.tokenize.sent_tokenize(segment)

                if sentences is None or len(sentences) == 0:
                    pprint(document)
                    pprint(sentences)
                    print("empty sentences")
                    continue
                pyserini_sample = {
                    "id": document["_id"] + "?seg={}&seg_id={}".format(segmentation, i),
                    "query": random.choice(sentences),
                    "passage": segment,
                }
                try:
                    writer.write(pyserini_sample)
                    total_count += 1
                except TypeError as e:
                    print("Type error: {}".format(e), pyserini_sample)

    print("Finished processing", dataset_name)


def main(args):

    if args.task == "looking_glass":
        process_dataset_pyserini_looking_glass_dedup()
        return

    data_org = args.data_org
    base_dir = args.dir

    print("Processing datasets matching prefix:", args.prefix)
    datasets = get_datasets_with_prefix(prefix=args.prefix, use_auth_token=HUGGINGFACE_TOKEN)
    filtered_datasets = []
    for dataset_name in datasets:
        lang = dataset_name.replace("/", "-").replace("_", "-").split("-")[3]
        if lang not in LANGUAGES:
            print("Skipping", dataset_name, "- the language is not whitespace compatible.")
            continue
        if os.path.isfile(get_json_path(dataset_name, base_dir, data_org, args.task)):
            print("Skipping", dataset_name + "- already processed.")
            continue
        filtered_datasets.append(dataset_name)
    print("Processing {} datasets:".format(len(filtered_datasets)))
    pprint(filtered_datasets)

    if args.task == "mongodb":
        process_dataset = partial(
            process_dataset_mongodb,
            base_dir=base_dir,
            data_org=data_org,
            passage_tokens=args.passage_len,
            overlap_tokens=args.overlap_len,
        )
    elif args.task == "pyserini":
        process_dataset = partial(
            process_dataset_pyserini, base_dir=base_dir, data_org=data_org, segmentation=args.segmentation
        )
    elif args.task == "ict" or args.task == "ict_1500":
        process_dataset = partial(
            process_dataset_ict,
            base_dir=base_dir,
            data_org=data_org,
            segmentation=args.segmentation,
            sampling_rate=args.sampling_rate,
        )
    else:
        raise ValueError("Unrecognized task!")

    workers = len(filtered_datasets)
    if workers > 256:
        workers = 256
    print("Number of workers:", workers)
    pool = Pool(workers)
    pool.map(process_dataset, filtered_datasets)
    pool.close()
    pool.join()


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
        choices=["pyserini", "mongodb", "ict", "ict_1500", "looking_glass"],
        help="Preprocessing task - pyserini or mongodb",
    )
    parser.add_argument("--dir", type=str, default="data/", help="Path to a directory where results will be stored.")
    parser.add_argument(
        "--data_org", type=str, default="bigscience-data", help="Huggingface hub organization maintaining the data."
    )
    parser.add_argument(
        "--prefix", type=str, default="roots_", help="Process datasets with names matching the prefix."
    )
    parser.add_argument("--passage_len", type=int, default=256, help="Estimated length of a paragraph in words.")
    parser.add_argument(
        "--overlap_len",
        type=int,
        default=16,
        help="Estimated length of an overlap between subsequent paragraphs in words.",
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        default="para_256_16",
        help="The segmentation used for Pyserini preprocessing, as defined in the MongoDB document.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=0.05,
        help="Sampling rate for evaluation data, expected a float between (0, 1.]",
    )
    args = parser.parse_args()

    main(args)
