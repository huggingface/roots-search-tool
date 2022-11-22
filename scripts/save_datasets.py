import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pprint import pprint

from datasets import load_dataset
from utils import get_datasets_with_prefix


def save_dataset(dataset_name, base_dir):
    if os.path.isdir(base_dir + dataset_name):
        print(base_dir + dataset_name + " already processed, passing")
        return
    dataset = load_dataset(dataset_name, use_auth_token=HUGGINGFACE_TOKEN, num_proc=cpu_count())
    dataset.save_to_disk(base_dir + dataset_name)

    # The code below doesn't seem to have effect, currently deleting cache manually.
    i = dataset["train"].cleanup_cache_files()
    print("Removing", i, "cached files.")


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
        help="We will save datasets the names of which match this prefix.",
    )
    parser.add_argument("--dir", type=str, default="data/", help="Path to a directory where datasets will be stored.")
    args = parser.parse_args()

    roots_datasets = get_datasets_with_prefix(prefix=args.prefix, use_auth_token=HUGGINGFACE_TOKEN)
    pprint(roots_datasets)

    workers = cpu_count()
    print("Number of workers:", workers)
    pool = Pool(workers)
    pool.map(partial(save_dataset, base_dir=args.dir), roots_datasets)
    pool.close()
    pool.join()
