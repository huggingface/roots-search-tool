import argparse
import ast
import glob
# import jnius
import json
import multiprocessing
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from pprint import pprint

# import jnius_config
import jsonlines
from tqdm import tqdm
from utils import (WHITESPACE_COMPATIBLE, get_datasets_with_prefix,
                   get_json_path, get_langauge)


# split a list in num parts evenly
def chunk_it(seq, num):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num  # 0 <= diff < num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks


def _run_thread(arguments):
    index = arguments["index"]
    k = arguments["k"]
    data = arguments["data"]

    # BM25 parameters #TODO
    # bm25_a = arguments["bm25_a"]
    # bm25_b = arguments["bm25_b"]
    # searcher.set_bm25(bm25_a, bm25_b)

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(index)

    provenance = []
    for document in tqdm(data):
        query = document["query"]
        try:
            hits = searcher.search(query, k=100)
            results = []
            for i in range(len(hits)):
                results.append(
                    {
                        "id": hits[i].docid,
                        "score": hits[i].score,
                        "passage": ast.literal_eval(hits[i].raw)["contents"],
                    }
                )
            document["results"] = results
            provenance.append(document)
        except Exception as e:
            print(query)
            print(e)

    return provenance


class BM25:
    def __init__(self, index, k, num_threads=None, Xms="32g", Xmx="40g"):

        """
        if Xms and Xmx:
            # to solve Insufficient memory for the Java Runtime Environment
            jnius_config.add_options("-Xms{}".format(Xms), "-Xmx{}".format(Xmx), "-XX:-UseGCOverheadLimit")
            print("Configured options:", jnius_config.get_options())
        """

        self.num_threads = multiprocessing.cpu_count() if num_threads is None else num_threads

        # initialize a ranker per thread
        self.arguments = []
        for id in tqdm(range(self.num_threads)):
            self.arguments.append(
                {"id": id, "index": index, "k": k,}
            )

    def feed_data(self, queries_data, logger=None):
        chunked_queries = chunk_it(queries_data, self.num_threads)

        for idx, arg in enumerate(self.arguments):
            arg["data"] = chunked_queries[idx]

    def run(self):
        pool = ThreadPool(self.num_threads)
        results = pool.map(_run_thread, self.arguments)

        provenance = []
        for x in results:
            provenance.extend(x)
        pool.terminate()
        pool.join()

        return provenance


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
        default="ict-predictions",
        choices=["ict-predictions"],
        help="Preprocessing task - pyserini or mongodb",
    )
    parser.add_argument("--dir", type=str, default="data/", help="Path to a directory where results will be stored.")
    parser.add_argument(
        "--data_org", type=str, default="bigscience-data", help="Huggingface hub organization maintaining the data."
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="/home/piktus_huggingface_co/bigscience/scisearch/data/bigscience-data-index/",
    )
    parser.add_argument(
        "--k", type=int, default=100,
    )
    parser.add_argument(
        "--lang", required=True, type=str,
    )
    args = parser.parse_args()

    data_org = args.data_org
    base_dir = args.dir
    prefix = data_org + "/roots_" + args.lang

    datasets = get_datasets_with_prefix(prefix=prefix, use_auth_token=HUGGINGFACE_TOKEN)
    print("Processing {} datasets matching prefix: {}".format(len(datasets), prefix))
    pprint(datasets)

    retriever = BM25(args.index_dir + args.lang, args.k)

    for dataset_name in datasets:
        query_filename = get_json_path(dataset_name, base_dir, data_org, "ict-1500")
        results_filename = get_json_path(dataset_name, base_dir, data_org, "ict-1500-predictions")

        if os.path.isfile(results_filename):
            print("Skipping", dataset_name + "- already processed.")
            continue

        print("Processing", dataset_name, "to be saved under", results_filename)

        reader = jsonlines.open(query_filename, mode="r")
        input_lines = [line for line in reader]
        retriever.feed_data(input_lines)
        provenance = retriever.run()

        writer = jsonlines.open(results_filename, mode="w")
        writer.write_all(provenance)
        writer.close()
