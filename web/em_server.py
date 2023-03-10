import argparse
import json
import logging
import os
import string
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import fasttext
from bigscience_pii_detect_redact import run_pii
from huggingface_hub import HfApi

import argparse
import pickle
import subprocess

from bisect import bisect_right
from datasets import load_from_disk
from tqdm import tqdm


hf_api = HfApi()
roots_datasets = {
    dset.id.split("/")[-1]: dset
    for dset in hf_api.list_datasets(
        author="bigscience-data",
        use_auth_token=os.environ.get("BIGSCIENCE_DATA_ACCESS_TOKEN"),
    )
}


class ThreadedPyseriniHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

    def __init__(self, server_address, handler_class, index_dir):
        super().__init__(server_address, handler_class)

        logging.info("initializing suffix arrays")
        self.pos2id = pickle.load(
            open(
                "/home/piktus_huggingface_co/lumi/dedup/roots_all_10/roots_all.train.pos2id.pkl",
                "rb",
            )
        )
        self.pos2id_list = sorted(self.pos2id.keys())
        self.datasets = {}
        for dataset in roots_datasets.keys():
            if dataset.startswith("roots") and (not dataset.startswith("roots-1e")):
                self.datasets[dataset] = load_from_disk(
                    "/home/piktus_huggingface_co/lumi/roots_all/" + dataset
                )["train"]

    def get_doc_for_pos(self, pos):
        """
        Gets id of the datapoint at position.
        """
        pos = bisect_right(self.pos2id_list, pos)
        dataset_name, docid = self.pos2id[self.pos2id_list[pos - 1]]
        return dataset_name, docid, self.datasets[dataset_name][docid]


class PyseriniHTTPRequestHandler(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def _process_result(self, doc, query):
        def find_whitespace(text):
            for i, c in enumerate(text):
                if c in string.whitespace:
                    yield i

        text = doc["text"]
        pos = text.find(query)
        whitespace_idx = [-1] + list(find_whitespace(text)) + [len(text)]
        # print(whitespace_idx)
        idx = bisect_right(whitespace_idx, pos)
        # print("idx", idx)
        start = whitespace_idx[max(0, idx - 50)] + 1
        end = whitespace_idx[min(len(whitespace_idx) - 1, idx + 50)]
        # print("start: {}, end: {}".format(start, end))
        processed_text = text[start:end]
        piis = run_pii(processed_text, None)
        if len(piis[1]) > 0:
            processed_text = piis[1]["redacted"]
        return processed_text

    def do_GET(self):
        logging.info(
            "GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers)
        )
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length).decode("utf-8")
        logging.info(
            "POST request,\nPath: {}\nHeaders:\n{}\nBody:\n{}\n".format(
                self.path, self.headers, post_data
            )
        )

        post_data = json.loads(post_data)
        query = post_data["query"]
        k = post_data["k"]
        received_results = post_data["received_results"]

        logging.info(
            "Query: {}, k: {}, received_results: {}".format(query, k, received_results)
        )

        query_bytes = query.encode("utf-8")
        tmp_file = "/tmp/fin_{}".format(uuid.uuid4())
        open(tmp_file, "wb").write(query_bytes)

        cmd = (
            "/home/piktus_huggingface_co/lumi/deduplicate-text-datasets/target/debug/dedup_dataset count-occurrences "
            "--data-file  /home/piktus_huggingface_co/lumi/dedup/roots_all_10/roots_all.train "
            "--query-file {}".format(tmp_file)
        )
        print(cmd)
        cmd_result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
        lines = cmd_result.stdout.decode("utf-8").split("\n")
        prefix = "Found at: "
        positions = [
            int(line[len(prefix) :].strip())
            for line in lines[received_results : received_results + k]
            if line.startswith(prefix)
        ]

        results = []
        for pos in tqdm(positions):
            dataset_name, docid, doc = self.server.get_doc_for_pos(pos)
            results.append(
                {
                    "text": self._process_result(doc, query),
                    "docid": "/" + dataset_name + "/" + str(docid),
                }
            )

        payload = {"results": results, "num_results": max(0, len(lines) - 4), "lang": "all"}
        self._set_response()
        self.wfile.write(json.dumps(payload).encode("utf-8"))


def run(index_dir, server_address, port):
    logging.basicConfig(level=logging.INFO)
    httpd = ThreadedPyseriniHTTPServer(
        (server_address, port), PyseriniHTTPRequestHandler, index_dir
    )

    sa = httpd.socket.getsockname()
    logging.info("Starting httpd on {} port {} ...".format(sa[0], sa[1]))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--index_dir",
        type=str,
        default="data/bigscience-data-index-128/",
        help="Path to the directory containing indices for respective languages",
    )
    parser.add_argument(
        "-a",
        "--server_address",
        required=True,
        type=str,
        help="Address of the server, e.g. '12.345.678.910'",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8080,
        help="Port on which to serve ",
    )
    args = parser.parse_args()
    run(args.index_dir, args.server_address, args.port)
