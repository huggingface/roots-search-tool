import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import fasttext
from bigscience_pii_detect_redact import run_pii
from huggingface_hub import hf_hub_download
from pyserini.analysis import Analyzer
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene.querybuilder import JTerm, get_phrase_query_builder

LANGUAGES = [
    "ar",
    "ca",
    "code",
    "en",
    "es",
    "eu",
    "fr",
    "id",
    "indic",
    "nigercongo",
    "pt",
    "vi",
    "zh",
]

MAX_DOCS = 5


def get_phrase_query(query, lang):
    phrase_query_builder = get_phrase_query_builder()
    words = query.split()
    for i, word in enumerate(words):
        terms = self.server.analyzer[lang].analyze(word)
        if len(terms) == 0:
            continue
        word = terms[0]
        phrase_query_builder.add(JTerm("contents", word))
    return phrase_query_builder.build()


class LanguageDetector:
    def __init__(self):
        self.fasstex = fasttext.load_model(hf_hub_download("julien-c/fasttext-language-id", "lid.176.bin"))

    def identify_lang(self, query):
        preds = self.fasstex.predict(query)
        return preds[0][0][len("__label__") :], preds[1][0]


class ThreadedPyseriniHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

    def __init__(self, server_address, handler_class, index_dir):
        super().__init__(server_address, handler_class)
        self.lang_detector = LanguageDetector()

        logging.info("initializing lucene")
        self.searcher = {}
        self.analyzer = {}
        for lang in LANGUAGES:
            self.searcher[lang] = LuceneSearcher(index_dir + "/{}/".format(lang))
            self.searcher[lang].set_language(lang)
            # self.searcher[lang].set_rm3(debug=True)
            # self.searcher[lang].set_rocchio(debug=True)
            self.analyzer[lang] = Analyzer(self.searcher[lang].object.analyzer)


class PyseriniHTTPRequestHandler(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def _process_hits(self, hits, lang, query_terms, highlight_terms=None):
        hits_entries = []
        if highlight_terms is None:
            highlight_terms = set()
        for hit in hits:
            raw = json.loads(hit.raw)
            hits_entry = {}
            hits_entry["docid"] = hit.docid
            hits_entry["score"] = hit.score
            hits_entry["text"] = raw["contents"]
            if "meta" in raw:
                hits_entry["meta"] = raw["meta"]
            actual_lang = lang
            if lang in ["indic", "nigercongo"]:
                actual_lang = hit.docid.split("_")[1].split("-")[1]
            hits_entry["lang"] = actual_lang
            piis = run_pii(hits_entry["text"], actual_lang)
            if len(piis[1]) > 0:
                hits_entry["text"] = piis[1]["redacted"]
            hits_entries.append(hits_entry)
            hit_terms = hits_entry["text"].split()
            for term in hit_terms:
                term_rewritten = set(self.server.analyzer[lang].analyze(term))
                if len(query_terms & term_rewritten) > 0:
                    highlight_terms.add(term)
        logging.info("Highlight terms: {}".format(str(highlight_terms)))
        return hits_entries, highlight_terms

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length).decode("utf-8")
        logging.info("POST request,\nPath: {}\nHeaders:\n{}\nBody:\n{}\n".format(self.path, self.headers, post_data))

        post_data = json.loads(post_data)
        if "flag" in post_data and bool(post_data["flag"]):
            # TODO: improve reporting
            self._set_response()
            self.wfile.write(json.dumps("Flagging OK").encode("utf-8"))
            return

        exact_search = False
        if "exact_search" in post_data and post_data["exact_search"]:
            exact_search = True

        query = post_data["query"]
        k = MAX_DOCS if "k" not in post_data or post_data["k"] is None else int(post_data["k"])

        if "lang" in post_data and post_data["lang"] != "" and post_data["lang"] is not None:
            lang = post_data["lang"]
            score = 1.0
        else:
            lang, score = self.server.lang_detector.identify_lang(query)
            if lang not in LANGUAGES:
                self._set_response()
                self.wfile.write(
                    json.dumps({"err": {"type": "unsupported_lang", "meta": {"detected_lang": lang}}}).encode("utf-8")
                )
                return

        logging.info("Query: {}".format(query))
        logging.info("Detected language {}, with score {}".format(lang, score))

        results = None
        if lang == "all":
            logging.info("Querying all available indices")
            highlight_terms = set()
            results = {}
            for lang in LANGUAGES:
                logging.info("Processing langguage: {}".format(lang))
                query_terms = set(self.server.analyzer[lang].analyze(query))
                if exact_search:
                    query = get_phrase_query(query, lang)
                hits_entries, new_highlight_terms = self._process_hits(
                    self.server.searcher[lang].search(query, k=k), lang, query_terms, highlight_terms
                )
                highlight_terms = highlight_terms | new_highlight_terms
                results[lang] = hits_entries
        else:
            query_terms = set(self.server.analyzer[lang].analyze(query))
            if exact_search:
                query = get_phrase_query(query, lang)
            results, highlight_terms = self._process_hits(
                self.server.searcher[lang].search(query, k=k), lang, query_terms
            )

        payload = {"results": results, "highlight_terms": list(highlight_terms)}
        self._set_response()
        self.wfile.write(json.dumps(payload).encode("utf-8"))


def run(index_dir, server_address, port):
    logging.basicConfig(level=logging.INFO)
    httpd = ThreadedPyseriniHTTPServer((server_address, port), PyseriniHTTPRequestHandler, index_dir)

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
        help="Address of the server, e.g. 'http://12.345.678.910'",
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
