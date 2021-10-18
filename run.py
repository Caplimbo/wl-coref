""" Runs experiments with CorefModel.

Try 'python run.py -h' for more details.
"""

import argparse
import json
import jsonlines
from contextlib import contextmanager
import datetime
import random
import sys
import time

import numpy as np  # type: ignore
import torch  # type: ignore

from coref import CorefModel
from transformers import RobertaTokenizerFast
import spacy
from spacy.tokens import Doc


class RbaTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def __call__(self, text):
        words = self.tokenizer.tokenize(text)
        # spacy.vocab.Vocab(self.tokenizer.get_vocab())
        return Doc(self.vocab, words=words)


# tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
NLP = spacy.load("en_core_web_sm", disable=["ner"])


# NLP.tokenizer = RbaTokenizer(NLP.vocab)


@contextmanager
def output_running_time():
    """Prints the time elapsed in the context"""
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        print(f"Total running time: {delta}")


def seed(value: int) -> None:
    """Seed random number generators to get reproducible results"""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def preprocess_doc(doc, tokenizer):
    token_map = {
        ".": ["."],
        ",": [","],
        "!": ["!"],
        "?": ["?"],
        ":": [":"],
        ";": [";"],
        "'s": ["'s"],
    }
    filter_func = lambda _: True
    # doc["span_clusters"] = [[tuple(mention) for mention in cluster]
    #                         for cluster in doc["span_clusters"]]
    word2subword = []
    subwords = []
    word_id = []
    for i, word in enumerate(doc["cased_words"]):
        tokenized_word = (
            token_map[word] if word in token_map else tokenizer.tokenize(word)
        )
        tokenized_word = list(filter(filter_func, tokenized_word))
        word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
        subwords.extend(tokenized_word)
        word_id.extend([i] * len(tokenized_word))
    doc["word2subword"] = word2subword
    doc["subwords"] = subwords
    doc["word_id"] = word_id
    return doc


def process_article(article: str):
    doc = {}
    doc["document_id"] = "nw001"
    processed = NLP(article)
    doc["cased_words"] = [token.text for token in processed]
    sents = [sent for sent in processed.sents]
    # print([sent for sent in sents])
    doc["sent_id"] = [sents.index(token.sent) for token in processed]
    doc["speaker"] = [None] * len(doc["cased_words"])
    doc["span_clusters"] = [[1, 2], [2, 3]]
    return doc


def run_on_articles(articles, device="cuda", batch_size=512):
    print("Loading Model...")

    model = CorefModel(CONFIG_FILE, EXPERIMENT_MODEL)

    model.config.a_scoring_batch_size = batch_size
    model.config.device = device
    load_device = lambda storage, loc: storage.cuda(0) if device != "cpu" else device
    model.load_weights(
        path=WEIGHT_FILE,
        map_location="cpu",
        ignore={
            "bert_optimizer",
            "general_optimizer",
            "bert_scheduler",
            "general_scheduler",
        },
    )
    print("Model Loaded!")

    docs = [preprocess_doc(process_article(article), model.tokenizer) for article in articles]
    start = time.time()
    for article, doc in zip(articles, docs):
        res = model.run(doc)
        # all_span_clusters = res.span_clusters
        # processed = NLP(article)
        # for cluster in all_span_clusters:
        #     for span in cluster:
        #         print(f"Span: {span}")
        #         print(f"word: {processed[span[0]: span[1]]}")
        #     print("----------------------------------")
        torch.cuda.empty_cache()
        del res
    end = time.time()
    return end-start


def read_articles_from_json_file(file_path):
    print("Reading from file...")
    articles = []
    with open(file_path, "r") as inf:
        for entry in jsonlines.Reader(inf):
            articles.append(entry['text'])
    print(f"Done Reading! {len(articles)} articles in all to process!")
    return articles

if __name__ == "__main__":
    BATCH_SIZE = 512
    CONFIG_FILE = "config.toml"
    WEIGHT_FILE = "data/roberta_(e20_2021.05.02_01.16)_release.pt"
    EXPERIMENT_MODEL = "roberta"

    seed(2020)

    articles = read_articles_from_json_file("../batch-processing/clients/articles.json")
    device = input("DEVICE: ")
    batch_size = int(input("BATCH_SIZE: "))
    num_articles = int(input("Number of Articles: "))
    duration = run_on_articles(articles[:num_articles], device=device, batch_size=batch_size)
    print(f"Running for {num_articles} articles took {duration} seconds.")
