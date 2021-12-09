import argparse
import itertools
import json
import pprint

import pandas as pd

from operator import itemgetter
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from src.clean import CleanNewspaperArticle


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--raw_train",
        required=True,
        help=" ".join([
            "The name to be saved as training *.tsv file.",
            "The column will contains [id, text, summary]."
            "(ex: ./data/raw/Training)",
        ]),
    )
    p.add_argument(
        "--raw_valid",
        required=True,
        help=" ".join([
            "The name to be saved as validate *.tsv file.",
            "The column will contains [id, text, summary]."
            "(ex: ./data/raw/Validation)",
        ]),
    )
    p.add_argument(
        "--raw_test",
        required=True,
        help=" ".join([
            "The name to be saved as test *.tsv file.",
            "The column will contains [id, text]."
            "(ex: ./data/raw/Test)",
        ]),
    )
    p.add_argument(
        "--data",
        type=str,
        default="data",
        help=" ".join([
            "This is the directory where preprocessed tsv files will be saved.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help=" ".join([
            "Whether to proceed with text cleaning. Preprocessing only occurs",
            "on the input text, not on the output summary.",
            "Default=%(default)s",
        ]),
    )
    
    config = p.parse_args()

    return config


def read_samples(raw_path: list, sort_dicts: bool = True) -> list:
    
    def _read_json(_sample: str) -> list:
        with open(_sample, "r", encoding="utf-8") as f:
            document = json.loads(f.read())["documents"]
        return document

    def _read_jsonl(_sample: str) -> list:
        with open(_sample, "r", encoding="utf-8") as f:
            documents = [json.loads(line) for line in f]
        return documents

    ## Empty list.
    documents = []

    ## Get datasets.
    for sample in Path(raw_path).glob("*.json*"):
        ## We only use "신문기사" dataset in total trainin & validation sets.
        if Path(sample).name.startswith("신문기사"):
            documents += _read_json(sample)
            break

        ## We only allow "*.jsonl" file in test set.
        elif Path(sample).name.endswith(".jsonl"):
            documents += _read_jsonl(sample)
            break

        else:
            continue
    
    if documents == []:
        raise AssertionError(f"Raw path has no '*.json' or '*.jsonl' file: {raw_path}")

    ## Sort by ids.
    if sort_dicts:
        documents = sorted(
            documents,
            key=itemgetter("id"),
            reverse=False,
        )
    
    return documents


def extract_lines(config, documents: List[Dict]) -> dict:
    ## We will save to jsonl files.
    extracted_features = []
    cleaner = CleanNewspaperArticle()

    for document in tqdm(documents, total=len(documents)): 
        ## In train, valid dataset, we can get key "text".
        if document.get("text") != None:
            id = document["id"]

            text = [line["sentence"].replace("\n", " ").strip() for line in itertools.chain(*document["text"])]
            if config.clean:
                text = cleaner(text, document["media_name"])
            text = " ".join(text)

            summary = document["abstractive"][0].replace("\n", " ").strip()

            ## If a document is total-advertisement, it can be skipped.
            ## Like "id" == "362852732", no abstractive summaries exists.
            if text == "" or summary == "":
                continue

            ## Wrong preprocessed article. (무등일보)
            if id == "338143341":
                continue

            ## Gather all.
            f = {
                "id": id,
                "text": text,
                "summary": summary,
            }

        ## In test dataset, we can get key "article_original".
        elif document.get("article_original") != None:
            id = document["id"]

            text = [line.replace("\n", " ").strip() for line in document["article_original"]]
            ## assert document["media"] in cleaner.media_name_to_function.keys()
            if config.clean:
                text = cleaner(text, document["media"])
            text = " ".join(text)

            if text == "":
                raise AssertionError(f"'text' must not be empty: id={id}")

            f = {
                "id": id,
                "text": text,
            }

        else:
            raise AssertionError(f"Document must have 'text' or 'article_original' key: {document.keys()}")

        extracted_features.append(f)

    return extracted_features


def save_lines(config, mode: str, documents: list) -> None:
    assert mode in ["train", "valid", "test"]

    ## Save as: ./data/train.tsv, ./data/valid.tsv, ./data/test.tsv.
    fpath = Path.cwd() / Path(config.data, f"{mode}.tsv")
    pd.DataFrame(documents).to_csv(fpath, sep="\t", encoding="utf-8")

    print(f"File {fpath} saved.")


def main(config):
    def print_config(config) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Read original json/jsonl files.
    tr_corpus = read_samples(config.raw_train)
    vl_corpus = read_samples(config.raw_valid)
    ts_corpus = read_samples(config.raw_test)

    ## Extract features.
    tr_documents = extract_lines(config, tr_corpus)
    vl_documents = extract_lines(config, vl_corpus)
    ts_documents = extract_lines(config, ts_corpus)

    save_lines(config, "train", tr_documents)
    save_lines(config, "valid", vl_documents)
    save_lines(config, "test",  ts_documents)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
