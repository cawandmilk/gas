import argparse
import itertools
import json
import pprint

import pandas as pd

from operator import itemgetter
from pathlib import Path
from tqdm import tqdm

from src.clean import CleanNewspaperArticle


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--raw_train",
        required=True,
    )
    p.add_argument(
        "--raw_valid",
        required=True,
    )
    p.add_argument(
        "--raw_test",
        required=True,
    )
    p.add_argument(
        "--data",
        type=str,
        default="data",
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

    for sample in Path(raw_path).glob("*.json*"):
        ## Backdoor.
        if not Path(sample).name.endswith(".jsonl") and not Path(sample).name.startswith("신문기사"):
            continue

        if str(sample).endswith(".json"):
            documents += _read_json(sample)
        elif str(sample).endswith(".jsonl"):
            documents += _read_jsonl(sample)
        else:
            ## We only allow json, jsonl files.
            raise AssertionError(f"Only '*.json' and '*.jsonl' files allowed: not :{sample}")

    ## Sort by ids.
    if sort_dicts:
        documents = sorted(
            documents,
            key=itemgetter("id"),
            reverse=False,
        )
    
    return documents


def extract_lines(config, documents: list) -> dict:
    ## We will save to jsonl files.
    extracted_features = []
    cleaner = CleanNewspaperArticle()

    for document in tqdm(documents, total=len(documents)): 
        ## In train, valid dataset, we can get key "text".
        if document.get("text") != None:
            f = {
                "id": document["id"],
                # "text": " ".join(cleaner(
                #     [line["sentence"].strip() for line in itertools.chain(*document["text"])], 
                #     document["media_name"],
                # )),
                "text": " ".join([line["sentence"].replace("\n", " ").strip() for line in itertools.chain(*document["text"])]), 
                "summary": document["abstractive"][0], ## list -> element (str)
            }
            ## Like "id" == "362852732", no abstractive summaries exists.
            if f["summary"] == "":
                continue

        ## In test dataset, we can get key "article_original".
        elif document.get("article_original") != None:
            f = {
                "id": document["id"],
                # "text": " ".join(cleaner(
                #     [i.strip() for i in document["article_original"]], 
                #     document["media"],
                # )),
                "text": " ".join([i.replace("\n", " ").strip() for i in document["article_original"]]),
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
