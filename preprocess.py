import argparse
import itertools
import json
import pprint
import re

from operator import itemgetter
from pathlib import Path


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--raw_train_path",
        required=True,
    )
    p.add_argument(
        "--raw_valid_path",
        required=True,
    )
    p.add_argument(
        "--raw_test_path",
        required=True,
    )
    p.add_argument(
        "--data",
        type=str,
        default="data",
    )
    p.add_argument(
        "--inp_suffix",
        type=str,
        default="if",
    )
    p.add_argument(
        "--tar_suffix",
        type=str,
        default="of",
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


def text_cleaning(text: str) -> str:
    ## Ref. https://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221347960543
    
    ## Remove email.
    pattern = "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)" 
    text = re.sub(pattern=pattern, repl="", string=text)
    
    ## Remove URL.
    pattern = "(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
    text = re.sub(pattern=pattern, repl="", string=text)
    
    ## Stand-alone korean 자음/모음.
    pattern = "([ㄱ-ㅎㅏ-ㅣ]+)"
    text = re.sub(pattern=pattern, repl="", string=text)
    
    ## HTML tags. -> ???
    # pattern = "<[^>]*>"
    # text = re.sub(pattern=pattern, repl="", string=text)
    
    ## Specail words.
    # pattern = "[^\w\s]"
    # text = re.sub(pattern=pattern, repl="", string=text)
    
    ## Strip and remove double space, line feed, carrage returns.
    text = " ".join([i.strip() for i in text.split()])

    return text


def extract_lines(config, documents: list) -> tuple:
    ## We will save to jsonl files.
    extracted_features = []

    for document in documents: 
        ## In train, valid dataset, we can get key "text".
        if document.get("text") != None:
            f = {
                "id": document["id"],
                "text": "\t".join([text_cleaning(line["sentence"]) for line in itertools.chain(*document["text"])]),
                "summary": document["abstractive"][0], ## list -> element (str)
            }

        ## In test dataset, we can get key "article_original".
        elif document.get("article_original") != None:
            f = {
                "id": document["id"],
                "text": "\t".join([text_cleaning(line) for line in document["article_original"]]),
            }

        else:
            raise AssertionError(f"Document must have 'text' or 'article_original' key: {document.keys()}")

        extracted_features.append(f)

    return extracted_features


def save_lines(config, mode: str, documents: list) -> None:
    assert mode in ["train", "valid", "test"]

    ## Naming.
    ## Suffixes "if" and "of" are from linux command "dd".
    ## It's not recommand to naming more then three words like "inp" & "tar".
    ##   ex) {train | source} -> corpus.train.ip
    ##   ex) {valid | target} -> corpus.valid.of
    text_fname = ".".join(["corpus", mode, config.inp_suffix])
    text_fpath = Path.cwd() / Path(config.data, text_fname)

    summary_fname = ".".join(["corpus", mode, config.tar_suffix])
    summary_fpath = Path.cwd() / Path(config.data, summary_fname)

    ## Write texts.
    with open(text_fpath, "w", encoding="utf-8") as f:
        context = "\n".join([document["text"] for document in documents])
        f.write(context)

    print(f"File {text_fpath} saved.")

    ## If summary is exists, then write it.
    if documents[0].get("summary") != None:
        with open(summary_fpath, "w", encoding="utf-8") as f:
            context = "\n".join([document["summary"] for document in documents])
            f.write(context)
    
        print(f"File {summary_fpath} saved.")


def main(config):
    def print_config(config) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Read original json/jsonl files.
    tr_corpus = read_samples(config.raw_train_path)
    vl_corpus = read_samples(config.raw_valid_path)
    ts_corpus = read_samples(config.raw_test_path)

    ## Check if it really sorted or not.
    # pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(
    #     [document["id"] for document in ts_corpus[:30]],
    # )

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
