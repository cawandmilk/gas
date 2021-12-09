import argparse
import json
import pprint

import numpy as np

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

    config = p.parse_args()

    return config


def read_samples(raw_path: list, sort_dicts: bool = True) -> dict:
    
    def _read_json(_sample: str) -> list:
        with open(_sample, "r", encoding="utf-8") as f:
            document = json.loads(f.read())["documents"]
        return document

    def _read_jsonl(_sample: str) -> list:
        with open(_sample, "r", encoding="utf-8") as f:
            documents = [json.loads(line) for line in f]
        return documents

    ## Empty list.
    # documents = {}

    # for sample in Path(raw_path).glob("*.json*"):
    #     if not str(sample).endswith(".json"):
    #         raise AssertionError(f"Only '*.json' and '*.jsonl' files allowed: not :{sample}")

    #     ftype = Path(sample).name.split("_")[0]
    #     documents[ftype] = _read_json(sample)

    #     ## Sort by ids.
    #     if sort_dicts:
    #         documents[ftype] = sorted(
    #             documents[ftype],
    #             key=itemgetter("id"),
    #             reverse=False,
    #         )

    for sample in Path(raw_path).glob("*.json*"):
        if not str(sample).endswith(".jsonl"):
            continue
            ## raise AssertionError(f"Only '*.json' and '*.jsonl' files allowed: not :{sample}")

        documents = _read_jsonl(sample)

        ## Sort by ids.
        if sort_dicts:
            documents = sorted(
                documents,
                key=itemgetter("id"),
                reverse=False,
            )

    return documents


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(config)
    print_config(vars(config))

    ## Parse original samples.
    tr_corpus = read_samples(config.raw_train_path)
    # vl_corpus = read_samples(config.raw_valid_path)
    # ts_corpus = read_samples(config.raw_test_path)

    # def _get_media_names(corpus: list, key: str = "media_name") -> list:
    #     return np.array([i.get("media_name") for i in corpus if i.get("media_name") != None])

    ## Size.
    # print("#(tr_corpus):")
    # for key, value in tr_corpus.items():
    #     print(f"  - {key}: {len(value)}")

        # unique_media_names = np.unique(_get_media_names(value))
        # print(f"  - unique media_name in {key}: {len(unique_media_names)}")

    # print("#(vl_corpus):")
    # for key, value in vl_corpus.items():
    #     print(f"  - {key}: {len(value)}")

        # unique_media_names = np.unique(_get_media_names(value))
        # print(f"  - unique media_name in {key}: {len(unique_media_names)}")

    ## Dataset per media_type.
    # for i, corpus in enumerate([tr_corpus, vl_corpus]):
    #     split = {0: "train", 1: "valid"}[i]
    #     for key, value in corpus.items():
    #         media_names = _get_media_names(value)
    #         nums = [{
    #             "media_name": media_name,
    #             "num": len(np.argwhere(media_names == media_name)),
    #         } for media_name in np.unique(media_names)]
    #         sorted_nums = sorted(
    #             nums,
    #             key=itemgetter("num"),
    #             reverse=True,
    #         )

    #         ## Show as markdown table format.
    #         for j in sorted_nums:
    #             media_name = j["media_name"]
    #             num = j["num"]
    #             print(f"|{split}|{key}|{media_name}|{num:,}|{num/len(media_names)*100:.2f}|")



    ## We need to eda in 'train corpus', not 'valid' or 'test' corpus.
    for category in ["신문기사"]:
        splited_documents = {}

        for document in tr_corpus[category]:
            media_name = document.get("media_name")
            ## If the 'key' is already in 'splited_documents'...
            if splited_documents.get(media_name) != None:
                ## Just append to list.
                splited_documents[media_name].append(document)
            else:
                ## Add key-value as list.
                splited_documents[media_name] = [document]

        ## Save by 'media_name'.
        save_path = Path(config.data, "split", category)
        save_path.mkdir(parents=True, exist_ok=True)

        for media_name in splited_documents.keys():
            with open(save_path / Path(media_name + ".json"), "w", encoding="utf-8") as f:
                json.dump(splited_documents[media_name], f, indent=4, ensure_ascii=False)

    # save_path = Path(config.raw_test_path, "new_test_.json")
    # with open(save_path, "w", encoding="utf-8") as f:
    #     json.dump(ts_corpus, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
