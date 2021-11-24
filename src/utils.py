import numpy as np
import pandas as pd

from pathlib import Path


def read_text(config, fpath: Path):
    ## Parse names.
    inp_suffix = config.lang[:2]
    tar_suffix = config.lang[2:]

    texts_fpath = Path(fpath.parent, ".".join([*fpath.name.split("."), inp_suffix]))
    summaries_fpath = Path(fpath.parent, ".".join([*fpath.name.split("."), tar_suffix]))

    with open(texts_fpath, "r", encoding="utf-8") as f:
        texts = f.readlines()

    for i in range(len(texts)):
        texts[i] = " ".join(texts[i].strip().split("\t"))

    return_value = {
        "texts": texts,
    }

    ## If target summaries are exist...
    if summaries_fpath.exists():
        with open(summaries_fpath, "r", encoding="utf-8") as f:
            summaries = f.readlines()

        for i in range(len(summaries)):
            summaries[i] = " ".join(summaries[i].strip().split("\t"))

        return_value["summaries"] = summaries

    return return_value


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad != None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data ** norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data ** norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def save_predictions(sample_submission_path: Path, predictions: list, save_to: Path) -> Path:
    ## Read a sample file.
    df = pd.read_csv(sample_submission_path, index_col=False)

    ## Record it.
    ## Thus test datasets are already sorted by 'id', we don't need to
    ## worry about shuffing.
    df.loc[:, "summary"] = np.array(predictions)
    
    ## Strip.
    df.loc[:, "summary"] = df.loc[:, "summary"].apply(lambda x: x.strip())

    ## Save.
    save_to.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_to, index=False)

    return save_to
