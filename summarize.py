import torch

from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

import argparse
import os
import pprint

import numpy as np

from pathlib import Path
from tqdm import tqdm

from finetune_plm_hftrainer import get_datasets
from src.utils import save_predictions


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_fpath", 
        required=True,
    )
    p.add_argument(
        "--test",
        required=True,
        help="Training set file name except the extention. (ex: train.en --> train)",
    )
    p.add_argument(
        "--gpu_id", 
        type=int, 
        default=-1,
    )
    p.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
    )
    p.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for beam search. Default=%(default)s",
    )
    p.add_argument(
        "--length_penalty",
        type=float,
        default=1.2,
    )
    p.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
    )
    p.add_argument(
        "--inp_max_len", 
        type=int, 
        default=1024,
    )
    p.add_argument(
        "--tar_max_len", 
        type=int, 
        default=128,
    )
    p.add_argument(
        "--sample_submission_path",
        type=str,
        default=os.path.join("data", "raw", "Test2", "new_sample_submission.csv"),
    )
    p.add_argument(
        "--submission_path",
        type=str,
        default="submission",
    )
    p.add_argument(
        "--auto_submit",
        action="store_true",
    )

    config = p.parse_args()

    return config


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(config)
    print_config(vars(config))

    ## Load data.
    saved_data = torch.load(
        config.model_fpath,
        map_location="cpu" if config.gpu_id < 0 else "cuda:%d" % config.gpu_id,
    )

    bart_best = saved_data["bart"]
    train_config = saved_data["config"]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(train_config.pretrained_model_name)

    ## Get datasets and index to label map.
    ts_ds = get_datasets(config, tokenizer, fpath=Path(config.test), shuffle=False, mode="test")
    ts_loader = torch.utils.data.DataLoader(
        ts_ds,
        batch_size=config.batch_size,
        shuffle=False,
    )

    ## We will not get lines by stdin, but file path.
    # lines = read_text()

    with torch.no_grad():
        ## Declare model and load pre-trained weights.
        ## You can get tokenizer from saved_data['tokenizer'].
        # tokenizer = PreTrainedTokenizerFast.from_pretrained(train_config.pretrained_model_name)
        model = BartForConditionalGeneration.from_pretrained(train_config.pretrained_model_name)
        model.load_state_dict(bart_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        ## Don't forget turn-on evaluation mode.
        model.eval()

        outputs = []
        for mini_batch in tqdm(ts_loader, total=len(ts_loader)):
            input_ids = mini_batch["input_ids"].to(device)
            # attention_mask = mini_batch["attention_mask"].to(device)

            ## Generate ids of summaries.
            ##   - https://huggingface.co/transformers/v2.11.0/model_doc/bart.html#transformers.BartForConditionalGeneration.generate
            output = model.generate(
                input_ids, 
                # attention_mask=attention_mask,
                # bos_token_id=tokenizer.bos_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                max_length=config.tar_max_len,          ## maximum summarization size
                min_length=config.tar_max_len // 4,     ## minimum summarization size
                early_stopping=True,                    ## stop the beam search when at least 'num_beams' sentences are finished per batch
                num_beams=config.beam_size,             ## beam search size
                eos_token_id=tokenizer.eos_token_id,    ## 1
                length_penalty=config.length_penalty,   ## value > 1.0 in order to encourage the model to produce longer sequences
                no_repeat_ngram_size=config.no_repeat_ngram_size,   ## same as 'trigram blocking'
            )
            ## If you want to decode by each sentence, you may 
            ## call 'decode' fn, not 'batch_decode'.
            output = tokenizer.batch_decode(
                output.tolist(), 
                skip_special_tokens=True,
            )

            ## Get all.
            outputs.extend(output)

    ## Save it.
    save_to = Path(
        config.submission_path,                                 ## submission/
        ".".join([
            Path(config.model_fpath).parts[-2],                 ## datetime
            *Path(config.model_fpath).name.split(".")[:-1],     ## moel_fpath
            f"LP-{config.length_penalty:.1f}",                  ## length penalty
            "csv",
        ]),
    )
    save_fpath = save_predictions(
        sample_submission_path=config.sample_submission_path, 
        predictions=outputs,
        save_to=save_to,
    )
    print(f"Submission save to: {save_to}")

    ## Submit.
    if config.auto_submit:
        from dacon_submit_api import dacon_submit_api
        ## Read personal submission token.
        with open("token", "r", encoding="utf-8") as f:
            token = f.readlines().strip()

        result = dacon_submit_api.post_submission_file(
            file_path=save_fpath,
            token=token, 
            cpt_id=235829, 
            team_name="이야기연구소 주식회사", 
            memo="",
        )
        print_config(result)

if __name__ == "__main__":
    config = define_argparser()
    main(config)
