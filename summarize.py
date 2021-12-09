import torch

from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

import argparse
import json
import os
import pprint

from operator import itemgetter
from pathlib import Path
from tqdm import tqdm

from finetune_plm_hftrainer import get_datasets
from src.bart_dataset import TextAbstractSummarizationCollator
from src.utils import save_predictions


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_fpath", 
        required=True,
        help=" ".join([
            "The path to the pytorch model checkpoint (*.pth) or Huggingface Model",
            "checkpoint directory (should contain the pytorch_model.bin file as a child).",
            "(ex1. ./ckpt/{YYYYmmDD-HHMMSS}/checkpoint-10590)",
            "(ex2: ./ckpt/{YYYYmmDD-HHMMSS}/{SOME_MODEL_PATH}.pth)",
        ]),
    )
    p.add_argument(
        "--test",
        required=True,
        help=" ".join([
            "Test *.tsv file name including columns named [id, text].",
            "(ex: ./data/test.tsv)",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--gpu_id", 
        type=int, 
        default=-1,
        help=" ".join([
            "The GPU number you want to use for inference.",
            "Only single GPU can be used, -1 means inference on CPU.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help=" ".join([
            "The batch size used for inference. In general, a value slightly",
            "larger than the batch size used for training is acceptable.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help=" ".join([
            "Number of beams for beam search. 1 means no beam search.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--length_penalty",
        type=float,
        default=0.8,
        help=" ".join([
            "Exponential penalty to the length. If it is greater than 1,",
            "long sentences are generated, and if it is less than 1, the",
            "generation proceeds toward shorter sentences.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help=" ".join([
            "The n-grams penalty makes sure that no n-gram appears twice",
            "by manually setting the probability of next words that could",
            "create an already seen n-gram to 0.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--inp_max_len", 
        type=int, 
        default=1024,
        help=" ".join([
            "Maximum length of tokenized input.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--tar_max_len", 
        type=int, 
        default=256,
        help=" ".join([
            "Maximum length of tokenized output (=summary). The minimum length",
            "is set to 1/4 of tar_max_len. If the maximum allowable length is",
            "too small, the sentence may not be completed and may break in the middle.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--var_len",
        action="store_true",
        help=" ".join([
            "Whether to allow the generation of variable-length summaries according to",
            "the average input length in batch units. If the value is true, the summaries",
            "have values from min(64, int(avg_len_per_batch * 0.05)) to",
            "min(256, int(avg_len_per_batch * 0.15)). Naturally, the input test data",
            "will be sorted by length in advance.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--sample_submission_path",
        type=str,
        default=os.path.join("data", "raw", "Test", "new_sample_submission.csv"),
        help=" ".join([
            "The path to the example answer file you want to reference.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--submission_path",
        type=str,
        default="submission",
        help=" ".join([
            "This is where the correct answers for submission, including summaries, are stored.",
            "Default=%(default)s",
        ]),
    )

    config = p.parse_args()

    return config


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(config)
    print_config(vars(config))

    ## Load data from huggingface checkpoints.
    if Path(config.model_fpath).is_dir():
        model = BartForConditionalGeneration.from_pretrained(config.model_fpath)

        with open(Path(config.model_fpath, "config.json"), "r", encoding="utf-8") as f:
            train_config = json.load(f)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(train_config["_name_or_path"])

    ## Load data from torch's file (*.pth).
    elif config.model_fpath.endswith(".pth"):
        saved_data = torch.load(
            config.model_fpath,
            map_location="cpu" if config.gpu_id < 0 else "cuda:%d" % config.gpu_id,
        )
        bart_best = saved_data["bart"]
        train_config = saved_data["config"]
        tokenizer = PreTrainedTokenizerFast.from_pretrained(train_config.pretrained_model_name)

        ## Load weights.
        model = BartForConditionalGeneration.from_pretrained(train_config.pretrained_model_name)
        model.load_state_dict(bart_best)

    else:
        raise AssertionError(f"We don't support such extension file: {config.model_fpath}")
        

    ## Get datasets and index to label map.
    ts_ds = get_datasets(tokenizer, fpath=Path(config.test), mode="test")
    ts_loader = torch.utils.data.DataLoader(
        ts_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=TextAbstractSummarizationCollator(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            inp_max_len=config.inp_max_len,
            tar_max_len=config.tar_max_len,
            mode="test",
        ),
    )

    ## We will not get lines by stdin, but file path.
    # lines = read_text()

    with torch.no_grad():
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        ## Don't forget turn-on evaluation mode.
        model.eval()

        outputs = []
        for mini_batch in tqdm(ts_loader, total=len(ts_loader)):
            id = mini_batch["id"]
            input_ids = mini_batch["input_ids"]
            attention_mask = mini_batch["attention_mask"]

            if config.var_len:
                ## Variable min, max length of target summaries.
                ## We know that summaries ~= text * 0.1.
                avg_len = int(input_ids.ne(tokenizer.pad_token_id).view(-1).sum() / input_ids.size(0))
                min_length = max(64,  int(avg_len * 0.05))
                max_length = min(256, int(avg_len * 0.15))
                ## And we don't need to set length penalty anymore.
                config.length_penalty = 1.0
            else:
                min_length = config.tar_max_len // 4
                max_length = config.tar_max_len

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            ## Generate ids of summaries.
            ##   - https://huggingface.co/transformers/v2.11.0/model_doc/bart.html#transformers.BartForConditionalGeneration.generate
            output = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_length=max_length,                  ## maximum summarization size
                min_length=min_length,                  ## minimum summarization size
                early_stopping=True,                    ## stop the beam search when at least 'num_beams' sentences are finished per batch
                num_beams=config.beam_size,             ## beam search size
                bos_token_id=tokenizer.bos_token_id,    ## <s> = 0
                eos_token_id=tokenizer.eos_token_id,    ## <\s> = 1
                pad_token_id=tokenizer.pad_token_id,    ## 3
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
            outputs.extend([{"id": id_, "output": output_} for id_, output_ in zip(id, output)])

    ## Sort and extract.
    outputs = sorted(
        outputs,
        key=itemgetter("id"),
        reverse=False,
    )
    outputs = [i["output"] for i in outputs]

    ## Save it.
    # if Path(config.model_fpath).is_dir():
    #     ## Find the *.pth file name.
    #     config.model_fpath = str(list(Path(config.model_fpath).parent.glob("*.pth"))[0])
    #     is_best = False
    # else:
    #     is_best = True

    save_to = Path(
        config.submission_path,                                 ## submission/
        ".".join([
            Path(config.model_fpath).parts[-2],                 ## datetime
            *Path(config.model_fpath).name.split(".")[:-1],     ## moel_fpath
            f"LP-{config.length_penalty:.1f}",                  ## length penalty
            f"{'' if config.var_len else 'no-'}var-len",
            f"{'best' if Path(config.model_fpath).is_dir() else 'latest'}",
            "csv",
        ]),
    )
    save_predictions(
        sample_submission_path=config.sample_submission_path, 
        predictions=outputs,
        save_to=save_to,
    )
    print(f"Submission save to: {save_to}")


if __name__ == "__main__":
    config = define_argparser()
    main(config)
