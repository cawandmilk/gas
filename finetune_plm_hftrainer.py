import torch

from transformers import Seq2SeqTrainingArguments, Trainer
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

import argparse
import datetime
import os
import pprint

from pathlib import Path

from src.bart_dataset import TextAbstractSummarizationDataset, TextAbstractSummarizationCollator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_fpath", 
        required=True,
        help=" ".join([
            "The name of the trained model being saved except the extension.",
            "(ex: hft.gogamza.kobart-base-v1.bs-16*2*8.lr-5e-5.wd-1e-2.warmup-2.adamw)",
        ]),
    )
    p.add_argument(
        "--train",
        required=True,
        help=" ".join([
            "Training *.tsv file name including columns named [id, text, summary].",
            "(ex: ./data/train.tsv)",
        ]),
    )
    p.add_argument(
        "--valid",
        required=True,
        help=" ".join([
            "Validate *.tsv file name including columns named [id, text, summary].",
            "(ex: ./data/valid.tsv)",
        ]),
    )
    p.add_argument(
        "--logs",
        type=str,
        default="logs",
        help=" ".join([
            "Top-level folder where logs for Tensorboard visualizations are stored.",
            "Logs are automatically written inside the sub-folder 'YYYYmmDD-HHMMSS'.",
            "(ex: ./logs/20211205-164445/{SOME_LOGS...}",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="ckpt",
        help=" ".join([
            "The top-level folder path where checkpoints are stored.",
            "In addition to the model automatically saved by Huggingface trainer,",
            "the checkpoint with the lowest(=best) validation loss will be saved with",
            "the *.pth extension by adding the current time in front of the model name",
            "specified in 'model_fpath' argument.",
            "Default=%(default)s",
        ]),
    )

    ## Recommended model list:
    ## - gogamza/kobart-base-v1
    ## - gogamza/kobart-base-v2
    ## - gogamza/kobart-summarization
    ## - ainize/kobart-news
    ## - hyunwoongko/kobart
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="gogamza/kobart-summarization",
        help=" ".join([
            "Calls from models published to Huggingface Hub.",
            "See: https://huggingface.co/models. ",
            "Default=%(default)s",
        ]),
    )

    p.add_argument(
        "--per_replica_batch_size", 
        type=int, 
        default=48,
        help=" ".join([
            "Batch size allocated per GPU.",
            "If only 1 GPU is available, it is the same value as 'global_batch_size'.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--n_epochs", 
        type=int, 
        default=5,
        help=" ".join([
            "The number of iterations of training & validation for the entire dataset.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=.2,
        help=" ".join([
            "The ratio of warm-up iterations that gradulally increase",
            "compared to the total number of iterations.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--lr", 
        type=float, 
        default=5e-5,
        help=" ".join([
            "The learning rate.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-2,
        help=" ".join([
            "Weight decay applied to the AdamW optimizer.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=" ".join([
            "Number of updates steps to accumulate the gradients for,",
            "before performing a backward/update pass.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--inp_max_len", 
        type=int, 
        default=1024,
        help=" ".join([
            "A value for slicing the input data.",
            "It is important to note that the upper limit is determined",
            "by the embedding value of the model you want to use.",
            "Default=%(default)s",
        ]),
    )
    p.add_argument(
        "--tar_max_len", 
        type=int, 
        default=256,
        help=" ".join([
            "A value for slicing the output data. It is used for model inference.",
            "if the value is too small, the summary may be truncated before completion.",
            "Default=%(default)s",
        ]),
    )

    config = p.parse_args()

    return config


def get_datasets(tokenizer, fpath: Path, mode: str = "train"):
    return TextAbstractSummarizationDataset(
        tokenizer,
        fpath,
        mode=mode,
    )


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Get pretrained tokenizer.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)
    ## Get datasets and index to label map.
    tr_ds = get_datasets(tokenizer, fpath=Path(config.train))
    vl_ds = get_datasets(tokenizer, fpath=Path(config.valid))

    ## Get new tokenizer.
    del tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)

    ## Get pretrained model with specified softmax layer.
    model = BartForConditionalGeneration.from_pretrained(config.pretrained_model_name)

    ## Path arguments.
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.ckpt, nowtime)
    logging_dir = Path(config.logs, nowtime, "run")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=config.per_replica_batch_size,
        per_device_eval_batch_size=config.per_replica_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.n_epochs,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        # save_steps=1000,
        fp16=True,
        dataloader_num_workers=4,
        disable_tqdm=False,
        load_best_model_at_end=True,
        ## As below, only Seq2SeqTrainingArguments' arguments.
        sortish_sampler=True,
        # predict_with_generate=True,
        # generation_max_length=config.tar_max_len,   ## 512
        # generation_num_beams=config.beam_size,      ## 5
    )

    ## Define trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=TextAbstractSummarizationCollator(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            inp_max_len=config.inp_max_len,
            tar_max_len=config.tar_max_len,
        ),
        train_dataset=tr_ds,
        eval_dataset=vl_ds,
    )

    ## Train.
    trainer.train()

    ## Save the best model.
    model_dir = list(Path(config.ckpt).glob("*"))[-1]

    torch.save({
        "bart": trainer.model.state_dict(),
        "config": config,
        "tokenizer": tokenizer,
    }, Path(model_dir, ".".join([config.model_fpath, "pth"])))


if __name__ == "__main__":
    config = define_argparser()
    main(config)
