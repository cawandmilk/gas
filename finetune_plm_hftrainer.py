import torch

from transformers import Trainer
from transformers import TrainingArguments
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

import argparse
import datetime
import pprint

from pathlib import Path

from src.bart_dataset import TextAbstractSummarizationCollator
from src.bart_dataset import TextAbstractSummarizationDataset
from src.utils import read_text


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_fpath", 
        required=True,
    )
    p.add_argument(
        "--train",
        required=True,
        help="Training set file name except the extention. (ex: train.en --> train)",
    )
    p.add_argument(
        "--valid",
        required=True,
        help="Validation set file name except the extention. (ex: valid.en --> valid)",
    )
    p.add_argument(
        "--lang",
        type=str,
        default="ifof",
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
    p.add_argument(
        "--data",
        type=str,
        default="data",
    )
    p.add_argument(
        "--logs",
        type=str,
        default="logs",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="ckpt",
    )

    ## Recommended model list:
    ## - gogamza/kobart-base-v1
    ## - gogamza/kobart-base-v2
    ## - gogamza/kobart-summarization
    ## - ainize/kobart-news
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="gogamza/kobart-base-v2",
    )

    p.add_argument(
        "--per_replica_batch_size", 
        type=int, 
        default=48,
    )
    p.add_argument(
        "--n_epochs", 
        type=int, 
        default=5,
    )
    p.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=.2,
    )
    p.add_argument(
        "--lr", 
        type=float, 
        default=5e-5,
    )
    p.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-2,
    )
    p.add_argument(
        "--inp_max_length", 
        type=int, 
        default=512,
    )
    p.add_argument(
        "--tar_max_length", 
        type=int, 
        default=128,
    )

    config = p.parse_args()

    return config


def get_datasets(config):
    ## Get list of documents.
    tr_documents = read_text(config, fpath=Path(config.train))
    vl_documents = read_text(config, fpath=Path(config.valid))

    tr_texts, tr_summaries = tr_documents["texts"], tr_documents["summaries"]
    vl_texts, vl_summaries = vl_documents["texts"], vl_documents["summaries"]

    train_dataset = TextAbstractSummarizationDataset(tr_texts, tr_summaries)
    valid_dataset = TextAbstractSummarizationDataset(vl_texts, vl_summaries)

    return train_dataset, valid_dataset


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Get pretrained tokenizer.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)
    ## Get datasets and index to label map.
    train_dataset, valid_dataset = get_datasets(config)

    ## Get pretrained model with specified softmax layer.
    model = BartForConditionalGeneration.from_pretrained(config.pretrained_model_name)

    ## Path arguments.
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.ckpt, nowtime)
    logging_dir = Path(config.logs, nowtime, "run")

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=config.per_replica_batch_size,
        per_device_eval_batch_size=config.per_replica_batch_size,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.n_epochs,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        # save_steps=1000,
        fp16=True,
        dataloader_num_workers=4,
        disable_tqdm=True,
        load_best_model_at_end=True,
    )

    ## Define trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=TextAbstractSummarizationCollator(
            tokenizer,
            inp_max_length=config.inp_max_length,
            tar_max_length=config.tar_max_length,
            with_text=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # compute_metrics=compute_metrics,
        callbacks=[

        ]
    )

    ## Train.
    trainer.train()

    torch.save({
        "bart": trainer.model.state_dict(),
        "config": config,
        "tokenizer": tokenizer,
    }, config.model_fpath)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
