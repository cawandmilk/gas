import torch
import torch_optimizer

from transformers import Trainer
from transformers import TrainingArguments
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
# from transformers import get_linear_schedule_with_warmup

import argparse
import datetime
import pprint

from pathlib import Path

# from src.bart_dataset import TextAbstractSummarizationCollator
from src.bart_dataset import TextAbstractSummarizationDataset
# from src.utils import read_tsv


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
    # p.add_argument(
    #     "--lang",
    #     type=str,
    #     default="ifof",
    # )
    # p.add_argument(
    #     "--inp_suffix",
    #     type=str,
    #     default="if",
    # )
    # p.add_argument(
    #     "--tar_suffix",
    #     type=str,
    #     default="of",
    # )
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
    ## - hyunwoongko/kobart
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="gogamza/kobart-summarization",
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
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

    config = p.parse_args()

    return config


def get_datasets(config, tokenizer, fpath: Path, shuffle: bool = True, mode: str = "train"):
    return TextAbstractSummarizationDataset(
        tokenizer,
        fpath,
        inp_max_len=config.inp_max_len,
        tar_max_len=config.tar_max_len,
        shuffle=shuffle,
        mode=mode,
    )


# def get_optimizers(config, num_training_steps: int):
#     optimizer = torch_optimizer.RAdam(
#         lr=config.lr,
#         weight_decay=config.weight_decay,
#     )
#     lr_scheduler = get_linear_schedule_with_warmup(
#         optimizer=optimizer,
#         num_warmup_steps=int(num_training_steps * config.warmup_ratio),
#         num_training_steps=num_training_steps,
#     )


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Get pretrained tokenizer.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)
    ## Get datasets and index to label map.
    tr_ds = get_datasets(config, tokenizer, fpath=Path(config.train))
    vl_ds = get_datasets(config, tokenizer, fpath=Path(config.valid), shuffle=False)

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
    )

    ## Define trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        # data_collator=TextAbstractSummarizationCollator(
        #     tokenizer,
        #     inp_max_length=config.inp_max_length,
        #     tar_max_length=config.tar_max_length,
        #     with_text=False,
        # ),
        train_dataset=tr_ds,
        eval_dataset=vl_ds,
        # compute_metrics=compute_metrics,
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
