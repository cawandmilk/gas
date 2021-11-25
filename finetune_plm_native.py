import torch
import torch_optimizer

from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup

import argparse
import pprint

from pathlib import Path

from src.bart_trainer import BartTrainer as Trainer
from src.bart_dataset import TextAbstractSummarizationCollator, TextAbstractSummarizationDataset
from src.utils import read_text


def define_argparser(is_continue: bool = False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            "--load_fn",
            required=True,
            help="Model file name to continue.",
        )

    p.add_argument(
        "--model_fpath", 
        required=True,
    )
    p.add_argument(
        "--train",
        required=not is_continue,
        help="Training set file name except the extention. (ex: train.en --> train)",
    )
    p.add_argument(
        "--valid",
        required=not is_continue,
        help="Validation set file name except the extention. (ex: valid.en --> valid)",
    )
    p.add_argument(
        "--lang",
        type=str,
        default="ifof",
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
    p.add_argument(
        "--data",
        type=str,
        default="data",
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
        "--gpu_id", 
        type=int, 
        default=-1,
    )
    p.add_argument(
        "--verbose", 
        type=int, 
        default=2,
    )

    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.,
        help="Threshold for gradient clipping. Default=%(default)s",
    )
    p.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
    )
    p.add_argument(
        "--n_epochs", 
        type=int, 
        default=5,
    )
    p.add_argument(
        "--lr", 
        type=float, 
        default=5e-5,
    )
    p.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=.2,
    )
    p.add_argument(
        "--adam_epsilon",
        type=float, 
        default=1e-8,
    )
    ## If you want to use RAdam, I recommend to use LR=1e-4.
    ## Also, you can set warmup_ratio=0.
    p.add_argument(
        "--use_radam", 
        action="store_true",
    )
    p.add_argument(
        "--valid_ratio", 
        type=float, 
        default=.2,
    )

    config = p.parse_args()

    return config


def get_datasets(config, tokenizer):
    ## Get list of documents.
    tr_documents = read_text(config, fpath=Path(config.train))
    vl_documents = read_text(config, fpath=Path(config.valid))

    tr_texts, tr_summaries = tr_documents["texts"], tr_documents["summaries"]
    vl_texts, vl_summaries = vl_documents["texts"], vl_documents["summaries"]

    train_loader = torch.utils.data.DataLoader(
        TextAbstractSummarizationDataset(tr_texts, tr_summaries),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextAbstractSummarizationCollator(
            tokenizer, 
            inp_max_length=config.inp_max_length,
            tar_max_length=config.tar_max_length,
        ),
    )
    valid_loader = torch.utils.data.DataLoader(
        TextAbstractSummarizationDataset(vl_texts, vl_summaries),
        batch_size=config.batch_size,
        collate_fn=TextAbstractSummarizationCollator(
            tokenizer, 
            inp_max_length=config.inp_max_length,
            tar_max_length=config.tar_max_length,
        ),
    )

    return train_loader, valid_loader


def get_optimizer(model, config):
    if config.use_radam:
        optimizer = torch_optimizer.RAdam(model.parameters(), lr=config.lr)
    else:
        raise AssertionError()
        ## Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon,
        )

    return optimizer


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Get pretrained tokenizer.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)
    ## Get datasets and index to label map.
    train_laoder, valid_loader = get_datasets(config, tokenizer)

    print(
        f"|train| = {len(train_laoder)}",
        f"|valid| = {len(valid_loader)}",
    )

    n_total_iterations = len(train_laoder) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        f"# total iters = {n_total_iterations}",
        f"# warmup iters = {n_warmup_steps}",
    )

    ## Get pretrained model with specified softmax layer.
    model = BartForConditionalGeneration.from_pretrained(config.pretrained_model_name)
    # if torch.cuda.device_count() > 1:
    #     print(f"{torch.cuda.device_count()} gpus available.")
    #     model = torch.nn.DataParallel(model)

    optimizer = get_optimizer(model, config)

    ## We will not use our own loss.
    crit = None
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations,
    )

    if config.gpu_id >= 0:
        # model.device("cuda")
        model.cuda(config.gpu_id)

    ## Start train.
    trainer = Trainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_laoder,
        valid_loader,
    )
    

if __name__ == "__main__":
    config = define_argparser()
    main(config)
