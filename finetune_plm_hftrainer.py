import torch

from transformers import Trainer
from transformers import TrainingArguments

from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

import argparse
import pprint

from pathlib import Path
from sklearn.metrics import accuracy_score

from src.bert_dataset import TextAbstractSummarizationCollator
from src.bert_dataset import TextAbstractSummarizationDataset
from src.utils import read_text


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_fpath", 
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

    ## Recommended model list:
    ## - ...
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="ainize/kobart-news",
    )
    p.add_argument(
        "--use_albert", 
        action="store_true",
    )
    p.add_argument(
        "--valid_ratio", 
        type=float, 
        default=.2,
    )
    p.add_argument(
        "--batch_size_per_device", 
        type=int, 
        default=32,
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
    documents = read_text(config)

    tr_texts, tr_summaries = documents["tr_texts"], documents["tr_summaries"]
    vl_texts, vl_summaries = documents["vl_texts"], documents["vl_summaries"]

    train_dataset = TextAbstractSummarizationDataset(tr_texts, tr_summaries)
    valid_dataset = TextAbstractSummarizationDataset(vl_texts, vl_summaries)

    return train_dataset, valid_dataset

# def compute_loss(output_size, pad_index):
#     ## Default weight for loss equals to 1, but we don't need to get loss for PAD token.
#     ## Thus, set a weight for PAD to zero.
#     loss_weight = torch.ones(output_size)
#     loss_weight[pad_index] = 0.
#     ## Instead of using Cross-Entropy loss,
#     ## we can use Negative Log-Likelihood(NLL) loss with log-probability.
#     crit = torch.nn.NLLLoss(
#         weight=loss_weight,
#         reduction="sum",
#     )

# def compute_loss(model, inputs):
#     """
#     How the loss is computed by Trainer. By default, all models 
#     return the loss in the first element. Subclass and override 
#     for custom behavior.

#       - inputs: mini-batched dictionary with inputs, outputs
#         -> For pretrained definition, we use 'inputs' keyword.

#     """


#     labels = None

#     outputs = model(**inputs)

#     if labels is not None:
#         loss = label_smoother(outputs, labels)
#     else:
#         # We don't use .loss here since the model may return tuples instead of ModelOutput.
#         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

#     return (loss, outputs) if return_outputs else loss


def main(config):
    def print_config(config):
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))
    print_config(config)

    ## Get pretrained tokenizer.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)
    ## Get datasets and index to label map.
    train_dataset, valid_dataset = get_datasets(config)

    print(
        f"|train| = {len(train_dataset)}",
        f"|valid| = {len(valid_dataset)}",
    )

    ## Pytorch style: batch_size_per_device
    ## Tensorflow style: per_replica_batch_size
    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        f"# total iters = {n_total_iterations}",
        f"# warmup iters = {n_warmup_steps}",
    )

    ## Get pretrained model with specified softmax layer.
    model = BartForConditionalGeneration.from_pretrained(
        config.pretrained_model_name,
    )

    training_args = TrainingArguments(
        output_dir="./.checkpoints",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=n_total_iterations // 100,
        save_steps=n_total_iterations // config.n_epochs,
        load_best_model_at_end=True,
    )

    def compute_loss(**kwargs):
        print(kwargs.keys())
        return None

    # def compute_metrics(pred):
    #     labels = pred.label_ids
    #     preds = pred.predictions.argmax(-1)

    #     return {
    #         "accuracy": accuracy_score(labels, preds)
    #     }

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
        # compute_loss=compute_loss,
        # compute_metrics=compute_metrics,
    )

    trainer.train()

    torch.save({
        "bart": trainer.model.state_dict(),
        "config": config,
        "vocab": None,
        "tokenizer": tokenizer,
    }, config.model_fpath)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
