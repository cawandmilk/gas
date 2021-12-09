import torch

import tqdm

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Dict

from src.utils import read_tsv


class TextAbstractSummarizationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        tokenizer,
        fpath: Path,
        mode: str = "train",
    ):
        super(TextAbstractSummarizationDataset, self).__init__()

        self.df = read_tsv(fpath)
        # self.tok = tokenizer -> don't keep
        
        ## Mode.
        assert mode in ["train", "test"]
        self.mode = mode

        ## Apply tokenize first to speed up in training phase and make code more simply.
        tqdm.tqdm.pandas(desc="Tokenizing input texts")
        self.df.loc[:, "text_tok"] = self.df.loc[:, "text"].progress_apply(lambda x: tokenizer.encode(x))
        self.df.loc[:, "text_tok_len"] = self.df.loc[:, "text_tok"].apply(lambda x: len(x))
        if self.mode == "train":
            tqdm.tqdm.pandas(desc="Tokenizing target summaries")
            self.df.loc[:, "summary_tok"] = self.df.loc[:, "summary"].progress_apply(lambda x: tokenizer.encode(x))
            self.df.loc[:, "summary_tok_len"] = self.df.loc[:, "summary_tok"].apply(lambda x: len(x))

        ## Sort by tokenized length with tqdm progress bar.
        ## 
        ## By sorting sequentially, starting with the longest sentence, 
        ## we can determine the maximum VRAM size the model is using for
        ## training. That is, if OOM does not occur for the maximum VRAM
        ## size at the beginning of training, it is guaranteed that OOM
        ## does not occur during training.
        self.df.sort_values(by=["text_tok_len"], axis=0, ascending=False, inplace=True)

    
    def __len__(self) -> int:
        return self.df.shape[0]


    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        instance = self.df.iloc[idx]

        return_value = {
            "id": instance["id"], ## for sorting in inference mode
            "text": instance["text_tok"],
        }
        if self.mode == "train":
            return_value["summary"] = instance["summary_tok"]
        
        return return_value


class TextAbstractSummarizationCollator():

    def __init__(
        self,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        inp_max_len: int = 1024,
        tar_max_len: int = 256,
        ignore_index: int = -100,
        mode: str = "train",
    ):
        super(TextAbstractSummarizationCollator, self).__init__()

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.inp_max_len = inp_max_len
        self.tar_max_len = tar_max_len
        self.ignore_index = ignore_index

        ## Mode.
        assert mode in ["train", "test"]
        self.mode = mode


    def _pad(self, sentences: List[List[int]], token_id: int) -> np.ndarray:
        ## We will pad as max length per batch, not "inp_max_len(=1024, etc)".
        max_length_per_batch = max([len(i) for i in sentences])

        ## Stack as dimension 0 (batch dimension).
        ## "token_id" can be "tokenizer.pad_token_id(=3)" or "ignore_index(=-100)"
        return np.stack([i + [token_id] * (max_length_per_batch - len(i)) for i in sentences], axis=0)


    def _train_collator(self, samples: List[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        ## Unpack.

        ## If input max length > 1024, you can see below error:
        ##   1) Assertion `srcIndex < srcSelectDimSize` failed
        ##   2) Device-side assert triggered
        tokenized_texts     = [s["text"][:self.inp_max_len]        for s in samples]
        tokenized_summaries = [s["summary"][:self.tar_max_len - 1] for s in samples] ## <bos> or <eos> token index

        ## Inputs for encoder.
        input_ids = self._pad(tokenized_texts, token_id=self.pad_token_id)
        attention_mask = (input_ids != self.pad_token_id).astype(float)

        ## Inputs for decoder (generator).
        decoder_input_ids = [[self.bos_token_id] + i for i in tokenized_summaries]      ## bos? eos?
        decoder_input_ids = self._pad(decoder_input_ids, token_id=self.pad_token_id)    ## eos
        decoder_attention_mask = (decoder_input_ids != self.pad_token_id).astype(float)

        ## Answer.
        labels = [i + [self.eos_token_id] for i in tokenized_summaries]
        labels = self._pad(labels, token_id=self.ignore_index) ## why no "padding_id" ???

        ## We ensure that generator's inputs' and outputs' shapes are equal.
        assert decoder_input_ids.shape == labels.shape
        
        ## Pack as pre-defined arguments:
        ## See: https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration
        return {
            "input_ids":                torch.from_numpy(input_ids),
            "attention_mask":           torch.from_numpy(attention_mask),
            "decoder_input_ids":        torch.from_numpy(decoder_input_ids),
            "decoder_attention_mask":   torch.from_numpy(decoder_attention_mask),
            "labels":                   torch.from_numpy(labels),
        }


    def _test_collator(self, samples: List[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        ## Unpack.
        ids             = [s["id"]                      for s in samples]
        tokenized_texts = [s["text"][:self.inp_max_len] for s in samples] ## no <bos> token included

        ## Inputs for encoder.
        input_ids = self._pad(tokenized_texts, token_id=self.pad_token_id)
        attention_mask = (input_ids != self.pad_token_id).astype(float)

        ## Pack as pre-defined arguments:
        ## See: https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration
        return {
            "input_ids":        torch.from_numpy(input_ids),
            "attention_mask":   torch.from_numpy(attention_mask),
            ## Additional information to make answer.
            "id":               ids,
        }


    def __call__(self, samples: List[Dict[str, List[int]]]) -> Dict[str, List[int]]:
        return self._train_collator(samples) if self.mode == "train" else self._test_collator(samples)
