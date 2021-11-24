import torch

import numpy as np


class TextAbstractSummarizationCollator():

    def __init__(
        self, 
        tokenizer, 
        inp_max_length: int, 
        tar_max_length: int, 
        with_text: bool = True,
    ):
        self.tokenizer = tokenizer
        self.inp_max_length = inp_max_length
        self.tar_max_length = tar_max_length
        self.with_text = with_text


    def _pad(self, sentences: list) -> list:
        ## We don't want to slice in this function, just add pad.
        assert all([len(i) <= self.tar_max_length for i in sentences])

        ## max_lenght: max length of current batch.
        ## target_max_length != max_length
        max_length = max([len(i) for i in sentences])

        return torch.tensor([i + [self.tokenizer.pad_token_id] * (max_length - len(i)) for i in sentences])


    def __call__(self, samples: dict) -> dict:
        texts = [s["text"] for s in samples]

        ## Input (text).
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.inp_max_length,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"] ## to ignore weights of padding position

        ## Pack as pre-defined arguments:
        ##   - https://huggingface.co/transformers/model_doc/bart.html
        return_value = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if samples[0].get("summary") != None:
            summaries = [s["summary"] for s in samples]
            ## Target (summary).
            decoding = [self.tokenizer.encode(i, padding=False, add_special_tokens=False) for i in summaries]

            ## We will generate 'decoder_input_ids' and 'labels' as below:
            ##   1) Insert <BOS> token in front of each batched sentences.
            ##   2) Slice 'decoder_input_ids' as 'tar_max_length'.
            ##   3) Remove <BOS> token and append <EOS> token of each batched sentences.
            ##   4) Add paddings either in 'decoder_input_ids' and 'labels'.
            decoder_input_ids = [[self.tokenizer.bos_token_id] + i for i in decoding]
            decoder_input_ids = [i[:self.tar_max_length] for i in decoder_input_ids]
            labels = [i[1:] + [self.tokenizer.eos_token_id] for i in decoder_input_ids]

            decoder_input_ids = self._pad(decoder_input_ids)
            labels = self._pad(labels)
            ## |decoder_input_ids| = (batch_size, variable_max_length)
            ## |labels|            = (batch_size, variable_max_length)

            ## Make attention mask.
            decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).float()
            ## |decoder_attention_mask| = (batch_size, tar_max_length)

            return_value["decoder_input_ids"] = decoder_input_ids
            return_value["decoder_attention_mask"] = decoder_attention_mask
            return_value["labels"] = labels

        if self.with_text:
            return_value["text"] = texts

        return return_value


class TextAbstractSummarizationDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        texts, 
        summaries: list = None,
    ):
        self.texts = texts
        self.summaries = summaries
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, item: int) -> dict:
        text = str(self.texts[item])
        return_value = {"text": text}

        if self.summaries != None:
            summary = str(self.summaries[item])
            return_value["summary"] = summary

        return return_value
