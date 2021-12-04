import torch

import numpy as np


class TextAbstractSummarizationCollator():

    def __init__(
        self, 
        tokenizer, 
        inp_max_length: int, 
        tar_max_length: int, 
        with_text: bool = True,
        is_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.inp_max_length = inp_max_length
        self.tar_max_length = tar_max_length
        self.with_text = with_text
        self.is_train = is_train


    def _pad(self, sentences: list, max_length: int) -> list:
        ## We don't want to slice in this function, just add pad.
        assert all([len(i) <= max_length for i in sentences])

        ## max_lenght: max length of current batch.
        ## target_max_length != max_length
        max_length_per_batch = max([len(i) for i in sentences])

        return torch.tensor([i + [self.tokenizer.pad_token_id] * (max_length_per_batch - len(i)) for i in sentences])


    def _train_collator(self, samples: dict) -> dict:
        texts = [s["text"] for s in samples]
        summaries = [s["summary"] for s in samples]

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

        ## Target (summary).
        decoding = [self.tokenizer.encode(i, padding=False, add_special_tokens=False) for i in summaries]

        ## We will generate 'decoder_input_ids' and 'labels' as below:
        ##   1) Insert <BOS> token in front of each batched sentences.
        ##   2) Slice 'decoder_input_ids' as 'tar_max_length'.
        ##   3) Remove <BOS> token and append <EOS> token of each batched sentences.
        ##   4) Add paddings either in 'decoder_input_ids' and 'labels'.
        decoder_input_ids = [[self.tokenizer.eos_token_id] + i for i in decoding] ## not <BOS> !!! (thus bidirectional)
        decoder_input_ids = [i[:self.tar_max_length] for i in decoder_input_ids]
        labels = [i[1:] + [self.tokenizer.eos_token_id] for i in decoder_input_ids]

        decoder_input_ids = self._pad(decoder_input_ids, max_length=self.tar_max_length)
        labels = self._pad(labels, max_length=self.tar_max_length)
        ## |decoder_input_ids| = (batch_size, variable_max_length)
        ## |labels|            = (batch_size, variable_max_length)

        ## Make attention mask.
        decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).float()
        ## |decoder_attention_mask| = (batch_size, tar_max_length)

        ## Pack as pre-defined arguments:
        ##   - https://huggingface.co/transformers/model_doc/bart.html
        return_value = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
        if self.with_text:
            return_value["text"] = texts

        return return_value


    def _test_collator(self, samples: dict) -> dict:
        texts = [s["text"] for s in samples]

        input_ids = []
        for text in texts:
            text = self.tokenizer.encode(text)
            # text = [self.tokenizer.eos_token_id] + text[:self.inp_max_length - 2] + [self.tokenizer.eos_token_id]
            text = text[:self.inp_max_length]

            input_ids.append(text)
        
        input_ids = self._pad(input_ids, max_length=self.inp_max_length)

        ## Pack as pre-defined arguments:
        ##   - https://huggingface.co/gogamza/kobart-summarization
        return_value = {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
        }
        if self.with_text:
            return_value["text"] = texts

        return return_value


    def __call__(self, samples: dict) -> dict:
        return self._train_collator(samples) if self.is_train else self._test_collator(samples)


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
