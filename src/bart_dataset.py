import torch

import numpy as np

from pathlib import Path
from typing import List, Dict

from src.utils import read_tsv


# class TextAbstractSummarizationCollator():

#     def __init__(
#         self, 
#         tokenizer, 
#         inp_max_length: int, 
#         tar_max_length: int, 
#         with_text: bool = True,
#         is_train: bool = True,
#         ignore_index: int = -100,
#     ):
#         self.tokenizer = tokenizer
#         self.inp_max_length = inp_max_length
#         self.tar_max_length = tar_max_length
#         self.with_text = with_text
#         self.is_train = is_train
#         self.ignore_index = ignore_index


#     def _pad(self, sentences: List[str], token_id: int, max_length: int) -> torch.tensor:
#         ## We don't want to slice in this function, just add pad.
#         assert all([len(i) <= max_length for i in sentences])

#         ## max_lenght: max length of current batch.
#         ## target_max_length != max_length
#         max_length_per_batch = max([len(i) for i in sentences])

#         return torch.tensor([i + [self.tokenizer.pad_token_id] * (max_length_per_batch - len(i)) for i in sentences])


#     def _train_collator(self, samples: List[Dict[str, str]]) -> Dict[str, List[Union[int, float]]]:
#         ## Unpack.
#         texts = [s["text"] for s in samples]
#         summaries = [s["summary"] for s in samples]

#         ## Input (text).
#         encoding = self.tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=self.inp_max_length,
#         )
#         input_ids = encoding["input_ids"]
#         attention_mask = encoding["attention_mask"] ## to ignore weights of padding position

#         ## Target (summary).
#         decoding = [self.tokenizer.encode(
#             i,
#             padding=False,
#             add_special_tokens=False,
#         ) for i in summaries]

#         ## Add special tokens. (EOS)
#         decoder_input_ids = [[self.tokenizer.bos_token_id] + i for i in decoding]   ## not <BOS>, but <EOS>
#         decoder_input_ids = [i[:self.tar_max_length] for i in decoder_input_ids]

#         labels = [i[1:] + [self.tokenizer.eos_token_id] for i in decoder_input_ids]
        
#         ## Pad with 'pad_token_id(=3)' or 'ignore_index(=-100)'.
#         decoder_input_ids = self._pad(decoder_input_ids, self.tokenizer.pad_token_id, self.tar_max_length)
#         labels = self._pad(labels, self.ignore_index, self.tar_max_length)

#         ## Attention mask.
#         # decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).float()
#         decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).float()

#         ## Pack as pre-defined arguments:
#         ##   - https://huggingface.co/transformers/model_doc/bart.html
#         return_value = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "decoder_input_ids": decoder_input_ids,
#             "decoder_attention_mask": decoder_attention_mask,
#             "labels": labels,
#         }
#         if self.with_text:
#             return_value["text"] = texts

#         return return_value


#     def _test_collator(self, samples: List[Dict[str, str]]) -> Dict[str, List[int]]:
#         texts = [s["text"] for s in samples]

#         encoding = [self.tokenizer.encode(
#             i,
#             padding=False,
#             add_special_tokens=False,
#         ) for i in texts]

#         input_ids = [[self.tokenizer.eos_token_id] + i for i in encoding]   ## not <BOS>, but <EOS>
#         input_ids = [i[:self.tar_max_length] for i in input_ids]
        
#         input_ids = self._pad(input_ids, self.ignore_index, self.inp_max_length)

#         ## Pack as pre-defined arguments:
#         ##   - https://huggingface.co/gogamza/kobart-summarization
#         return_value = {
#             "input_ids": input_ids,
#             # "attention_mask": attention_mask,
#         }
#         if self.with_text:
#             return_value["text"] = texts

#         return return_value


#     def __call__(self, samples: dict) -> dict:
#         return self._train_collator(samples) if self.is_train else self._test_collator(samples)


class TextAbstractSummarizationDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        tokenizer,
        fpath: Path,
        inp_max_len: int = 1024,
        tar_max_len: int = 128,
        ignore_index: int = -100, ## pad to labels
        shuffle: bool = True,
        seed: int = 42,
        mode: str = "train",
    ):
        self.tokenizer = tokenizer
        self.df = read_tsv(fpath)
        self.len = self.df.shape[0]
        self.inp_max_len = inp_max_len
        self.tar_max_len = tar_max_len
        self.ignore_index = ignore_index

        ## Check nan.
        assert self.df.isnull().sum().sum() == 0

        ## Shuffle dataset.
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)

        ## Mode.
        assert mode in ["train", "test"]
        self.mode = mode
    

    def _pad(self, texts: List[int], max_len: int, token_id: int) -> List[int]:
        if len(texts) < max_len:
            texts = np.concatenate([texts, [token_id] * (max_len - len(texts))])
        else:
            texts = np.array(texts[:max_len])

        return texts

    
    def _get_item_for_train_mode(self, idx: int) -> Dict[str, torch.tensor]:
        ## Withdraw columns.
        instance = self.df.iloc[idx]

        ## Encoding inputs.
        input_ids = self.tokenizer.encode(instance["text"])
        input_ids = self._pad(input_ids, max_len=self.inp_max_len, token_id=self.tokenizer.pad_token_id)

        ## Decoding inputs.
        decoder_input_ids = self.tokenizer.encode(instance["summary"])
        decoder_input_ids = np.concatenate([[self.tokenizer.eos_token_id], decoder_input_ids])
        labels = np.concatenate([decoder_input_ids[1:], [self.tokenizer.eos_token_id]])

        ## Padding.
        decoder_input_ids = self._pad(decoder_input_ids, max_len=self.tar_max_len, token_id=self.tokenizer.pad_token_id)
        labels = self._pad(labels, max_len=self.tar_max_len, token_id=self.ignore_index) ## pad with -100

        ## Attention mask.
        attention_mask = (input_ids != 0).astype(float) ## original: compare with 'self.tokenizer.pad_token_id'.
        decoder_attention_mask = (decoder_input_ids != 0).astype(float)

        return {
            "input_ids":                torch.from_numpy(input_ids),
            "attention_mask":           torch.from_numpy(attention_mask),
            "decoder_input_ids":        torch.from_numpy(decoder_input_ids),
            "decoder_attention_mask":   torch.from_numpy(decoder_attention_mask),
            "labels":                   torch.from_numpy(labels),
        }

    def _get_item_for_test_mode(self, idx: int) -> Dict[str, torch.tensor]:
        ## Withdraw columns.
        instance = self.df.iloc[idx]

        ## Encoder inputs.
        input_ids = self.tokenizer.encode(instance["text"])
        input_ids = np.concatenate([[self.tokenizer.eos_token_id], input_ids])

        ## Padding.
        input_ids = self._pad(input_ids, max_len=self.inp_max_len, token_id=self.tokenizer.pad_token_id)

        return {
            "input_ids": torch.from_numpy(input_ids),
        }


    def __len__(self) -> int:
        return self.len
    

    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        return self._get_item_for_train_mode(idx) if self.mode == "train" else self._get_item_for_test_mode(idx)
