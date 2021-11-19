import torch


class TextClassificationCollator():

    def __init__(
        self, 
        tokenizer, 
        max_length: int, 
        with_text: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples: dict) -> dict:
        texts = [s["text"] for s in samples]
        labels = [s["label"] for s in samples]

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        return_value = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],   ## to ignore weights of padding position
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.with_text:
            return_value["text"] = texts

        return return_value


class TextClassificationDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        texts, 
        labels,
    ):
        self.texts = texts
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, item: int) -> dict:
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            "text": text,
            "label": label,
        }
