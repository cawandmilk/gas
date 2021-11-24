import torchtext

import os

PAD = 1
BOS = 2
EOS = 3


class DataLoader():

    def __init__(
        self,
        train_fn=None,
        valid_fn=None,
        exts=None,
        batch_size: int = 64,
        device: str = "cpu",
        max_vocab: int = 99_999_999,
        max_length: int = 255,
        fix_length: int = None,
        use_bos: bool = True,
        use_eos: bool = True,
        shuffle: bool = True,
        dsl: bool = False,
    ):
        super(DataLoader, self).__init__()

        self.src = torchtext.legacy.data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            fix_length=fix_length,
            init_token="<BOS>" if dsl else None,
            eos_token="<EOS>" if dsl else None,
        )

        self.tgt = torchtext.legacy.data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            fix_length=fix_length,
            init_token="<BOS>" if use_bos else None,
            eos_token="<EOS>" if use_eos else None,
        )

        if train_fn != None and valid_fn != None and exts != None:
            train = TranslationDataset(
                path=train_fn,
                exts=exts,
                fields=[("src", self.src), ("tgt", self.tgt)],
                max_length=max_length,
            )
            valid = TranslationDataset(
                path=valid_fn,
                exts=exts,
                fields=[("src", self.src), ("tgt", self.tgt)],
                max_length=max_length,
            )

            self.train_iter = torchtext.legacy.data.BucketIterator(
                train,
                batch_size=batch_size,
                device=f"cuda:{device}" if device >= 0 else "cpu",
                shuffle=shuffle,
                sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                sort_within_batch=True,
            )
            self.valid_iter = torchtext.legacy.data.BucketIterator(
                valid,
                batch_size=batch_size,
                device=f"cuda:{device}" if device >= 0 else "cpu",
                shuffle=False,
                sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                sort_within_batch=True,
            )

            self.src.build_vocab(train, max_size=max_vocab)
            self.tgt.build_vocab(train, max_size=max_vocab)

    
    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab



class TranslationDataset(torchtext.legacy.data.Dataset):
        
    def __init__(
        self, 
        path, 
        exts, 
        fields, 
        max_length: int = None, 
        **kwargs,
    ):
        if not isinstance(fields[0], (tuple, list)):
            fields = [("src", fields[0]), ("tgt", fields[1])]
        
        if not path.endswith("."):
            path += "."

        src_path, tgt_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path, encoding="utf-8") as src_file, open(tgt_path, encoding="utf-8") as tgt_file:
            for src_line, tgt_line in zip(src_file, tgt_file):
                src_line, tgt_line = src_line.strip(), tgt_line.strip()
                if max_length and max_length < max(len(src_line.split()), len(tgt_line.split())):
                    continue
                if src_line != "" and tgt_line != "":
                    examples += [torchtext.legacy.data.Example.fromlist([src_line, tgt_line], fields)]
        
        super().__init__(examples, fields, **kwargs)


    @staticmethod
    def sort_key(ex):
        return torchtext.legacy.data.interleave_keys(len(ex.src), len(ex.tgt))


if __name__ == "__main__":
    import sys
    loader = DataLoader(
        sys.argv[1],
        sys.argv[2],
        (sys.argv[3], sys.argv[4]),
        batch_size=128,
    )

    print(len(loader.src.vocab))
    print(len(loader.tgt.vocab))

    for batch_index, batch in enumerate(loader.train_iter):
        print(batch.src)
        print(batch.tgt)

        if batch_index > 1:
            break
