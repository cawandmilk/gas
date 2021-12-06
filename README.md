### Dacon-gas

Forked from:
  - https://github.com/kh-kim/simple-ntc

Keyword for useful method:
  - bart for seq2seq
  - from transformers import Trainer, TrainingArguments
  - logging_dir arguments -> tensorboard

Sites:
  - https://huggingface.co/transformers/model_summary.html?highlight=pegasus
  - https://velog.io/@jaehyeong/Paper-Review-PEGASUSPre-training-with-Extracted-Gap-sentences-for-Abstractive-Summarization
  - https://huggingface.co/gogamza/kobart-summarization
  - https://huggingface.co/upskyy/kobart-summarization-v3

Bart modeling:
  - https://huggingface.co/transformers/_modules/transformers/modeling_bart.html

Input ids: 512 -> 768? 1024?

### Dataset

|split|doc_type|#|uniq(media_name)|
|:-:|:-:|:-:|:-:|
|train|법률|27,033|0|
|train|사설잡지|63,067|64|
|train|신문기사|271,093|42|
|valid|법률|3,004|0|
|valid|사설잡지|7,008|40|
|valid|신문기사|30,122|4|


### Result (Private LB Scores)

* Fixed Params:
  - Model: gogamza/kobart-base-v1
  - Batch size: 256 (=16x2x8)
    - Per replica batch size = 16 (Tesla V100 32GB VRAM)
    - \# GPUs = 2
    - Gradient accumulate steps = 8


|Date|Cleaning|Epoch|Tr-loss|Vl-loss|LP|ROUGE-1|ROUGE-2|ROUGE-N|Note|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|2021-12-06|X|10(7)|1.0466|1.2471|1.2|0.3671|0.1801|0.2778||
|2021-12-06|X|10(7)|1.0466|1.2471|1.0|0.3672|0.1798|0.2785||
|2021-12-06|X|10(7)|1.0466|1.2471|0.8|0.3678|0.1803|0.2800|**final submission**|

