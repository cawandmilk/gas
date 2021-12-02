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


### Result

|Date|Model|Epoch|BS|LR|Warm-up|Opt|Tr-loss|Vl-loss|ROUGE-1|ROUGE-2|ROUGE-N|Note|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|2021-11-24|gogamza/kobart-summarization|5|32\*1\*1|5e-5|0|RAdam|3.09|2.09?|**0.4061**|**0.2485**|**0.3299**|train w/ dev. (head 1000)|
|2021-11-24|gogamza/kobart-base-v2|5|32\*1\*1|5e-5|0|RAdam|3.22|2.49?|0.0357|0.0038|0.0327|train w/ dev. (head 1000)|
|2021-11-25|gogamza/kobart-base-v2|5|48\*1\*1|5e-5|0|RAdam|1.73|2.14|0.1062|0.0069|0.0792||
|2021-11-27|gogamza/kobart-summarization|5|48\*2\*1|5e-5|0.2|AdamW||||||(only 법률)|
