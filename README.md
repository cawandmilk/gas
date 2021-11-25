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


### Result

|Date|Model|Epoch|BS|LR|Warm-up|Opt|Tr-loss|Vl-loss|ROUGE-1|ROUGE-2|ROUGE-N|Note|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|2021-11-24|gogamza/kobart-summarization|5|32|5e-5|0|RAdam|3.09|2.09?|**0.4061**|**0.2485**|**0.3299**|train w/ dev. (head 1000)|
|2021-11-24|gogamza/kobart-base-v2|5|32|5e-5|0|RAdam|3.22|2.49?|0.0357|0.0038|0.0327|train w/ dev. (head 1000)|
|2021-11-25|gogamza/kobart-base-v2|5|48|5e-5|0|RAdam|1.73|2.14|0.1062|0.0069|0.0792||
|2021-11-25|gogamza/kobart-base-v2|5|48|5e-5|0.2|RAdam|1.73|2.14||||(current)|
