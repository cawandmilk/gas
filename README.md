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

Preprocess format:
  - https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/README.md
  
```
{"text": "I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder", "summary": "I'm sitting in a room where I'm waiting for something to happen"}
{"text": "I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.", "summary": "I'm a gardener and I'm a big fan of flowers."}
{"text": "Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share", "summary": "It's that time of year again."}
```

Bart modeling:
  - https://huggingface.co/transformers/_modules/transformers/modeling_bart.html