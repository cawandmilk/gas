# 데이콘 경진대회 복원 설명서


## File Download

훈련&검증&평가용 데이터 세트는 모두 저작권이 존재하여 개인적으로 배포가 불가능한 바, 직접 다운로드 및 압축 해제를 진행하시어 아래와 같이 초기 디렉토리를 구성해주시기 바랍니다.

```bash
$ tree ./data
./data
└── [  52]  raw
    ├── [  94]  Test
    │   ├── [129K]  new_sample_submission.csv
    │   ├── [ 21M]  new_test_.json
    │   ├── [ 34M]  new_test.jsonl
    │   └── [  53]  old
    │       ├── [ 81K]  sample_submission.csv
    │       └── [ 19M]  test.jsonl
    ├── [ 231]  Training
    │   ├── [ 90M]  법률_train_original.json
    │   ├── [346M]  사설잡지_train_original.json
    │   ├── [1.2G]  신문기사_train_original.json
    │   ├── [ 18M]  법률_train_original.zip
    │   ├── [ 83M]  사설잡지_train_original.zip
    │   └── [296M]  신문기사_train_original.zip
    └── [ 231]  Validation
        ├── [8.5M]  법률_valid_original.json
        ├── [ 35M]  사설잡지_valid_original.json
        ├── [140M]  신문기사_valid_original.json
        ├── [1.6M]  법률_valid_original.zip
        ├── [7.9M]  사설잡지_valid_original.zip
        └── [ 34M]  신문기사_valid_original.zip

5 directories, 17 files
```

## Environments

실험 결과에 중대한 영향을 미칠 수 있는 주요 하드웨어 제원 및 개발 환경 버전은 아래와 같습니다.

- OS: Ubuntu 16.04.7 LTS
- RAM: 177GB
- GPU: NVIDIA Tesla V100 (32GB) * 2장
  - CUDA Version: 11.3
- Python: 3.8
- DL Framework:
  - PyTorch 1.10.0+cu113


`Python 3.8`이 설치된 환경에서 다음과 같이 가상환경을 구성하고, 필요 라이브러리를 설치합니다.

```bash
$ python -m venv venv
$ source ./venv/bin/activate
(venv) $ pip install --upgrade pip
(venv) $ pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
(venv) $ pip install -r requirements.txt
```


## Preprocess

원본 파일에서 필요한 정보만을 추출하여 `*.tsv` (tab separated value) 파일을 만듭니다. 일부 결측치 및 이상치 처리 과정이 포함되며, 텍스트 클리닝 과정은 구현이 되어있지만 그 성능이 좋지 않아 적용하지 않았습니다.

```bash
(venv) $ python preprocess.py \
    --raw_train ./data/raw/Training \
    --raw_valid ./data/raw/Validation \
    --raw_test ./data/raw/Test
```

전처리된 파일들은 다음과 같은 결과를 보여야 합니다.

```bash
$ wc -l ./data/*.tsv
     6597 ./data/test.tsv
   271089 ./data/train.tsv
    30123 ./data/valid.tsv
   307809 total
```


## Training

아래와 같은 명령어로 훈련이 가능하며, 본 개발 환경과 동일한 하드웨어 제원을 갖추었다면 약 8시간 이내로 훈련이 종료됩니다.

```bash
(venv) $ python finetune_plm_hftrainer.py \
    --train ./data/train.tsv \
    --valid ./data/valid.tsv \
    --pretrained_model_name gogamza/kobart-base-v1 \
    --per_replica_batch_size 16 \
    --lr 5e-5 \
    --weight_decay 1e-2 \
    --gradient_accumulation_steps 8 \
    --n_epochs 10 \
    --model_fpath model
```


## Inference (Entrypoint)

훈련이 완료된 `*.pth` 형식의 적절한 모델 가중치 파일이 존재한다면 아래와 같은 명령어로 추론을 진행할 수 있으며, 추론 소요 시간은 약 50분 입니다.

```bash
(venv) $ python summarize.py \
    --test ./data/test.tsv \
    --model_fpath ./ckpt/20211207-192805/model.pth \
    --gpu_id 0 \
    --length_penalty 0.8 \
    --batch_size 64
```

추론이 완료된 submission 파일은 `./submission`에 위치하게 됩니다.
