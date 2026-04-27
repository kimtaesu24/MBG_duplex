# Personaplex MBG Finetuning

Moshi의 `moshi-finetune` 코드를 기반으로 [NVIDIA Personaplex 7B](https://huggingface.co/nvidia/personaplex-7b-v1) 모델을 파인튜닝합니다.

## 설치

```bash
cd /home2/s20235100/Conversational-AI/personaplex_MBG/moshi-finetune

# finetune 패키지 설치
pip install -e .

# personaplex moshi 패키지 설치 (아직 안 했다면)
cd /home2/s20235100/Conversational-AI/personaplex_MBG/moshi
pip install -e .
```

## 데이터 준비

### JSONL 형식

각 라인은 다음 형식의 JSON 객체입니다:

```json
{"path": "/path/to/audio.wav", "duration": 30.5}
```

### JSON alignment 파일

오디오 파일과 같은 디렉토리에 동일한 이름의 `.json` 파일이 필요합니다:

```json
{
  "alignments": [
    ["hello", [0.0, 0.5], "SPEAKER_MAIN"],
    ["world", [0.6, 1.2], "SPEAKER_MAIN"]
  ]
}
```

alignment 파일 생성 도구: `annotate.py` (Moshi/moshi-finetune 참고)

## 설정

`config/example.yaml`을 복사해서 수정합니다:

```bash
cp config/example.yaml config/my_experiment.yaml
```

주요 설정:

```yaml
data:
  train_data: /path/to/dataset.jsonl

run_dir: /path/to/output_dir

moshi_paths:
  # HuggingFace에서 자동 다운로드 (기본값):
  hf_repo_id: nvidia/personaplex-7b-v1
  # 또는 로컬 경로 직접 지정:
  # moshi_path:     /path/to/model.safetensors
  # mimi_path:      /path/to/tokenizer-*.safetensors
  # tokenizer_path: /path/to/tokenizer_spm_32k_3.model

full_finetuning: true  # Personaplex는 LoRA 미지원 → 항상 true
param_dtype: bfloat16
```

## 학습 실행

```bash
cd /home2/s20235100/Conversational-AI/personaplex_MBG/moshi-finetune

# PYTHONPATH에 personaplex moshi 패키지를 추가합니다
export PYTHONPATH=/home2/s20235100/Conversational-AI/personaplex_MBG/moshi:$PYTHONPATH

# 단일 GPU
torchrun --nproc_per_node=1 train.py config/my_experiment.yaml

# 다중 GPU (예: 4 GPU)
torchrun --nproc_per_node=4 train.py config/my_experiment.yaml
```

## 핵심 차이점 (vs. 원본 Moshi moshi-finetune)

| 항목 | Moshi | Personaplex |
|---|---|---|
| LoRA | 지원 | **미지원** (full finetuning) |
| Forward API | `model(codes, condition_tensors)` | `model.forward_train(codes)` |
| Conditioning | `condition_provider` 사용 | 없음 |
| n_q / dep_q | 8 / 8 | **16 / 8** |
