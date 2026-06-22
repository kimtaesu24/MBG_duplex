"""
count_dataset.py
================
train.py와 동일한 sphn.dataset_jsonl 경로로 실제 샘플 수를 확인합니다.
Mimi 인코딩은 건너뛰고 카운팅만 수행합니다.

사용법:
    python count_dataset.py config/ami_vap_gpt_voice_face.yaml

    # 멀티GPU 시뮬레이션 (4-GPU 기준 rank 0 기준 갯수):
    python count_dataset.py config/ami_vap_gpt_voice_face.yaml --world_size 4

    # eval split 확인:
    python count_dataset.py config/ami_vap_gpt_voice_face.yaml --split eval
"""
import argparse
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../moshi")))

import sphn
from finetune.args import TrainArgs
from finetune.data.dataset import parse_data_sources, DataDir, DataFile


def count_sphn(jsonl_path: str, duration_sec: float, sample_rate: int,
               rank: int, world_size: int) -> dict:
    """sphn.dataset_jsonl 로 직접 이터레이션하여 샘플 수를 셉니다."""
    dataset = sphn.dataset_jsonl(
        str(jsonl_path),
        duration_sec=duration_sec,
        num_threads=4,
        sample_rate=sample_rate,
        pad_last_segment=True,
    )
    dataset = dataset.seq(skip=rank, step_by=world_size)

    n_samples = 0
    total_audio_sec = 0.0
    start_times = defaultdict(int)

    for sample in dataset:
        n_samples += 1
        wav_len = sample["unpadded_len"]
        total_audio_sec += wav_len / sample_rate
        start_times[round(sample["start_time_sec"], 1)] += 1

    return {
        "n_samples": n_samples,
        "total_audio_sec": total_audio_sec,
        "start_time_dist": dict(sorted(start_times.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config 파일 경로")
    parser.add_argument("--world_size", type=int, default=1,
                        help="시뮬레이션할 GPU 수 (기본값: 1)")
    parser.add_argument("--rank", type=int, default=0,
                        help="확인할 rank (기본값: 0)")
    parser.add_argument("--split", choices=["train", "eval"], default="train",
                        help="확인할 split (기본값: train)")
    args = parser.parse_args()

    train_args: TrainArgs = TrainArgs.load(args.config, drop_extra_fields=False)

    duration_sec = train_args.duration_sec
    sample_rate = 24000  # Mimi 고정값

    pretrain_data = train_args.data.train_data if args.split == "train" else train_args.data.eval_data
    if not pretrain_data:
        print(f"[ERROR] {args.split} 데이터 경로가 설정되지 않았습니다.")
        sys.exit(1)

    sources, weights = parse_data_sources(pretrain_data)

    print("=" * 60)
    print(f"Config       : {args.config}")
    print(f"Split        : {args.split}")
    print(f"duration_sec : {duration_sec}s")
    print(f"sample_rate  : {sample_rate} Hz")
    print(f"world_size   : {args.world_size}  (rank={args.rank} 기준)")
    print("=" * 60)

    grand_total = 0
    grand_sec = 0.0

    for i, (source, weight) in enumerate(zip(sources, weights)):
        jsonl_files = source.jsonl_files
        print(f"\n[Source {i+1}]  weight={weight:.3f}  ({source.path})")

        src_total = 0
        src_sec = 0.0

        for jsonl_path in jsonl_files:
            result = count_sphn(
                str(jsonl_path),
                duration_sec=duration_sec,
                sample_rate=sample_rate,
                rank=args.rank,
                world_size=args.world_size,
            )
            n = result["n_samples"]
            sec = result["total_audio_sec"]
            src_total += n
            src_sec += sec

            print(f"  {jsonl_path.name}")
            print(f"    samples (rank={args.rank}) : {n:,}")
            print(f"    audio duration      : {sec/3600:.2f}h")

            # start_time 분포 요약 (처음 몇 개만)
            dist = result["start_time_dist"]
            unique_starts = sorted(dist.keys())
            print(f"    unique start_times  : {len(unique_starts)}")
            preview = {k: dist[k] for k in unique_starts[:5]}
            if len(unique_starts) > 5:
                preview["..."] = "..."
            print(f"    start_time sample   : {preview}")

        print(f"  ── Source {i+1} 합계: {src_total:,} samples  ({src_sec/3600:.2f}h)")
        grand_total += src_total
        grand_sec += src_sec

    print()
    print("=" * 60)
    print(f"TOTAL (rank={args.rank}, world_size={args.world_size})")
    print(f"  실제 샘플 수  : {grand_total:,}")
    print(f"  오디오 길이   : {grand_sec/3600:.2f}h")
    if args.world_size > 1:
        estimated_all = grand_total * args.world_size
        print(f"  전체 추정 (x{args.world_size}): {estimated_all:,}")
    print("=" * 60)

    if args.world_size == 1:
        batch_size = train_args.batch_size
        num_mb = train_args.num_microbatches
        steps_per_epoch = grand_total // batch_size
        print(f"\nbatch_size={batch_size}, num_microbatches={num_mb} 기준:")
        print(f"  steps/epoch (단순)  : {steps_per_epoch:,}")
        print(f"  max_steps={train_args.max_steps:,}  →  {train_args.max_steps / max(steps_per_epoch,1):.1f} epochs")


if __name__ == "__main__":
    main()
