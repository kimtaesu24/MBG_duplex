"""Run test_inference2.py on one shard of the DualTalk OOD test jsonl.

You launch one process per GPU yourself; this script just takes the total
process count and this process's id, writes its own shard jsonl, and runs
test_inference2.py on it.

Example (4 GPUs, one process each):
    CUDA_VISIBLE_DEVICES=3 python run_dualtalk_ood_infer.py -n 4 -i 0
    CUDA_VISIBLE_DEVICES=4 python run_dualtalk_ood_infer.py -n 4 -i 1
    CUDA_VISIBLE_DEVICES=5 python run_dualtalk_ood_infer.py -n 4 -i 2
    CUDA_VISIBLE_DEVICES=6 python run_dualtalk_ood_infer.py -n 4 -i 3

Quick check:
    CUDA_VISIBLE_DEVICES=3 python run_dualtalk_ood_infer.py -n 1 -i 0 --limit 2
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
EXP_ROOT = FINETUNE_DIR / "output/hyades/output/ami_endtoend_v7_2_online_teacher"

DEFAULT_CKPT = EXP_ROOT / "checkpoints/checkpoint_000500"
DEFAULT_CONFIG = EXP_ROOT / "args.yaml"
DEFAULT_TEST_JSONL = FINETUNE_DIR / "experiments_dualtalk/dualtalk_ood.jsonl"


def load_jsonl(path: Path) -> list:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    p = argparse.ArgumentParser("DualTalk OOD sharded inference")
    p.add_argument("--num-procs", "-n", type=int, default=1,
                   help="Total number of processes you are launching.")
    p.add_argument("--proc-id", "-i", type=int, default=0,
                   help="This process's id (0 <= id < num_procs).")
    p.add_argument("--test-jsonl", type=str, default=str(DEFAULT_TEST_JSONL),
                   help="Test set jsonl (default: experiments_dualtalk/dualtalk_ood.jsonl).")
    p.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--voice-sample", type=str, default=None,
                   help="Override the voice_sample of every entry in the jsonl.")
    p.add_argument("--no-voice-prompt", action="store_true",
                   help="Run without any voice prompt.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Default: <exp>/dualtalk_ood_infer/<ckpt_name>/fusion_bc025_vad025")
    p.add_argument("--limit", type=int, default=0,
                   help="Use only the first N entries of the jsonl (0 = all).")
    p.add_argument("--skip-done", action="store_true",
                   help="Skip entries that already have a result in the output dir.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fusion-bc-weight", type=float, default=0.25)
    p.add_argument("--fusion-vap-weight", type=float, default=0.0)
    p.add_argument("--fusion-vad-weight", type=float, default=0.25)
    p.add_argument("--epad-control", type=str, default="fusion",
                   choices=("none", "legacy", "fusion"))
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="Extra args forwarded to test_inference2.py (after --).")
    args = p.parse_args()

    if args.num_procs < 1:
        sys.exit("[ERROR] --num-procs must be >= 1")
    if not (0 <= args.proc_id < args.num_procs):
        sys.exit(f"[ERROR] --proc-id must be in [0, {args.num_procs})")

    test_jsonl = Path(args.test_jsonl)
    if not test_jsonl.is_file():
        sys.exit(f"[ERROR] Test jsonl not found: {test_jsonl}")

    ckpt_dir = Path(args.ckpt_dir)
    if not (ckpt_dir / "consolidated/lora.safetensors").is_file() and \
       not (ckpt_dir / "consolidated/consolidated.safetensors").is_file():
        sys.exit(f"[ERROR] No checkpoint weights under {ckpt_dir}/consolidated/")
    if not Path(args.config).is_file():
        sys.exit(f"[ERROR] Config not found: {args.config}")

    out_dir = Path(args.output_dir) if args.output_dir else (
        EXP_ROOT / "dualtalk_ood_infer" / ckpt_dir.name / "fusion_bc025_vad025"
    )
    shard_dir = out_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    entries = load_jsonl(test_jsonl)
    if not entries:
        sys.exit(f"[ERROR] Test jsonl is empty: {test_jsonl}")
    total = len(entries)
    if args.limit > 0:
        entries = entries[:args.limit]

    # Resolve voice prompts. test_inference2.py looks for the .pt cache next to
    # each entry's voice_sample wav; warn early if it is missing.
    if args.no_voice_prompt:
        for e in entries:
            e.pop("voice_sample", None)
    elif args.voice_sample:
        for e in entries:
            e["voice_sample"] = args.voice_sample
    missing_pt = sorted({
        os.path.splitext(e["voice_sample"])[0] + ".pt"
        for e in entries
        if e.get("voice_sample")
        and not os.path.isfile(os.path.splitext(e["voice_sample"])[0] + ".pt")
    })
    if missing_pt:
        print(f"[WARN] {len(missing_pt)} voice prompt .pt cache(s) missing; those "
              f"samples fall back to the default voice. e.g. {missing_pt[0]}")

    # Shard here (not via --world-size/--rank) so each process owns its own
    # jsonl file and nothing is written concurrently.
    shard = entries[args.proc_id::args.num_procs]

    skipped = 0
    if args.skip_done:
        kept = []
        for e in shard:
            base = Path(e["path"]).stem
            if (out_dir / f"{base}.json").exists() and \
               (out_dir / f"{base}_output.wav").exists():
                skipped += 1
            else:
                kept.append(e)
        shard = kept

    shard_jsonl = shard_dir / f"inputs_{args.proc_id}of{args.num_procs}.jsonl"
    with open(shard_jsonl, "w") as f:
        for e in shard:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f" proc      : {args.proc_id} / {args.num_procs}"
          f"   (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")
    print(f" ckpt      : {ckpt_dir}")
    print(f" test set  : {test_jsonl}  (total {total})")
    print(f" this shard: {len(shard)} samples"
          + (f"  (skip-done {skipped})" if args.skip_done else ""))
    print(f"             {shard_jsonl}")
    print(f" fusion    : {args.epad_control} bc={args.fusion_bc_weight} "
          f"vap={args.fusion_vap_weight} vad={args.fusion_vad_weight}")
    print(f" voice     : {'disabled' if args.no_voice_prompt else (args.voice_sample or 'from jsonl')}")
    print(f" output    : {out_dir}")
    print("=" * 60, flush=True)

    if not shard:
        print("[INFO] Nothing to do for this shard.")
        return

    cmd = [
        sys.executable, "test_inference2.py",
        "--config", str(args.config),
        "--test-jsonl", str(shard_jsonl),
        "--ckpt-dir", str(ckpt_dir),
        "--output-dir", str(out_dir),
        "--device", args.device,
        "--seed", str(args.seed),
        "--epad-control", args.epad_control,
        "--fusion-bc-weight", str(args.fusion_bc_weight),
        "--fusion-vap-weight", str(args.fusion_vap_weight),
        "--fusion-vad-weight", str(args.fusion_vad_weight),
    ]
    if args.no_voice_prompt:
        cmd.append("--no-voice-prompt")
    cmd += [a for a in args.extra if a != "--"]

    print("[RUN] " + " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd, cwd=str(FINETUNE_DIR)).returncode)


if __name__ == "__main__":
    main()
