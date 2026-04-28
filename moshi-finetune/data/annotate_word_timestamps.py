import glob
import json
import os

STEREO_DIR   = '/home2/s20235100/Conversational-AI/personaplex_MBG/moshi-finetune/data/stereo_ami_balanced/data_stereo'
DATASET_ROOT = '/home2/s20235100/data_process/AMI_corpus/dataset'
SAVE_DIR     = '/home2/s20235100/Conversational-AI/personaplex_MBG/moshi-finetune/data/stereo_ami_balanced/data_stereo'

# ------------------------------------------------------------------
# Build index: session_id -> record from our dataset
# ------------------------------------------------------------------
print("Loading dataset records...")
dataset = {}
for path in glob.glob(f'{DATASET_ROOT}/*/*.json'):
    for rec in json.load(open(path, 'r', encoding='utf-8')):
        sid = rec['bc_audio'].split('/')[-1].replace('_bc.wav', '').replace('.wav', '')
        dataset[sid] = rec
print(f"  {len(dataset)} records indexed")

# ------------------------------------------------------------------
# Annotate each stereo JSON
# ------------------------------------------------------------------
stereo_jsons = sorted(glob.glob(f'{STEREO_DIR}/*.json'))
print(f"Annotating {len(stereo_jsons)} stereo JSON files...")

matched = 0
missing = 0
for stereo_path in stereo_jsons:
    fname = os.path.basename(stereo_path)       # EN2001a_2_stereo.json
    sid   = fname.replace('_stereo.json', '')   # EN2001a_2

    rec = dataset.get(sid)
    if rec is None:
        missing += 1
        continue

    alignments = []

    # Backchannel words — keep relative_start/end as-is (relative to backchannel start)
    for bc in rec['backchannels']:
        for w in bc['words']:
            if w['relative_start'] is None or w['relative_end'] is None:
                continue
            alignments.append([w['word'], [w['relative_start'], w['relative_end']], "SPEAKER_MAIN"])

    out = {"alignments": alignments}
    out_path = os.path.join(SAVE_DIR, fname)
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    matched += 1

print(f"Done — annotated: {matched}, no match: {missing}")
