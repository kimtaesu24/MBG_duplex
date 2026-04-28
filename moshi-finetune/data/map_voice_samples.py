import glob
import json
import os

TEST_DIR     = '/home2/s20235100/Conversational-AI/personaplex_MBG/moshi-finetune/data/stereo_ami_balanced_train'
DATASET_ROOT = '/home2/s20235100/data_process/AMI_corpus/dataset'
INPUT_JSONL  = f'{TEST_DIR}/data.jsonl'
SAVE_JSONL   = f'{TEST_DIR}/data_with_voice_sample.jsonl'

# ------------------------------------------------------------------
# Build index: session_id -> voice sample path
# ------------------------------------------------------------------
print("Loading dataset records...")
voice_sample_index = {}
for path in glob.glob(f'{DATASET_ROOT}/*/*.json'):
    for rec in json.load(open(path, 'r', encoding='utf-8')):
        sid          = rec['bc_audio'].split('/')[-1].replace('_bc.wav', '').replace('.wav', '')
        voice_sample = rec.get('backchannel_speaker_voice_sample')
        if voice_sample is None:
            headset = rec.get('backchannel_headset')
            if headset:
                session   = headset.split('.')[0]
                candidate = f'{DATASET_ROOT}/{session}/voice_samples/{headset}.wav'
                voice_sample = candidate if os.path.exists(candidate) else None
        voice_sample_index[sid] = voice_sample
print(f"  {len(voice_sample_index)} records indexed")

# ------------------------------------------------------------------
# Read data.jsonl, annotate each entry with voice_sample, save
# ------------------------------------------------------------------
missing   = 0
out_lines = []
with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        entry        = json.loads(line)
        sid          = os.path.basename(entry['path']).replace('_stereo.wav', '')
        voice_sample = voice_sample_index.get(sid)
        if voice_sample is None:
            print(f"  No voice sample for {sid}")
            missing += 1
        entry['voice_sample'] = voice_sample
        out_lines.append(entry)

with open(SAVE_JSONL, 'w', encoding='utf-8') as f:
    for entry in out_lines:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Done — total: {len(out_lines)}, missing: {missing}")
print(f"Saved to: {SAVE_JSONL}")
