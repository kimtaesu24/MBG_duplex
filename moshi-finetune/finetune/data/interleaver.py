import json
import math
import os
from collections import deque
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece
import torch
# moshi.conditioners does not exist in the personaplex version of moshi.
# ConditionAttributes is defined here as a local stub — personaplex training
# does not use condition tensors, so this is only needed as a type placeholder.
from dataclasses import field
from typing import Any


@dataclass
class ConditionAttributes:
    """Stub replacement for moshi.conditioners.ConditionAttributes.
    Personaplex training does not use conditioning, so this is empty.
    """
    text: dict[str, Any] = field(default_factory=dict)
    wav: dict[str, Any] = field(default_factory=dict)

Alignment = tuple[str, tuple[float, float], str]
TokenizedAlignment = tuple[list[int], tuple[float, float], str]


@dataclass
class Sample:
    codes: torch.Tensor
    condition_attributes: ConditionAttributes | None = None
    vap_targets: torch.Tensor | None = None  # [1, T]
    voice_prompt_emb: torch.Tensor | None = None  # [V, dim]
    face_motion_gt: torch.Tensor | None = None    # [T_face, 54] at 25 fps
    valid_mask: torch.Tensor | None = None        # [T_mimi] bool: True = real audio, False = silence-padded
    valid_face_frames: int | None = None          # number of valid face frames (25 fps)


@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    vap_targets: torch.Tensor | None = None       # [B, T]
    voice_prompt_embs: torch.Tensor | None = None  # [B, V, dim]
    face_motion_gt: torch.Tensor | None = None    # [B, T_face, 54] at 25 fps
    valid_mask: torch.Tensor | None = None        # [B, T_mimi] bool
    valid_face_frames: torch.Tensor | None = None # [B] int: valid face frame count per sample

    @classmethod
    def collate(cls, batch: list[Sample]) -> "Batch":
        codes = torch.cat([b.codes for b in batch])
        vap_targets = None
        if batch[0].vap_targets is not None:
            vap_targets = torch.cat([b.vap_targets for b in batch])

        voice_prompt_embs = None
        if all(b.voice_prompt_emb is not None for b in batch):
            # cache shape is fixed [1, K, T] int64 — cat along batch dim, no padding needed
            voice_prompt_embs = torch.cat([b.voice_prompt_emb for b in batch], dim=0)  # [B, K, T]

        face_motion_gt = None
        if all(b.face_motion_gt is not None for b in batch):
            # Pad to max T_face in this batch before stacking
            max_t = max(b.face_motion_gt.shape[0] for b in batch)
            padded_face = []
            for b in batch:
                fm = b.face_motion_gt  # [T_face, 54]
                if fm.shape[0] < max_t:
                    pad = torch.zeros(max_t - fm.shape[0], fm.shape[1], dtype=fm.dtype)
                    fm = torch.cat([fm, pad], dim=0)
                padded_face.append(fm)
            face_motion_gt = torch.stack(padded_face, dim=0)  # [B, T_face, 54]

        valid_mask = None
        if all(b.valid_mask is not None for b in batch):
            valid_mask = torch.stack([b.valid_mask for b in batch], dim=0)  # [B, T_mimi]

        valid_face_frames = None
        if all(b.valid_face_frames is not None for b in batch):
            valid_face_frames = torch.tensor(
                [b.valid_face_frames for b in batch], dtype=torch.long
            )  # [B]

        if batch[0].condition_attributes is None:
            return Batch(codes, vap_targets=vap_targets, voice_prompt_embs=voice_prompt_embs,
                         face_motion_gt=face_motion_gt, valid_mask=valid_mask,
                         valid_face_frames=valid_face_frames)
        return Batch(codes, [b.condition_attributes for b in batch], vap_targets=vap_targets,
                     voice_prompt_embs=voice_prompt_embs, face_motion_gt=face_motion_gt,
                     valid_mask=valid_mask, valid_face_frames=valid_face_frames)


def tokenize(
    tokenizer: sentencepiece.SentencePieceProcessor,
    text: str,
    bos: bool = True,
    alpha: float | None = None,
):
    """Tokenize the given string, accounting for new lines, potentially adding a BOS token."""
    nl_piece = tokenizer.encode("\n")[-1]
    if alpha is not None:
        tokens = tokenizer.encode(
            text.split("\n"), enable_sampling=True, alpha=alpha, nbest_size=-1
        )
    else:
        tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: [*a, nl_piece, *b], tokens)
    if bos:
        tokens = [tokenizer.bos_id(), *tokens]
    return tokens


class Interleaver:
    """Interleaver with basic featuress
    Args:
        tokenizer: text tokenizer used by the model.
        audio_frame_rate (float): frame rate of the audio tokenizer.
        text_padding (int): special token used for text padding.
        end_of_text_padding (int): special token used to indicate end of text padding.
        zero_padding (int): special token id indicating that a 0 should be used instead
            of an actual embedding.
        in_word_padding (int | None): padding used within a word segment. Will default to `text_padding`.
        keep_main_only (bool): if True, will only keep the alignments with the main speaker.
        keep_and_shift (bool): if True, will not drop any alignment, except for those with negative duration.
        use_bos_eos: (bool): if True, inserts BOS, EOS for change of turns.
        audio_delay (float): delay between the text and audio.
            A positive value means the text will be ahead of the audio.
        proba (float): probability of keeping the text.
        device: device location for the output tensors.
    """

    def __init__(
        self,
        tokenizer: sentencepiece.SentencePieceProcessor,
        audio_frame_rate: float,
        text_padding: int,
        end_of_text_padding: int,
        zero_padding: int,
        in_word_padding: int | None = None,
        keep_main_only: bool = False,
        main_speaker_label: str = "SPEAKER_MAIN",
        use_bos_eos: bool = False,
        keep_and_shift: bool = False,
        audio_delay: float = 0.0,
        proba: float = 1.0,
        device: str | torch.device = "cuda",
    ):
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.zero_padding = zero_padding
        self.in_word_padding = (
            self.text_padding if in_word_padding is None else in_word_padding
        )
        self.keep_main_only = keep_main_only
        self.main_speaker_label = main_speaker_label
        self.use_bos_eos = use_bos_eos
        self.keep_and_shift = keep_and_shift
        self.audio_delay = audio_delay
        self.proba = proba
        self.device = device

    @property
    def special_tokens(self) -> set[int]:
        """Return the set of special tokens used by this interleaver."""
        return {
            self.text_padding,
            self.end_of_text_padding,
            self.tokenizer.bos_id(),
            self.tokenizer.eos_id(),
            self.zero_padding,
            self.in_word_padding,
        }

    def _tokenize(self, alignments: list[Alignment]) -> list[TokenizedAlignment]:
        # Tokenizes each word individually into a list of ints.
        out = []
        for word, ts, speaker in alignments:
            toks = tokenize(self.tokenizer, word.strip(), bos=False)
            out.append((toks, ts, speaker))
        return out

    def _keep_main_only(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        return [a for a in alignments if a[2] == main_speaker]

    def _keep_those_with_duration(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Removes all words with negative or 0 durations.
        return [a for a in alignments if a[1][0] < a[1][1]]

    def _add_delay(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Delay the audio with respect to the text, e.g. positive values mean the audio is late on the text.
        return [
            (a[0], (a[1][0] - self.audio_delay, a[1][1] - self.audio_delay), a[2])
            for a in alignments
            if a[1][1] > self.audio_delay
        ]

    def _insert_bos_eos(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        # EOS and BOS is different from what it was in the old Interleaver, it is now symmetrical:
        # if the main speaker talks after another speaker (or is the first to talk), BOS is prepended to the first word.
        # Similary, if any other speaker speaks either first, or after the main speaker, a EOS is prepended.
        # This is in contrast with the legacy Interleaver, where the EOS would be inserted immediately
        # at the end of the turn of the main speaker.
        out: list[TokenizedAlignment] = []
        last_speaker = None
        for toks, ts, speaker in alignments:
            toks = list(toks)
            if speaker == last_speaker:
                pass
            elif speaker == main_speaker:
                toks.insert(0, self.tokenizer.bos_id())
            elif last_speaker == main_speaker:
                assert out
                toks.insert(0, self.tokenizer.eos_id())
            last_speaker = speaker
            out.append((toks, ts, speaker))
        return out

    def build_token_stream(
        self,
        alignments: list[TokenizedAlignment] | None,
        segment_duration: float,
    ) -> torch.Tensor:
        """Builds the token stream from the tokenized alignments."""
        T = math.ceil(segment_duration * self.audio_frame_rate)
        if alignments is None:
            text_tokens = [self.zero_padding] * T
        else:
            text_tokens = [self.text_padding] * T
            i = 0
            to_append_stack: deque = deque()
            last_word_end = -1
            for t in range(T):
                while (
                    i < len(alignments)
                    and alignments[i][1][0] * self.audio_frame_rate < t + 1
                ):
                    tokenized = alignments[i][0]
                    last_word_end = int(alignments[i][1][1] * self.audio_frame_rate)
                    if self.keep_and_shift:
                        to_append_stack.extend(tokenized)
                    else:
                        to_append_stack = deque(tokenized)
                    i += 1
                if to_append_stack:
                    if t > 0 and text_tokens[t - 1] in [
                        self.text_padding,
                        self.in_word_padding,
                    ]:
                        text_tokens[t - 1] = self.end_of_text_padding
                    next_token = to_append_stack.popleft()
                    text_tokens[t] = next_token
                elif t <= last_word_end:
                    text_tokens[t] = self.in_word_padding
        if self.audio_delay < 0:
            prefix_length = int(self.audio_frame_rate * -self.audio_delay)
            text_tokens[:prefix_length] = [self.zero_padding] * prefix_length
        return torch.tensor(text_tokens, device=self.device).view(1, 1, -1)

    def prepare_item(
        self,
        alignments: list[Alignment] | None,
        segment_duration: float,
        main_speaker: str | None = None,
    ) -> torch.Tensor:
        """Responsible with processing the alignments and calling `build_token_stream`."""
        if alignments is None:
            tokenized = None
        else:
            tokenized = self._tokenize(sorted(alignments, key=lambda x: x[1][0]))
            if self.keep_main_only:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._keep_main_only(tokenized, main_speaker)
            elif self.use_bos_eos:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._insert_bos_eos(tokenized, main_speaker)
            tokenized = self._keep_those_with_duration(tokenized)
            if self.audio_delay != 0:
                tokenized = self._add_delay(tokenized)
        return self.build_token_stream(tokenized, segment_duration)


def dicho(alignment, val, i=0, j=None):
    if j is None:
        j = len(alignment)
    if i == j:
        return i
    k = (i + j) // 2
    if alignment[k][1][0] < val:
        return dicho(alignment, val, k + 1, j)
    else:
        return dicho(alignment, val, i, k)


_FLAME_PATH_CACHE: dict[str, Optional[Path]] = {}  # stem → resolved FLAME .npy path (or None)


class InterleavedTokenizer:
    def __init__(
        self,
        mimi,
        interleaver,
        duration_sec: float,
        vap_manifest_path: str = "",
        flame_root: str = "",
        flame_speaker: str = "bc",
    ):
        """
        Args:
            mimi: Mimi audio tokenizer.
            interleaver: Text interleaver.
            duration_sec: Segment duration in seconds.
            vap_manifest_path: Path to VAP label manifest JSON (optional).
            flame_root: Root directory of FLAME .npy files, structured as
                ``{flame_root}/{speaker}/{split}/{stem}_{speaker}.npy``
                where speaker ∈ {"bc", "ut"} and split ∈ {"train", "valid", "test"}.
            flame_speaker: Primary speaker suffix for the agent channel ("bc" or "ut").
        """
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        self.mimi_sample_rate = mimi.sample_rate  # typically 24000

        # VAP manifest lookup table
        self.vap_lookup = {}  # (file_id, vap_step_index) -> label_int
        self.vap_hop_s = 0.08  # default
        if vap_manifest_path and os.path.exists(vap_manifest_path):
            print(f"Loading VAP manifest from {vap_manifest_path}...")
            with open(vap_manifest_path, 'r') as f:
                manifest = json.load(f)
                self.vap_hop_s = manifest.get("config", {}).get("hop_duration_s", 0.08)
                for s in manifest.get("samples", []):
                    f_id = s["file_id"]
                    # Store by (file_id, vap_step_index); step = offset_seconds / hop_s
                    idx = int(round(s["offset_seconds"] / self.vap_hop_s))
                    self.vap_lookup[(f_id, idx)] = s["label_int"]
            print(f"Loaded {len(self.vap_lookup)} VAP targets (hop={self.vap_hop_s}s).")

        # FLAME / 3DMM face motion configuration
        self.flame_root = flame_root
        self.flame_speaker = flame_speaker
        self._motion_fps = 25  # FLAME data frame rate

    def tokenize_with_encoded_audio(
        self,
        audio_tokens: torch.Tensor,
        start_sec: float,
        path: str,
        voice_prompt_emb: torch.Tensor | None = None,
        actual_wav_samples: int | None = None,
    ) -> Sample:
        """Finish tokenization given pre-encoded mimi tokens.

        audio_tokens: raw output of mimi.encode for one sample — shape [C, K, T_enc]
        where C = number of audio channels (typically 2 for stereo conversations).
        Called by the batched encoding path in data_loader so mimi.encode is invoked
        only once per batch rather than once per sample.
        """
        audio_tokens = audio_tokens[..., : self.num_audio_frames]
        this_num_audio_frames = audio_tokens.shape[-1]
        audio_tokens = torch.nn.functional.pad(
            audio_tokens[..., : self.num_audio_frames],
            (0, self.num_audio_frames - this_num_audio_frames),
            value=self.interleaver.zero_padding,
        )
        audio_tokens = audio_tokens.view(1, -1, self.num_audio_frames)

        info_file = os.path.splitext(path)[0] + ".json"
        with open(info_file) as f:
            data = json.load(f)
        alignments = data["alignments"]

        start_alignment = dicho(alignments, start_sec)
        end_alignment = dicho(alignments, start_sec + self.duration_sec)
        alignments = [
            (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
            for a in alignments[start_alignment:end_alignment]
        ]

        text_tokens = self.interleaver.prepare_item(alignments, this_num_audio_frames)
        text_tokens = torch.nn.functional.pad(
            text_tokens,
            (0, self.num_audio_frames - text_tokens.shape[-1]),
            value=self.interleaver.zero_padding,
        )

        codes = torch.cat([text_tokens, audio_tokens], dim=1)

        # --- VAP Target Processing ---
        vap_targets = None
        if self.vap_lookup:
            vap_targets = torch.full((1, self.num_audio_frames), -100, dtype=torch.long, device=codes.device)
            raw_file_id = os.path.splitext(os.path.basename(path))[0]

            base = raw_file_id.replace("_stereo", "")
            file_id_candidates = [raw_file_id, base + "_ut", base]

            moshi_frame_duration = 1.0 / self.mimi.frame_rate
            t_arr = np.arange(self.num_audio_frames, dtype=np.float64)
            vap_steps = np.rint(
                (start_sec + t_arr * moshi_frame_duration) / self.vap_hop_s
            ).astype(np.int64)

            matched_fid = None
            for fid in file_id_candidates:
                labels = np.array(
                    [self.vap_lookup.get((fid, int(s)), -100) for s in vap_steps],
                    dtype=np.int64,
                )
                valid = labels != -100
                if valid.any():
                    vap_targets[0, valid] = torch.from_numpy(labels[valid]).to(codes.device)
                    matched_fid = fid
                    break

            if matched_fid is None:
                print(f"[WARNING] No VAP target found for '{raw_file_id}' "
                      f"All targets set to -100.")

        # ── FLAME / 3DMM face motion ─────────────────────────────────────────
        face_motion_gt = None
        if self.flame_root:
            face_motion_gt = self._load_face_motion(path, start_sec)

        # ── Silence-padding validity mask ─────────────────────────────────────
        # When a short clip is zero-padded to batch max_T before mimi encoding,
        # the padded region produces silence mimi tokens that should be excluded
        # from all losses (audio, text, VAP, face).
        valid_mask = None
        valid_face_frames = None
        if actual_wav_samples is not None:
            actual_mimi_frames = min(
                math.ceil(actual_wav_samples * self.mimi.frame_rate / self.mimi_sample_rate),
                self.num_audio_frames,
            )
            valid_mask = torch.zeros(self.num_audio_frames, dtype=torch.bool)
            valid_mask[:actual_mimi_frames] = True

            n_face_frames = int(self.duration_sec * self._motion_fps)
            valid_face_frames = min(
                int(actual_wav_samples / self.mimi_sample_rate * self._motion_fps),
                n_face_frames,
            )

        return Sample(codes, data.get("text_conditions", None), vap_targets=vap_targets,
                      voice_prompt_emb=voice_prompt_emb, face_motion_gt=face_motion_gt,
                      valid_mask=valid_mask, valid_face_frames=valid_face_frames)

    def _find_flame_path(self, stem: str) -> Optional[Path]:
        """Return the FLAME .npy path for *stem*, or None if not found.

        Search order: configured primary speaker, then fallback speaker.
        For each speaker, tries subdirectory patterns:
          {flame_root}/{speaker}/{split}/{stem}_{speaker}.npy
          {flame_root}/{split}/{stem}_{speaker}.npy   (flat layout)
        Results are cached per (stem, speaker) pair.
        """
        cache_key = f"{stem}|{self.flame_speaker}"
        if cache_key in _FLAME_PATH_CACHE:
            return _FLAME_PATH_CACHE[cache_key]

        root = Path(self.flame_root)
        speakers = [self.flame_speaker, "ut" if self.flame_speaker == "bc" else "bc"]
        splits = ["train", "valid", "val", "test"]
        result: Optional[Path] = None

        for speaker in speakers:
            target = f"{stem}_{speaker}"
            for split in splits:
                # Standard layout: {root}/{speaker}/{split}/{target}.npy
                p = root / speaker / split / f"{target}.npy"
                if p.exists():
                    result = p
                    break
                # Flat layout: {root}/{split}/{target}.npy
                p = root / split / f"{target}.npy"
                if p.exists():
                    result = p
                    break
            if result is not None:
                break

        _FLAME_PATH_CACHE[cache_key] = result
        return result

    def _load_face_motion(self, audio_path: str, start_sec: float) -> Optional[torch.Tensor]:
        """Load the FLAME motion window aligned to this audio segment.

        Returns a float32 tensor of shape [T_face, 54] where T_face = duration_sec * 25,
        or None when no matching FLAME file is found.
        """
        n_face_frames = int(self.duration_sec * self._motion_fps)
        start_frame = int(start_sec * self._motion_fps)

        # Derive the base stem by stripping the audio extension and "_stereo" suffix.
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        stem = stem.replace("_stereo", "")

        flame_path = self._find_flame_path(stem)
        if flame_path is None:
            return None

        try:
            arr = np.load(flame_path, allow_pickle=True)
            if arr.dtype == object:
                # Array of per-frame dicts → parse each frame
                motion_list = []
                for frame in arr:
                    expr = np.asarray(frame.get("expr", []), dtype=np.float32)
                    if expr.shape[0] < 50:
                        expr = np.pad(expr, (0, 50 - expr.shape[0]))
                    expr = expr[:50]
                    jaw = np.asarray(frame.get("jaw_pose", [0.0, 0.0, 0.0]), dtype=np.float32)[:1]
                    neck = np.asarray(frame.get("neck_pose", [0.0, 0.0, 0.0]), dtype=np.float32)[:3]
                    motion_list.append(np.concatenate([expr, jaw, neck]))
                motion = np.stack(motion_list, axis=0).astype(np.float32)
            else:
                motion = np.asarray(arr, dtype=np.float32)
                if motion.shape[-1] == 56:
                    # 56-dim (expr50 + jaw3 + neck3) → 54-dim (expr50 + jaw1 + neck3)
                    motion = np.concatenate([motion[..., :50], motion[..., 50:51], motion[..., 53:56]], axis=-1)
                elif motion.shape[-1] > 54:
                    motion = motion[..., :54]
        except Exception:
            return None

        if motion.ndim != 2 or motion.shape[-1] != 54:
            return None

        # Slice temporal window matching the audio segment.
        end_frame = start_frame + n_face_frames
        if start_frame >= motion.shape[0]:
            return None
        window = motion[start_frame:end_frame]
        if window.shape[0] < n_face_frames:
            pad = np.zeros((n_face_frames - window.shape[0], 54), dtype=np.float32)
            window = np.concatenate([window, pad], axis=0)
        return torch.from_numpy(window).float()  # [T_face, 54]

    def __call__(self, wav: np.ndarray, start_sec: float, path: str, voice_prompt_emb: torch.Tensor | None = None) -> Sample:
        with torch.no_grad():
            audio_tensor = torch.Tensor(wav).cuda()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # [T] -> [1, T]
            audio_tokens = self.mimi.encode(audio_tensor[:, None])
        return self.tokenize_with_encoded_audio(audio_tokens, start_sec, path, voice_prompt_emb)
