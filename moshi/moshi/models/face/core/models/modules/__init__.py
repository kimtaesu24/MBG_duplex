#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

from .metrics import calc_mesh_loss, calc_val_metrics

# Lazy imports — MimiModelWrapper and Wav2Vec2Model pull in heavy dependencies
# (torchaudio, transformers) that are not always available.  Only import them
# when explicitly requested to avoid ImportError at package load time.
def __getattr__(name):
    if name == "MimiModelWrapper":
        from .mimi import MimiModelWrapper
        return MimiModelWrapper
    if name == "Wav2Vec2Model":
        from .wav2vec import Wav2Vec2Model
        return Wav2Vec2Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
