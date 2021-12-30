---
layout: page
title: Automatic Speech Recognition and Text-to-Speech
---

### What is the speech recognition?
Mapping acoustic waveforms to sequences of graphemes

### What is the input to a speech recognizer?
A series of acoustic waves that are sampled, quantized, and converted to a spectral representation like the log mel spectrum.

### What are the two common paradigms for speech recognition?
Two common paradigms for speech recognition are the encoder-decoder with attention model, and models based on the CTC loss function. Attention based models have higher accuracies, but models based on CTC more easily adapt to streaming: outputting graphemes online instead of waiting until the acoustic input is complete.

### How is ASR evaluated?
ASR is evaluated using the Word Error Rate; the edit distance between the hypothesis and the gold transcription.

### What is the architecture for TTS?
TTS systems are also based on the encoder-decoder architecture. The encoder maps letters to an encoding, which is consumed by the decoder which generates mel spectrogram output. A neural vocoder then reads the spectrogram and generates waveforms.

### What is the role of text normalization in TTS?
TTS systems require a first pass of text normalization to deal with numbers and abbreviations and other non-standard words.

### How is TTS evaluated?
TTS is evaluated by playing a sentence to human listeners and having them give a mean opinion score (MOS) or by doing AB tests.