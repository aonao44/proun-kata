# Assets

## Sample Transcriptions

- `sample_transcriptions.json`: maps audio SHA-1 digests to reference phoneme sequences used by the stub pipeline for demos.
  - `assets/samples/can_I.wav` → digest `99acd520267258bbeeacf5fc0e3e1c17452ae69b`
  - Source audio: internal pronunciation practice sample (16kHz mono WAV).

## Samples

- `samples/can_I.wav`: short phrase "can I" recorded for testing.

## Phoneme Language Models

- `phonelm/phone_bigram.json`: heuristic English phoneme bigram log-likelihoods generated in-repo to guide beam-search decoding.
