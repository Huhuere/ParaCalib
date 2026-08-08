# ParaCalib: Semantically Calibrated Paralinguistic Modeling for Depression Detection

ParaCalib uses an audio-language model (ALM) captioner to generate contextualized vocal descriptions from speech. An LLM-based Paralinguistic State Extractor (PSE) then maps these descriptions into structured **SC-Para** states. Finally, an attention-based multiple instance learning (MIL) classifier aggregates utterance-level SC-Para vectors into a subject-level representation for binary depression classification.

The pipeline consists of an optional dataset-specific preprocessing stage followed by three main stages corresponding to the core components of the proposed framework.

0. **`stage0_daic_woz_preprocessing.py`** — *DAIC-WOZ audio segmentation.*
   Segments the DAIC-WOZ interview corpus into participant-only, utterance-level WAV clips and generates the corresponding PHQ-8 binary and score label files. The resulting utterance-level audio clips are used as input to Stage 1.

1. **`stage1_alm_captioning.py`** — *ALM-based vocal captioning.*
   Batch-sends utterance-level audio clips to an audio-language captioner served through vLLM, such as Qwen3-Omni-Captioner, and collects free-text contextualized vocal descriptions.

2. **`stage2_pse_extraction.py`** — *PSE-based SC-Para extraction.*
   Sends each vocal caption to a reasoning LLM served through vLLM, such as DeepSeek-R1-Distill-Qwen, acting as the Paralinguistic State Extractor (PSE). For each utterance, the PSE produces a structured SC-Para JSON representation containing five categorical paralinguistic states, a depression-related paralinguistic evidence score, and a confidence score.

3. **`stage3_attentive_mil.py`** — *Attention-based MIL classification.*
   Trains an additive-attention MIL classifier that assigns learned attention weights to a subject's utterance-level SC-Para vectors and aggregates them into a subject-level representation for binary depression classification.

## Installation

The code requires Python 3.9 or later.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

Stages 1 and 2 require a running [vLLM](https://github.com/vllm-project/vllm) server exposing an OpenAI-compatible API.

For example, an ALM captioner for Stage 1 can be served on port `8901`:

```bash
vllm serve <captioner-model> \
    --port 8901 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --allowed-local-media-path / \
    -tp 1
```

## Getting Started

### 0. DAIC-WOZ audio segmentation

This preprocessing stage is optional and specific to the DAIC-WOZ dataset.

DAIC-WOZ ships separate `train`/`dev`/`test` label CSVs, each listing a different set of
subjects; run the script once per split, pointing `LABELS_CSV`/`OUTPUT_DIR` at that split:

Configure the required paths using environment variables, or edit the corresponding default values in `stage0_daic_woz_preprocessing.py`:

```bash
export AUDIO_DIR=path/to/daic_woz/audio_wav
export TRANS_DIR=path/to/daic_woz/transcripts

export LABELS_CSV=path/to/daic_woz/train_split_Depression_AVEC2017.csv
export OUTPUT_DIR=daic_woz_train_utterances
python stage0_daic_woz_preprocessing.py

export LABELS_CSV=path/to/daic_woz/dev_split_Depression_AVEC2017.csv
export OUTPUT_DIR=daic_woz_dev_utterances
python stage0_daic_woz_preprocessing.py

export LABELS_CSV=path/to/daic_woz/test_split_Depression_AVEC2017.csv
export OUTPUT_DIR=daic_woz_test_utterances
python stage0_daic_woz_preprocessing.py
```

The generated clips follow the naming convention:

```text
<participant_id>_s<n>_AUDIO.wav
```

These utterance-level WAV files are used as input to Stage 1.

### 1. ALM-based vocal captioning

Configure the audio directory, API endpoint, and output path in the configuration block of `stage1_alm_captioning.py`.

For example, if the ALM captioner is served locally on port `8901`, configure the script to use the corresponding OpenAI-compatible endpoint and then run:

```bash
python stage1_alm_captioning.py
```

The script processes utterance-level WAV files and stores the generated vocal descriptions for subsequent PSE extraction.

### 2. PSE-based SC-Para extraction

Configure the Stage 2 input, vLLM endpoint, model name, and output directory:

```bash
export BATCH_JSON=batch_results_api.json
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
export OUTPUT_DIR=caption_analysis_results_train
export MODE=sequential

python stage2_pse_extraction.py
```

`MODE` can be set to:

```text
sequential
```

or:

```text
concurrent
```

depending on the desired request mode.

By default, run the full dataset without setting `SAMPLE_MODE`. For subset or debugging runs, `SAMPLE_MODE` can optionally be set to:

```bash
export SAMPLE_MODE=subjects
```

or:

```bash
export SAMPLE_MODE=utterances
```

The resulting SC-Para representations are written as per-utterance JSON files under `OUTPUT_DIR`.

### 3. Attention-based MIL classification

Train and evaluate the attentive MIL classifier with:

```bash
python stage3_attentive_mil.py \
    --root_dir caption_analysis_results_train \
    --labels_csv labels.csv \
    --val_root_dir caption_analysis_results_dev \
    --val_labels_csv labels_dev.csv \
    --test_root_dir caption_analysis_results_test \
    --test_labels_csv labels_test.csv \
    --seeds "50,150,250,350,450" \
    --output_csv results_attn_pooling.csv \
    --run_log run_logs \
    --device cuda
```

Each label CSV must contain the columns:

```text
conv_id,label
```

where `label` is the binary depression label (`0` or `1`) and `conv_id` corresponds to the subject directory containing the per-utterance SC-Para JSON files produced by Stage 2.

Use:

```bash
python stage3_attentive_mil.py --help
```

to view the full list of command-line options.

## Dataset

This repository does not distribute audio recordings, transcripts, clinical labels, or other restricted dataset files.

The datasets used in the paper are third-party resources and must be obtained directly from their respective providers under the applicable data-use agreements.

### DAIC-WOZ

**[DAIC-WOZ](https://dcapswoz.ict.usc.edu/)**
Distress Analysis Interview Corpus — Wizard of Oz, USC Institute for Creative Technologies.

Access can be requested through the official [DAIC-WOZ download form](https://dcapswoz.ict.usc.edu/daic-woz-database-download/).

```bibtex
@inproceedings{gratch2014distress,
  title={The Distress Analysis Interview Corpus of human and computer interviews},
  author={Gratch, Jonathan and Artstein, Ron and Lucas, Gale M and Stratou, Giota and Scherer, Stefan
          and Nazarian, Angela and Wood, Rachel and Boberg, Jill and DeVault, David and Marsella, Stacy
          and Traum, David R and Rizzo, Skip and Morency, Louis-Philippe},
  booktitle={Proceedings of LREC},
  pages={3123--3128},
  year={2014}
}
```

The provided `stage0_daic_woz_preprocessing.py` script is specifically designed for DAIC-WOZ.

After obtaining the dataset, configure `AUDIO_DIR`, `TRANS_DIR`, and `LABELS_CSV` as described in [Getting Started](#getting-started).


