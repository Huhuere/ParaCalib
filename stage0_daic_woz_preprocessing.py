"""Stage 0: DAIC-WOZ per-utterance audio segmentation.

Segments the DAIC-WOZ interview corpus into participant-only, utterance-level
WAV clips for the Stage 1 ALM captioner. For each subject, participant turns
are read from the transcript CSV, filtered by a minimum duration, and exported
as individual clips with matching PHQ-8 binary/score label files.

DAIC-WOZ ships separate train/dev/test label CSVs; run this script once per
split, setting LABELS_CSV and OUTPUT_DIR accordingly each time.
"""

import os

import pandas as pd
from pydub import AudioSegment

# ========== Configuration ==========
AUDIO_DIR = os.getenv("AUDIO_DIR", "path/to/daic_woz/audio_wav")
TRANS_DIR = os.getenv("TRANS_DIR", "path/to/daic_woz/transcripts")
LABELS_CSV = os.getenv("LABELS_CSV", "path/to/daic_woz/train_split_Depression_AVEC2017.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "train_utterances")

MIN_DURATION = float(os.getenv("MIN_DURATION", "1.0"))  # minimum utterance duration, seconds

# Known audio/transcript timestamp offsets (seconds) for a few misaligned subjects
OFFSET_MAP = {
    318: 34.0,
    321: 3.355,
    341: 6.07,
    362: 16.54,
}


def load_labels(csv_path: str) -> pd.DataFrame:
    """Load PHQ-8 labels, de-duplicated by participant ID."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="Participant_ID")
    return df[["Participant_ID", "PHQ8_Binary", "PHQ8_Score"]]


def pick_utterances(transcript_csv: str, speaker: str = "Participant",
                     min_duration: float = 1.0, time_offset: float = 0.0) -> pd.DataFrame:
    """Return the speaker's utterances sorted by start time, filtered to
    duration >= min_duration. time_offset is added to start/stop times to
    correct known audio/transcript misalignment."""
    df = pd.read_csv(transcript_csv)
    df = df[df["speaker"] == speaker].copy()
    if df.empty:
        return df

    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["stop_time"] = pd.to_numeric(df["stop_time"], errors="coerce")
    df = df.dropna(subset=["start_time", "stop_time"])

    if time_offset:
        df["start_time"] += time_offset
        df["stop_time"] += time_offset

    df["duration"] = df["stop_time"] - df["start_time"]
    df = df[df["duration"] >= min_duration]
    return df.sort_values("start_time", ascending=True)


def process_subject(pid: int, label: int, phq_score: int, out_dir: str,
                     min_duration: float = 1.0, time_offset: float = 0.0) -> int:
    """Segment one subject's participant turns into per-utterance WAV clips,
    each with a matching .label (binary) and .phq_label (PHQ-8 score) file.
    Returns the number of clips written."""
    audio_path = os.path.join(AUDIO_DIR, f"{pid}_AUDIO.wav")
    trans_path = os.path.join(TRANS_DIR, f"{pid}_TRANSCRIPT.csv")

    if not os.path.exists(audio_path) or not os.path.exists(trans_path):
        print(f"[Skip] audio/transcript missing for {pid}")
        return 0

    utt_df = pick_utterances(trans_path, min_duration=min_duration, time_offset=time_offset)
    if utt_df.empty:
        print(f"[Skip] no utterances >= {min_duration}s for {pid}")
        return 0

    audio = AudioSegment.from_wav(audio_path)
    count = 0
    for _, row in utt_df.iterrows():
        start_ms = int(row["start_time"] * 1000)
        stop_ms = int(row["stop_time"] * 1000)
        clip = audio[start_ms:stop_ms]
        if len(clip) < int(min_duration * 1000):
            continue

        count += 1
        base = f"{pid}_s{count}_AUDIO"
        clip.export(os.path.join(out_dir, f"{base}.wav"), format="wav")
        with open(os.path.join(out_dir, f"{base}.label"), "w", encoding="utf-8") as f:
            f.write(str(label))
        with open(os.path.join(out_dir, f"{base}.phq_label"), "w", encoding="utf-8") as f:
            f.write(str(phq_score))

    return count


def run(out_dir: str):
    """Segment every subject in LABELS_CSV, applying the OFFSET_MAP
    correction automatically for the subjects it lists. LABELS_CSV/out_dir
    correspond to a single split (train/dev/test); call this once per split."""
    os.makedirs(out_dir, exist_ok=True)
    df = load_labels(LABELS_CSV)

    total = 0
    for _, row in df.iterrows():
        if pd.isna(row["Participant_ID"]):
            continue
        pid = int(row["Participant_ID"])

        label = int(row["PHQ8_Binary"])
        phq_score = int(row["PHQ8_Score"])
        offset = OFFSET_MAP.get(pid, 0.0)

        n_clips = process_subject(pid, label, phq_score, out_dir,
                                   min_duration=MIN_DURATION, time_offset=offset)
        total += n_clips
        print(f"{pid} (label={label}, phq={phq_score}, offset={offset}s): {n_clips} clips")

    print(f"\nDone. Total clips: {total}")


if __name__ == "__main__":
    run(OUTPUT_DIR)