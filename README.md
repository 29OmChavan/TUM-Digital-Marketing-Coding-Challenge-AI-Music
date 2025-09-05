# AI & Music Pipeline - TUM Digital Marketing Coding Challenge

A reproducible pipeline that generates AI music tracks, extracts audio features, and creates preview snippets using Replicate's API.

## Overview

This project implements a three-step pipeline:
1. **Generate 30 audio tracks** using AI models via Replicate
2. **Extract audio features** for each track using librosa
3. **Create 15-second snippets** with fade effects for previews

## Quick Start

### Prerequisites
- Python 3.12+ (64-bit recommended)
- FFmpeg installed and on PATH
- Replicate API token

### Setup
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set API token
echo REPLICATE_API_TOKEN=your_token_here > .env
```

### Run Pipeline
```bash
python -m src.pipeline
```

Outputs: `outputs/audio/`, `outputs/snippets/`, `outputs/tracks.csv`

## Implementation Details

### Step 1: Audio Generation
- **Model**: `lucataco/ace-step` (cost-effective for experimentation)
- **Input**: Structured prompts with tags and lyrics
- **Output**: 30 audio files saved to `outputs/audio/`
- **Metadata**: Records `song_id`, `prompt`, `model`, `audio_path`, `generation_time_seconds`

**Why this model**: Lower cost than alternatives while accepting structured musical inputs (tags + lyrics), making it suitable for budget-conscious experimentation.

**Alternative model**: `meta/musicgen` - General-purpose text-to-music with higher quality but increased cost.

### Step 2: Feature Extraction
- **Library**: librosa for audio analysis
- **Features computed**:
  - `duration_seconds` - Track length
  - `tempo_bpm` - Beat detection
  - `rms_mean` - Average loudness
  - `spectral_centroid_mean` - Brightness/timbre
  - `zero_crossing_rate_mean` - Noise/percussiveness
  - `mfcc1_mean` - First MFCC coefficient

**Why these features**: Fast, interpretable descriptors covering rhythm, loudness, timbre, and spectral characteristics essential for music analysis.

**Additional features considered**: `chroma_stft_mean` (pitch class), `spectral_bandwidth_mean` (spectral spread), `spectral_rolloff_mean` (frequency rolloff).

### Step 3: Snippet Creation
- **Default method**: Random 15-second segment extraction
- **Alternative method**: `highest_rms` - Energy-based selection of most dynamic segment
- **Effects**: 0.5s fade-in and fade-out applied via pydub
- **Output**: Preview files saved to `outputs/snippets/`

**Why random segments**: Unbiased previews that don't favor specific musical structures.

**Alternative snippet method**: Beat-aligned segments - Extract segments starting at detected beat boundaries for more musical coherence.

## Usage Options

```bash
# Generate fewer tracks (budget control)
python -m src.pipeline --num-tracks 10

# Use energy-based snippet selection
python -m src.pipeline --snippet-method highest_rms

# Switch to alternative model
python -m src.pipeline --model meta/musicgen
```

## Project Structure
```
.
├── requirements.txt
├── .env.example
├── outputs/
│   ├── audio/          # Generated tracks
│   ├── snippets/       # 15s previews
│   └── tracks.csv      # Metadata + features
└── src/
    ├── pipeline.py     # Main pipeline
    ├── prompts.py      # Curated prompt pool
    └── audio_utils.py  # Feature extraction & snippets
```

## Requirements Compliance
- ✅ Python virtual environment
- ✅ API key in `.env` (not committed)
- ✅ Complete `requirements.txt`
- ✅ Reproducible setup and execution
- ✅ Professional documentation with alternatives

## Submission Checklist
- [x] 30 audio tracks generated
- [x] Audio features extracted (6+ features)
- [x] 15s snippets with fade effects
- [x] Complete CSV metadata
- [x] Professional README with alternatives