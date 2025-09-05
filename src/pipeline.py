import argparse
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from .prompts import sample_prompts
from .audio_utils import (
    ensure_dir,
    download_audio_to_wav,
    compute_audio_features,
    make_snippet,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_replicate_client():
    # Lazy import to keep import-time dependencies minimal
    import replicate

    return replicate


def run_model(
    replicate_mod: Any,
    model: str,
    prompt: str,
    retries: int = 3,
    backoff: float = 2.0,
) -> List[str]:
    """Run the Replicate model and return a list of audio URLs."""
    # Use specific model input formats for known models
    if "lucataco/ace-step" in model:
        # Map our prompt to 'tags' and reuse it as basic lyrics if none provided
        input_data = {
            "tags": prompt,
            "lyrics": prompt,
        }
        # If no version hash supplied, pin to provided version for consistency
        if ":" not in model:
            model_with_version = "lucataco/ace-step:280fc4f9ee507577f880a167f639c02622421d8fecf492454320311217b688f1"
        else:
            model_with_version = model
    elif "meta/musicgen" in model or "musicgen" in model:
        input_data = {
            "prompt": prompt,
            "model_version": "stereo-large",
            "output_format": "mp3",
            "normalization_strategy": "peak"
        }
        model_with_version = "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb"
    else:
        # Fallback for other models - try common input formats
        input_data = {"prompt": prompt}
        model_with_version = model
    
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            out = replicate_mod.run(model_with_version, input=input_data)
            
            # Handle the output based on type
            if out is None:
                continue
            if isinstance(out, str):
                return [out]
            if isinstance(out, list):
                # Filter to strings
                return [x for x in out if isinstance(x, str)]
            if isinstance(out, dict):
                # Try common keys
                for key in ["audio", "audio_url", "output", "url"]:
                    if key in out and isinstance(out[key], str):
                        return [out[key]]
                # If dict contains list under 'audio' or 'output'
                for key in ["audio", "output"]:
                    if key in out and isinstance(out[key], list):
                        return [x for x in out[key] if isinstance(x, str)]
            # Handle Replicate output objects with .url() method
            if hasattr(out, 'url') and callable(getattr(out, 'url')):
                return [out.url()]
                
        except Exception as e:  # noqa: BLE001
            last_err = e
            logging.warning("Replicate call failed (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(backoff * attempt)
            continue
    
    if last_err:
        raise last_err
    raise RuntimeError("Model did not return a usable audio URL")


def main():
    parser = argparse.ArgumentParser(description="AI & Music pipeline")
    parser.add_argument("--num-tracks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="lucataco/ace-step")
    parser.add_argument("--force", action="store_true", help="Regenerate even if CSV exists")
    parser.add_argument("--snippet-seconds", type=int, default=15)
    parser.add_argument(
        "--snippet-method",
        type=str,
        default="random",
        choices=["random", "highest_rms"],
        help="How to pick the 15s snippet",
    )
    args = parser.parse_args()

    load_dotenv()
    setup_logging()
    replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_api_token:
        raise RuntimeError("REPLICATE_API_TOKEN not set. Create .env or export the variable.")

    # Paths
    audio_dir = os.path.join("outputs", "audio")
    snippet_dir = os.path.join("outputs", "snippets")
    csv_path = os.path.join("outputs", "tracks.csv")
    ensure_dir(audio_dir)
    ensure_dir(snippet_dir)

    # Prompts
    prompts = sample_prompts(args.num_tracks, seed=args.seed)

    # Prepare rows
    rows: List[Dict[str, Any]] = []

    # Load existing CSV if present and not forcing
    existing: Optional[pd.DataFrame] = None
    if os.path.exists(csv_path) and not args.force:
        try:
            existing = pd.read_csv(csv_path)
        except Exception:
            existing = None

    replicate_mod = get_replicate_client()
    # Try to capture model version for provenance
    model_version: Optional[str] = None
    try:
        model_obj = replicate_mod.models.get(args.model)
        if getattr(model_obj, "version", None):
            model_version = str(model_obj.version)
    except Exception:
        model_version = None

    for idx, prompt in enumerate(tqdm(prompts, desc="Generating tracks")):
        song_id = str(uuid.uuid4())
        audio_filename = f"song_{idx+1:02d}.wav"
        audio_path = os.path.join(audio_dir, audio_filename)

        if existing is not None and (existing.get("audio_path") == audio_path).any():
            # Skip generation; we will still recompute features/snippet below
            gen_time = float(existing.loc[existing["audio_path"] == audio_path, "generation_time_seconds"].iloc[0])
        else:
            start = time.perf_counter()
            urls = run_model(replicate_mod, args.model, prompt)
            gen_time = time.perf_counter() - start
            if not urls:
                raise RuntimeError("No audio URLs returned by the model")
            # Save first URL
            download_audio_to_wav(urls[0], audio_path)

        # Features
        features = compute_audio_features(audio_path)

        # Snippet
        snippet_filename = f"song_{idx+1:02d}_snippet.wav"
        snippet_path = os.path.join(snippet_dir, snippet_filename)
        spath, smethod, slen = make_snippet(
            source_wav=audio_path,
            target_wav=snippet_path,
            snippet_ms=int(args.snippet_seconds * 1000),
            fade_ms=500,
            method=args.snippet_method,
            seed=args.seed + idx,
        )

        row: Dict[str, Any] = {
            "song_id": song_id,
            "prompt": prompt,
            "model": args.model,
            "model_version": model_version or "",
            "audio_path": audio_path,
            "generation_time_seconds": round(gen_time, 3),
            "snippet_path": spath,
            "snippet_method": smethod,
            "snippet_length": slen,
        }
        row.update(features)
        rows.append(row)

    df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(csv_path))
    df.to_csv(csv_path, index=False)
    print(f"Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()


