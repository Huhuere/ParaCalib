"""Stage 1: ALM-based utterance-level vocal captioning.

Batch audio captioning client for a vLLM-served audio-language model (ALM),
e.g. Qwen3-Omni-Captioner. Sends audio files to a vLLM OpenAI-compatible chat
completions endpoint and collects the resulting contextualized vocal
descriptions (captions), with sequential or concurrent execution.

Usage:
    1. Start a vLLM server exposing an OpenAI-compatible chat completions API.
    2. Run this script to batch-process a directory of audio files.
"""

import concurrent.futures
import json
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Union

import requests


class VLLMAPIClient:
    """Thin HTTP client for a vLLM OpenAI-compatible chat completions endpoint."""

    def __init__(self, api_url: str = "http://localhost:8901/v1/chat/completions", timeout: int = 300):
        self.api_url = api_url
        self.timeout = timeout
        self.session = requests.Session()

    def test_connection(self) -> bool:
        """Check whether the API endpoint is reachable."""
        try:
            response = self.session.get(
                self.api_url.replace('/v1/chat/completions', '/health'),
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            try:
                self.session.post(self.api_url, json={"messages": []}, timeout=5)
                return True
            except Exception:
                return False

    def inference(self, audio_path: str, temperature: float = 0,
                  top_p: float = 1.0, max_tokens: int = 16384) -> dict:
        """Run inference on a single audio file (local path or URL)."""
        if audio_path.startswith('http://') or audio_path.startswith('https://'):
            audio_url = audio_path
        else:
            audio_url = f"file://{Path(audio_path).absolute().as_posix()}"

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}}
                    ]
                }
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        start_time = time.time()
        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                output_text = result['choices'][0]['message']['content']
                return {
                    'status': 'success',
                    'output_text': output_text,
                    'inference_time': elapsed,
                    'response': result
                }
            return {
                'status': 'failed',
                'error': f"HTTP {response.status_code}: {response.text}",
                'inference_time': elapsed
            }

        except requests.exceptions.Timeout:
            return {
                'status': 'failed',
                'error': f'Request timed out (> {self.timeout}s)',
                'inference_time': self.timeout
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'inference_time': time.time() - start_time
            }


def get_audio_files(audio_dir: str) -> List[str]:
    """List all audio files in a directory."""
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f'*{ext}'))

    return [str(f) for f in sorted(audio_files)]


def save_checkpoint(results: List[dict], checkpoint_path: str):
    """Persist partial results so a run can be resumed later."""
    checkpoint_data = {
        "checkpoint": True,
        "saved_at": datetime.now().isoformat(),
        "processed_count": len(results),
        "results": results
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)


def save_results(results: List[dict], output_path: str, total_time: float):
    """Write the final results and summary statistics to disk."""
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    avg_time = total_time / len(results) if results else 0

    output_data = {
        "summary": {
            "total_files": len(results),
            "successful": successful,
            "failed": failed,
            "total_time_seconds": round(total_time, 2),
            "average_time_per_file": round(avg_time, 2),
            "timestamp": datetime.now().isoformat(),
            "model_type": "vLLM-API",
        },
        "results": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


def batch_test_api_sequential(
    audio_files: List[str],
    client: VLLMAPIClient,
    output_json: str,
    save_checkpoint_every: int = 10
):
    """Process audio files one at a time. Suited for debugging and small runs."""
    results = []
    total_files = len(audio_files)
    checkpoint_file = output_json.replace('.json', '_checkpoint.json') if save_checkpoint_every > 0 else None
    
    print(f"\nStarting sequential processing of {total_files} files")
    print("=" * 60)
    
    for idx, audio_path in enumerate(audio_files, 1):
        audio_name = Path(audio_path).name
        print(f"\n[{idx}/{total_files}] Processing: {audio_name}")
        
        result_data = client.inference(audio_path)
        
        result = {
            "index": idx,
            "audio_file": audio_path,
            "audio_name": audio_name,
            "output_text": result_data.get('output_text'),
            "inference_time_seconds": round(result_data['inference_time'], 2),
            "status": result_data['status'],
            "timestamp": datetime.now().isoformat()
        }
        
        if result_data['status'] == 'failed':
            result['error'] = result_data['error']
        
        results.append(result)
        
        if result_data['status'] == 'success':
            output_text = result_data['output_text']
            print(f"  Success, elapsed: {result_data['inference_time']:.2f}s")
            print(f"  Output: {output_text[:100]}..." if len(output_text) > 100 else f"  Output: {output_text}")
        else:
            print(f"  Failed: {result_data['error']}")
        
        if checkpoint_file and idx % save_checkpoint_every == 0:
            save_checkpoint(results, checkpoint_file)
            print(f"  Checkpoint saved ({idx}/{total_files})")
    
    return results


def batch_test_api_concurrent(
    audio_files: List[str],
    client: VLLMAPIClient,
    output_json: str,
    max_workers: int = 5,
    save_checkpoint_every: int = 10
):
    """Process audio files concurrently for higher throughput.

    Args:
        max_workers: Concurrency level (recommended 3-10).
    """
    results = []
    results_lock = Lock()
    total_files = len(audio_files)
    checkpoint_file = output_json.replace('.json', '_checkpoint.json') if save_checkpoint_every > 0 else None
    processed_count = 0
    
    print(f"\nStarting concurrent processing of {total_files} files")
    print(f"Workers: {max_workers}")
    print("=" * 60)
    
    def process_single_file(idx_and_path):
        nonlocal processed_count
        idx, audio_path = idx_and_path
        audio_name = Path(audio_path).name
        
        result_data = client.inference(audio_path)
        
        result = {
            "index": idx,
            "audio_file": audio_path,
            "audio_name": audio_name,
            "output_text": result_data.get('output_text'),
            "inference_time_seconds": round(result_data['inference_time'], 2),
            "status": result_data['status'],
            "timestamp": datetime.now().isoformat()
        }
        
        if result_data['status'] == 'failed':
            result['error'] = result_data['error']
        
        with results_lock:
            results.append(result)
            processed_count += 1
            current_count = processed_count
        
        status_label = "OK" if result_data['status'] == 'success' else "FAIL"
        print(f"[{status_label}] [{current_count}/{total_files}] {audio_name} - {result_data['inference_time']:.2f}s")
        
        if checkpoint_file and current_count % save_checkpoint_every == 0:
            with results_lock:
                save_checkpoint(results, checkpoint_file)
                print(f"Checkpoint saved ({current_count}/{total_files})")
        
        return result
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_file, (idx + 1, path))
            for idx, path in enumerate(audio_files)
        ]
        concurrent.futures.wait(futures)
    
    results.sort(key=lambda x: x['index'])
    
    return results


def batch_test_vllm_api(
    audio_source: Union[str, List[str]],
    output_json: str = "batch_results_api.json",
    api_url: str = "http://localhost:8901/v1/chat/completions",
    mode: str = "concurrent",
    max_workers: int = 5,
    save_checkpoint_every: int = 10,
    timeout: int = 300
):
    """Batch-test a vLLM-served audio-language model over its API.

    Args:
        audio_source: Directory of audio files, or an explicit list of file paths/URLs.
        output_json: Output JSON file path.
        api_url: vLLM API endpoint.
        mode: 'sequential' or 'concurrent'.
        max_workers: Concurrency level (only used when mode='concurrent').
        save_checkpoint_every: Save a checkpoint every N processed files.
        timeout: Per-request timeout in seconds.
    """
    print("=" * 60)
    print("  Batch Audio Captioning (vLLM API mode)")
    print("=" * 60)
    print(f"API endpoint: {api_url}")
    print(f"Mode: {mode}")
    if mode == "concurrent":
        print(f"Workers: {max_workers}")
    
    client = VLLMAPIClient(api_url, timeout)
    
    print("\nTesting API connection...")
    if not client.test_connection():
        print("Could not connect to the vLLM API service.")
        print("\nMake sure the vLLM server is running, e.g.:")
        print("  single GPU: vllm serve <model> --port 8901 --host 127.0.0.1 --dtype bfloat16 "
              "--max-model-len 32768 --allowed-local-media-path / -tp 1")
        print("  multi GPU:  vllm serve <model> --port 8901 --host 127.0.0.1 --dtype bfloat16 "
              "--max-model-len 32768 --allowed-local-media-path / -tp 4")
        return
    
    print("Connected.")
    
    if isinstance(audio_source, str):
        print(f"\nAudio directory: {audio_source}")
        audio_files = get_audio_files(audio_source)
    else:
        audio_files = audio_source
    
    if not audio_files:
        print("Error: no audio files found.")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    start_time = time.time()
    
    if mode == "sequential":
        results = batch_test_api_sequential(
            audio_files, client, output_json, save_checkpoint_every
        )
    elif mode == "concurrent":
        results = batch_test_api_concurrent(
            audio_files, client, output_json, max_workers, save_checkpoint_every
        )
    else:
        print(f"Error: unsupported mode '{mode}', use 'sequential' or 'concurrent'")
        return
    
    total_time = time.time() - start_time
    
    save_results(results, output_json, total_time)
    
    checkpoint_file = output_json.replace('.json', '_checkpoint.json')
    if Path(checkpoint_file).exists():
        try:
            Path(checkpoint_file).unlink()
        except Exception:
            pass
    
    print("\n" + "=" * 60)
    print("  Done")
    print("=" * 60)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per file: {total_time/len(results):.2f}s")
    print(f"Throughput: {len(results)/total_time*60:.2f} files/min")


if __name__ == '__main__':
    # Directory containing audio files to process, or use AUDIO_LIST below instead
    AUDIO_DIR = "path/to/audio_dir"

    # Alternative: an explicit list of files/URLs
    # AUDIO_LIST = [
    #     "https://example.com/audio1.mp3",
    #     "/path/to/audio2.wav",
    # ]

    OUTPUT_JSON = "batch_results_api.json"
    API_URL = "http://localhost:8901/v1/chat/completions"
    MODE = "concurrent"       # "sequential" or "concurrent" (recommended)
    MAX_WORKERS = 5           # recommended 3-10, tune based on server load
    SAVE_CHECKPOINT_EVERY = 10
    TIMEOUT = 300             # request timeout in seconds

    batch_test_vllm_api(
        AUDIO_DIR,
        OUTPUT_JSON,
        API_URL,
        MODE,
        MAX_WORKERS,
        SAVE_CHECKPOINT_EVERY,
        TIMEOUT
    )

    # Or pass a list of files/URLs directly:
    # batch_test_vllm_api(
    #     AUDIO_LIST,
    #     OUTPUT_JSON,
    #     API_URL,
    #     MODE,
    #     MAX_WORKERS,
    #     SAVE_CHECKPOINT_EVERY,
    #     TIMEOUT
    # )
