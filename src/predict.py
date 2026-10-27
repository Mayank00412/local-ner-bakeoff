"""Run one local model. Gold annotations never reach the adapter."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import threading
import time
from pathlib import Path

from src.common import PROMPT, clean_entities, parse_output

MODELS = {
    'gliner': 'gliner-community/gliner_small-v2.5',
    'qwen05b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen15b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen3b': 'Qwen/Qwen2.5-3B-Instruct-GGUF',
    'qwen7b': 'Qwen/Qwen2.5-7B-Instruct-GGUF',
}
GGUF_MODELS = {'qwen3b', 'qwen7b'}


class MemorySampler:
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = 0
        self.stop = threading.Event()

    def sample(self):
        while not self.stop.is_set():
            processes = [self.process] + self.process.children(recursive=True)
            total = 0
            for proc in processes:
                try:
                    total += proc.memory_info().rss
                except Exception:
                    pass
            self.peak = max(self.peak, total)
            self.stop.wait(0.02)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def load_model(key, revision, threads):
    if key in GGUF_MODELS:
        from src.llama_backend import load_llama
        return load_llama(key, revision, threads)
    import torch
    torch.set_num_threads(threads)
    name = MODELS[key]
    if key == 'gliner':
        from gliner import GLiNER
        model = GLiNER.from_pretrained(name, revision=revision).to('cpu').eval()
        mapping = {'person': 'PERSON', 'organization': 'ORG', 'location': 'LOCATION', 'event': 'EVENT'}

        def predict(text):
            with torch.inference_mode():
                items = model.predict_entities(text, list(mapping), threshold=0.5)
            return [{'text': e['text'], 'label': mapping[e['label']]} for e in items], None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=torch.float32).to('cpu').eval()

        def predict(text):
            messages = [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': text}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors='pt')
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=384, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return None, raw
    return predict, {'name': name, 'revision': revision, 'parameters': sum(p.numel() for p in model.parameters()),
                     'dtype': str(next(model.parameters()).dtype), 'quantization': 'none', 'device': 'cpu'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--revision', required=True, help='Pinned Hugging Face commit SHA')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    import psutil
    rows = [json.loads(line) for line in Path(a.input).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate input IDs')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.with_suffix('.metrics.json').exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {out}')
    predict, model_info = load_model(a.model, a.revision, a.threads)
    # Synthetic warm-up separate from both evaluation splits.
    for _ in range(2):
        try:
            predict('Amira Patel visited London with Northstar Labs.')
        except ValueError:
            pass
    latencies, audit, failures = [], [], 0
    with MemorySampler() as memory, out.open('x', encoding='utf-8') as f:
        for row in rows:
            start = time.perf_counter()
            try:
                raw = None
                entities, raw = predict(row['text'])
                if entities is None:
                    entities = parse_output(raw)
                cleaned, rejected = clean_entities(entities, row['text'])
                error = None
            except ValueError as ex:
                cleaned, rejected, error = [], 0, str(ex)
                failures += 1
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            f.write(json.dumps({'id': row['id'], 'entities': cleaned}, ensure_ascii=False) + '\n')
            f.flush()
            audit.append({'id': row['id'], 'seconds': elapsed, 'raw': raw, 'parse_error': error, 'rejected': rejected})
            print(f"{a.model} {row['id']}: {elapsed:.3f}s", flush=True)
    versions = {pkg: importlib.metadata.version(pkg) for pkg in ('torch', 'transformers', 'gliner', 'psutil')}
    metrics = {'model': model_info, 'input_sha256': hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
               'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(), 'threads': a.threads,
               'snippets': len(rows), 'median_seconds': statistics.median(latencies),
               'p95_seconds': sorted(latencies)[math.ceil(.95 * len(latencies))-1],
               'mean_seconds': statistics.mean(latencies), 'peak_rss_mib': memory.peak / 2**20,
               'parse_failures': failures, 'versions': versions, 'python': platform.python_version(),
               'os': platform.platform(), 'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
               'ram_gib': psutil.virtual_memory().total / 2**30, 'audit': audit}
    out.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
