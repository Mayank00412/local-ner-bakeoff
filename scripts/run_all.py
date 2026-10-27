"""Resolve immutable revisions, run dev, freeze settings, then run test once."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, str(ROOT))
from src.predict import MODELS
from src.common import PROMPT


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    from huggingface_hub import HfApi
    lock = ROOT / 'models.lock.json'
    if lock.exists():
        models = json.loads(lock.read_text())
    else:
        models = {key: {'name': name, 'revision': HfApi().model_info(name).sha} for key, name in MODELS.items()}
        lock.write_text(json.dumps(models, indent=2))
    for key, model in models.items():
        target = ROOT / 'predictions' / f'{key}_dev.jsonl'
        if not target.with_suffix('.metrics.json').exists():
            run('-m', 'src.predict', '--model', key, '--revision', model['revision'],
                '--input', 'data/dev.jsonl', '--output', str(target))
        with (ROOT / 'results' / f'{key}_dev_score.txt').open('w') as f:
            subprocess.run([sys.executable, 'score.py', 'data/dev.jsonl', str(target), '--errors'], cwd=ROOT, stdout=f, check=True)
    freeze = ROOT / 'results' / 'frozen_settings.json'
    if not freeze.exists():
        freeze.write_text(json.dumps({'models': models, 'threshold': 0.5, 'threads': 4,
            'max_new_tokens': 384, 'do_sample': False, 'warmups': 2,
            'prompt': PROMPT, 'test_tuned': False,
            'dev_policy': 'Fixed settings; dev is a functional check, no search.'}, indent=2))
    for key, model in models.items():
        target = ROOT / 'predictions' / f'{key}.jsonl'
        if not target.with_suffix('.metrics.json').exists():
            run('-m', 'src.predict', '--model', key, '--revision', model['revision'],
                '--input', 'data/test.jsonl', '--output', str(target))
        with (ROOT / 'results' / f'{key}_score.txt').open('w') as f:
            subprocess.run([sys.executable, 'score.py', 'data/test.jsonl', str(target), '--errors'], cwd=ROOT, stdout=f, check=True)
    run('-m', 'src.report')
    with (ROOT / 'requirements.lock.txt').open('w') as f:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)


if __name__ == '__main__':
    main()
