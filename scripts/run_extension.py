"""Add the 3B/7B runs without changing the original experiment outputs."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONIOENCODING'] = 'utf-8'
from src.common import PROMPT

KEYS = ('qwen3b', 'qwen7b')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = json.loads((ROOT/'models.lock.json').read_text())
    manifest = ROOT/'results/extension_original_hashes.json'
    if not manifest.exists():
        files = [p for p in (ROOT/'predictions').iterdir()
                 if p.name.startswith(('gliner', 'qwen05b', 'qwen15b'))]
        files += [ROOT/'results/frozen_settings.json']
        manifest.write_text(json.dumps({p.relative_to(ROOT).as_posix(): digest(p) for p in files}, indent=2))
    original = json.loads(manifest.read_text())
    for name, expected in original.items():
        assert digest(ROOT/name) == expected, f'Original output changed: {name}'
    frozen = json.loads((ROOT/'results/frozen_settings.json').read_text())
    assert frozen['prompt'] == PROMPT, 'Original prompt was changed'
    for split in ('dev', 'test'):
        if split == 'test':
            path = ROOT/'results/extension_frozen_settings.json'
            settings = {'models': {key: lock[key] for key in KEYS},
                'runtime': json.loads((ROOT/'runtime.lock.json').read_text()),
                'prompt': PROMPT, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                'threads': 4, 'max_new_tokens': 384, 'warmups': 2, 'temperature': 0,
                'repeat_penalty': 1.1, 'repeat_last_n': 2048, 'context_tokens': 2048,
                'cache_prompt': False, 'quantization': 'Q4_K_M', 'device': 'cpu',
                'test_texts_and_errors_previously_inspected': True,
                'phase': 'User-requested post-review extension; no prompt/cleaner tuning; not a fresh held-out model-selection experiment'}
            if path.exists():
                assert json.loads(path.read_text()) == settings
            else:
                path.write_text(json.dumps(settings, indent=2))
        for key in KEYS:
            suffix = '_dev' if split == 'dev' else ''
            target = ROOT/f'predictions/{key}{suffix}.jsonl'
            if not target.with_suffix('.metrics.json').exists():
                subprocess.run([sys.executable, '-m', 'src.predict', '--model', key,
                    '--revision', lock[key]['revision'], '--input', f'data/{split}.jsonl',
                    '--output', str(target)], cwd=ROOT, check=True)
            with (ROOT/f'results/{key}{suffix}_score.txt').open('w', encoding='utf-8') as f:
                subprocess.run([sys.executable, 'score.py', f'data/{split}.jsonl', str(target), '--errors'],
                               cwd=ROOT, stdout=f, check=True)
    for name, expected in original.items():
        assert digest(ROOT/name) == expected, f'Original output changed: {name}'
    subprocess.run([sys.executable, '-m', 'src.report'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'scripts/verify_submission.py'], cwd=ROOT, check=True)
    print('Extension complete; original predictions, metadata and settings preserved.', flush=True)


if __name__ == '__main__':
    main()
