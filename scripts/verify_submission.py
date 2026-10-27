"""Validate real deliverables and independently invoke the original scorer."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import MODELS
from src.common import PROMPT

EXPECTED = {
    'score.py': '61d18d1ee7660625c8f0e06102ad458327f9cecef4baf45777eea87ea1c45e50',
    'data/dev.jsonl': 'e2268db9e8038956219c9df8df0d17084839c35356466f53d9bc8e77e1260aee',
    'data/test.jsonl': '4e92b1200692fda2494c8052c589d384f613483ffadb450e94f438384990397d',
}


def main():
    for name, expected in EXPECTED.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    gold = [json.loads(line) for line in (ROOT/'data/test.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(gold) == 50
    lock = json.loads((ROOT/'models.lock.json').read_text())
    prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()
    for model in MODELS:
        path = ROOT / f'predictions/{model}.jsonl'
        pred = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert [p['id'] for p in pred] == [g['id'] for g in gold]
        assert all(set(p) == {'id', 'entities'} for p in pred)
        metrics = json.loads(path.with_suffix('.metrics.json').read_text())
        assert metrics['snippets'] == 50 and len(metrics['audit']) == 50
        assert metrics['input_sha256'] == EXPECTED['data/test.jsonl']
        assert metrics['prompt_sha256'] == prompt_hash
        assert metrics['model']['revision'] == lock[model]['revision']
        assert all(a['seconds'] > 0 for a in metrics['audit'])
        result = subprocess.run([sys.executable, str(ROOT/'score.py'), str(ROOT/'data/test.jsonl'), str(path)],
                                capture_output=True, text=True, check=True)
        assert 'missing predictions: 0  unknown ids: 0' in result.stdout
        print(model, 'OK')
    manifest = ROOT/'results/extension_original_hashes.json'
    if manifest.exists():
        for name, expected in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
        print('Original experiment output hashes preserved.')
    print(f'Original files intact; all {len(MODELS)*len(gold)} test predictions and metrics verified.')


if __name__ == '__main__':
    main()
