"""Compile metrics using the supplied scorer's unmodified matching functions."""
import csv
import json
from pathlib import Path
import score
from src.predict import MODELS


def main():
    gold = score.load('data/test.jsonl')
    rows, examples = [], {}
    texts = {r['id']: r['text'] for r in map(json.loads, Path('data/test.jsonl').read_text(encoding='utf-8').splitlines())}
    for key in MODELS:
        pred = score.load(f'predictions/{key}.jsonl')
        if set(gold) != set(pred):
            raise ValueError(f'{key}: IDs do not exactly match gold')
        metrics = json.loads(Path(f'predictions/{key}.metrics.json').read_text())
        row = {'model': key, 'model_name': metrics['model']['name'],
               'parameters': metrics['model']['parameters'], 'quantization': metrics['model']['quantization'],
               'runtime': metrics['model'].get('runtime', 'PyTorch / Transformers or GLiNER'),
               'phase': 'post-review extension' if key in ('qwen3b', 'qwen7b') else 'original frozen baseline'}
        for label in [*score.LABELS, 'ALL']:
            tp = fp = fn = 0
            for ident, g in gold.items():
                p = pred[ident]
                if label != 'ALL':
                    g = {e for e in g if e[1] == label}
                    p = {e for e in p if e[1] == label}
                tp += len(g & p); fp += len(p - g); fn += len(g - p)
            p, r, f = score.prf(tp, fp, fn)
            row.update({f'{label}_precision': p, f'{label}_recall': r, f'{label}_f1': f})
        row.update({k: metrics[k] for k in ['median_seconds', 'p95_seconds', 'mean_seconds', 'peak_rss_mib', 'parse_failures']})
        # AWS Lightsail Linux 32 GiB / 8 vCPU $164 monthly; 730-hour amortization.
        row['assumed_usd_hour'] = 164 / 730
        row['estimated_usd_1000'] = metrics['mean_seconds'] * 1000 / 3600 * row['assumed_usd_hour']
        rows.append(row)
        examples[key] = [{'id': i, 'text': texts[i], 'false_positives': sorted(pred[i]-g),
                          'false_negatives': sorted(g-pred[i])} for i, g in gold.items() if pred[i] != g]
    with Path('results/results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    Path('results/results.json').write_text(json.dumps(rows, indent=2))
    Path('results/error_examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf-8')
    html = '<!doctype html><meta charset="utf-8"><title>Local NER results</title><style>body{font:15px system-ui;margin:40px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}th{background:#e8eef5}</style><h1>Local NER bake-off — five models</h1><p>Measured local CPU results. F1 uses the supplied scorer. Cost extrapolates local throughput at $164/730 per hour.</p><p>The 3B/7B models are Q4_K_M with llama.cpp; the original models use float32 with PyTorch. Runtime and quantization affect latency, memory and potentially quality: this is a pipeline comparison, not a controlled parameter-count-only study. The added runs use the previously inspected test set with the original prompt and cleaner unchanged.</p><table><tr>'
    fields = ['model_name', 'quantization', 'runtime', 'phase', 'ALL_precision', 'ALL_recall', 'ALL_f1', 'PERSON_f1', 'ORG_f1', 'LOCATION_f1', 'EVENT_f1', 'median_seconds', 'p95_seconds', 'peak_rss_mib', 'estimated_usd_1000']
    titles = ['Model', 'Quantization', 'Runtime', 'Experiment phase', 'Micro precision', 'Micro recall', 'Micro F1', 'PERSON F1', 'ORG F1', 'LOCATION F1', 'EVENT F1', 'Median (s)', 'P95 (s)', 'Peak RAM (MiB)', 'USD / 1,000']
    html += ''.join(f'<th>{k}</th>' for k in titles) + '</tr>'
    for row in rows:
        html += '<tr>' + ''.join(f'<td>{row[k]:.4f}</td>' if isinstance(row[k], float) else f'<td>{row[k]}</td>' for k in fields) + '</tr>'
    Path('results/results.html').write_text(html + '</table>', encoding='utf-8')


if __name__ == '__main__':
    main()
