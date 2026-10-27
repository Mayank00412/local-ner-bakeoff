# Local NER bake-off — five models

Local comparison of GLiNER small v2.5 and Qwen2.5 0.5B, 1.5B, 3B, and 7B
on the supplied PERSON / ORG / LOCATION / EVENT task. No model was trained
or fine-tuned. Inference stayed local; network access was used for downloads.

Final test micro F1: GLiNER **0.900**, Qwen 0.5B **0.338**, Qwen 1.5B
**0.663**, Qwen 3B **0.638**, Qwen 7B **0.889**. GLiNER has the highest
balanced F1 and recall and lowest latency and estimated compute cost. Qwen 7B
has the highest precision (**0.978**), with 2 false positives versus GLiNER's 19.

See [results](results/results.html), [CSV](results/results.csv),
[recommendation](NOTE.md), and [full report](FULL_REPORT.html)
([Markdown](FULL_REPORT.md)). All 250 completed test predictions were verified.
The [printable PDF study guide](output/pdf/NER_Bakeoff_Interview_Study_Guide.pdf)
contains all 28 sections and 33 practice questions.
The full report is the consolidated project and interview-study guide: sections
18–28 include concepts, worked calculations, a source walkthrough, 33 practice
questions, a safe live demonstration, and a study checklist. The final ZIP also
contains `SUBMISSION_MANIFEST.json` with payload file checksums.

## Experiment history

Original models use float32 PyTorch. Added 3B/7B models use official Q4_K_M
GGUF files through llama.cpp CPU to fit this 16 GB laptop. This is a pipeline
comparison; runtime and quantization confound a parameter-count-only comparison.

The added runs are a user-requested **post-review extension** on the same test
set, after original errors were inspected. The original prompt and cleaner
were retained without tuning. These are not fresh untouched holdout results.
Original predictions, metadata, and settings remain unchanged and hash-checked;
old reports are archived in `results/original_three_model`.

The first 7B test attempt was interrupted after 13 snippets. Its partial file
is preserved in `results/interrupted_extension` and excluded from reported
metrics. The full 50-snippet restart used identical frozen settings. Completed
runs produced 300 predictions: 50 dev and 250 test; the partial attempt is extra.

## Reproduce on Windows

Use Python 3.13.5 (64 bit). This laptop's `python` PATH alias did not work,
so the first command uses the installed interpreter. From the project directory:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
$env:HF_HOME = "$PWD\.cache\huggingface"
.\.venv\Scripts\python.exe scripts/setup_larger_models.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/run_all.py
.\.venv\Scripts\python.exe scripts/verify_submission.py
```

On other machines use the installed Python path. Initial dependency ranges are
in `requirements.txt`; exact installed versions are in `requirements.lock.txt`.
Primary model revisions are pinned in `models.lock.json`. GGUF downloads total
about 6.79 GB; both 7B shards are required. The setup script verifies model
SHA-256 values and downloads the pinned Windows CPU runtime from
`runtime.lock.json`. Allow additional disk space for original models and cache.
No API keys or paid/cloud inference are required.

Completed prediction files with metrics are reused. Running the supplied project
again therefore does not repeat inference. For an intentional rerun, archive
existing prediction/results files first and disclose the rerun. Partial outputs
are not silently overwritten. Run models sequentially.

To run only the added phase with original outputs present:

```powershell
.\.venv\Scripts\python.exe scripts/run_extension.py
```

It checks original hashes, runs missing 3B/7B dev evaluations, records extension
settings before test, runs the added test evaluations, and compiles five models.
Scoring, report rebuilding, verification, and packaging:

```powershell
.\.venv\Scripts\python.exe score.py data/test.jsonl predictions/qwen7b.jsonl --errors
.\.venv\Scripts\python.exe -m src.report
.\.venv\Scripts\python.exe scripts/build_full_report.py
.\.venv\Scripts\python.exe scripts/verify_submission.py
.\scripts\package.ps1
.\.venv\Scripts\python.exe scripts/verify_archive.py
```

The prediction CLI accepts `gliner`, `qwen05b`, `qwen15b`, `qwen3b`, and `qwen7b`.
Use `python -m src.predict`, with `--model`, `--revision`, `--input`, and
`--output`. Run setup before new GGUF inference. Do not tune on test.

## Hardware, models, and settings

Windows 11; Intel i7-11800H @ 2.30 GHz; 16,905,965,568 bytes RAM (15.75 GiB).
RTX 3050 Laptop GPU (4 GB) and Intel integrated graphics were unused. All models
use CPU, four compute threads, one snippet at a time, and two synthetic warm-ups.

- GLiNER: `gliner-community/gliner_small-v2.5`, 166,023,936 parameters,
  float32; threshold 0.5; person/organization/location/event mapped to labels.
- Qwen 0.5B: `Qwen/Qwen2.5-0.5B-Instruct`, 494,032,768 parameters, float32.
- Qwen 1.5B: `Qwen/Qwen2.5-1.5B-Instruct`, 1,543,714,304 parameters, float32.
- Qwen 3B: `Qwen/Qwen2.5-3B-Instruct-GGUF`, 3,085,938,688 base parameters,
  official Q4_K_M, llama.cpp b11935 CPU x64.
- Qwen 7B: `Qwen/Qwen2.5-7B-Instruct-GGUF`, 7,615,616,512 parameters,
  official Q4_K_M, same CPU runtime.

All Qwen models use the same system prompt, Qwen ChatML, greedy decoding,
a 384-token output cap, and repetition penalty 1.1. New runs use temperature 0,
seed 0, a 2,048-token context/repetition window, no cross-request prompt cache,
and zero GPU layers. ChatML rendering was checked against the original tokenizer.
The hidden loopback-only server is a worker child and stops when the worker exits.
No demonstrations from test, fine-tuning, or output-repair model were used.

`results/frozen_settings.json` preserves the original experiment;
`results/extension_frozen_settings.json` records the later phase and prior test
exposure. Dev was a functional check, not a prompt/threshold search. Adapters
receive text only; gold is used for scoring and analysis. Metrics record model
revisions, versions, hardware, raw output, parse failures, rejected entries,
and timings. GGUF stored tensor elements are distinguished from base parameters.

## Files and validation

- `ASSIGNMENT.md`, `data/*.jsonl`, `score.py`: supplied originals, unchanged.
- `src/common.py`: shared prompt, parser, and gold-independent cleaner.
- `src/predict.py`: inference, output, timing, RAM sampling, audit metadata.
- `src/llama_backend.py`: quantized model/server adapter.
- `predictions/*.jsonl` and `*.metrics.json`: ten completed runs and audits.
- `results/*_score.txt`: original scorer output, including errors.
- `results/results.csv`, `.json`, `.html`: five-model comparison.
- `results/error_examples.json`: observed false positives and false negatives.
- `results/extension_original_hashes.json`: original-output preservation manifest.
- `results/*_llama_server_*.log`: diagnostics, including preflight issues fixed
  before evaluation records were processed.
- `NOTE.md`, `FULL_REPORT.md`, `FULL_REPORT.html`: short and full reports.

The parser accepts a uniquely identifiable JSON object, including fenced output.
Malformed JSON/schema produces an empty prediction and logged error. Cleaning
normalizes labels, maps ORGANIZATION/ORGANISATION to ORG, strips possessives,
deduplicates pairs, and rejects malformed entries, unsupported labels, and names
absent as whole spans in input. `LOC` is unsupported. Cleaning never uses gold;
it does not repair semantic labels, honorifics, or all nested predictions.
Five unit tests, supplied-file hashes, all test IDs, prompt/input hashes, model
revisions, timing coverage, original-output hashes, and scorer execution passed.

## Measurements and cost

Quality uses the original scorer's exact case-insensitive entity/label matching
and deduplication. Overall F1 is micro F1, not a mean of four label F1 scores.
All per-label precision, recall, and F1 values are in CSV/JSON and scorer logs.

Latency uses `perf_counter` around inference, parsing, and cleaning; excludes
loading, two warm-ups, and prediction-file I/O. P95 is nearest-rank ceil(0.95*n).
Memory is peak sampled worker-plus-descendant RSS at roughly 20 ms intervals.
GGUF duration includes loopback HTTP; RSS includes llama-server. It is approximate
CPU RAM, not GPU VRAM, incremental allocation, or loading peak. Test measurements
started after downloads finished; the early 3B dev check overlapped downloading.

Use the recorded Linux AWS Lightsail 32 GB / 8 vCPU public IPv4 host assumption,
US East (N. Virginia), $164/month amortized over 730 hours: $0.2246575/hour.
[Source checked 27 October 2026](https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-bundles.html).
This amortization is not a quoted hourly tariff or actual bill.

`USD per 1,000 = mean_seconds * 1000 / 3600 * (164 / 730)`.

No cloud machine was benchmarked. Estimate excludes idle time, loading, downloads,
retries, storage, taxes, management, and hardware differences. One host price
is not instance-size optimization. Fifty synthetic snippets, fixed baselines,
single completed timing runs, and post-review model additions limit conclusions.
Confirm the expanded ranking on fresh held-out data.

## Sources and AI disclosure

[GLiNER](https://huggingface.co/gliner-community/gliner_small-v2.5),
[Qwen 0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[Qwen 1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
[Qwen 3B GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF),
[Qwen 7B GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF),
[llama.cpp b11935](https://github.com/ggml-org/llama.cpp/releases/tag/b11935).

Codex generated the harness, set up the environment, executed local experiments
and validation, and drafted documentation and analysis.
