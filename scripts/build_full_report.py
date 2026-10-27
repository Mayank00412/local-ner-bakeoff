"""Build the comprehensive project report from the recorded experiment."""
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import PROMPT


def main():
    md, web, contents = [], [], []

    def heading(title, level=2):
        anchor = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        md.append('#' * level + ' ' + title)
        web.append(f'<h{level} id="{anchor}">{html.escape(title)}</h{level}>')
        if level == 2:
            contents.append((title, anchor))

    def para(text):
        md.append(text)
        web.append('<p>' + html.escape(text) + '</p>')

    def bullets(items):
        md.append('\n'.join('- ' + s for s in items))
        web.append('<ul>' + ''.join('<li>' + html.escape(s) + '</li>' for s in items) + '</ul>')

    def code(text, lang='text'):
        md.append(f'```{lang}\n{text}\n```')
        web.append('<pre><code>' + html.escape(text) + '</code></pre>')

    def table(headers, rows):
        # Markdown edition uses lists; HTML edition provides comparison tables.
        md.append('\n\n'.join('- ' + '; '.join(f'{h}: {v}' for h, v in zip(headers, row)) for row in rows))
        web.append('<div class="scroll"><table><thead><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>')

    results = json.loads((ROOT/'results/results.json').read_text())
    best_quality = max(results, key=lambda r: r['ALL_f1'])
    best_cost = min(results, key=lambda r: r['estimated_usd_1000'])
    locks = json.loads((ROOT/'models.lock.json').read_text())
    metrics = {p.stem.replace('.metrics', ''): json.loads(p.read_text()) for p in (ROOT/'predictions').glob('*.metrics.json')}
    heading('Local NER Bake-off — Consolidated Report and Interview Study Guide', 1)
    para('Report prepared 28 October 2026. This report explains the completed project, the work performed, every main deliverable, measured results, and the limits of the conclusions. It supplements the shorter assignment submission note; it does not replace or alter the original experiment.')
    heading('1. What the project does')
    para('The expanded project compares five pretrained models on named entity recognition (NER): GLiNER and Qwen2.5 0.5B, 1.5B, 3B, and 7B. The task is to find names in short text and assign PERSON, ORG, LOCATION, or EVENT. The aim is to compare extraction quality, speed, memory, and estimated compute cost.')
    para('This is a local Python command-line evaluation project. GLiNER and the smaller Qwen models use PyTorch; the added 3B/7B models use official Q4_K_M GGUF checkpoints with llama.cpp on CPU. No model was trained, fine-tuned, or deployed to the cloud. HTML files are static reports.')
    para(f"Across the five measured pipelines, {best_cost['model']} has the lowest estimated compute cost and {best_quality['model']} has the highest micro F1 ({best_quality['ALL_f1']:.4f}). Results apply to this dataset and these settings. Runtime and quantization differ, so this is not a parameter-count-only comparison.")
    para('Important experiment-history disclosure: the initial three-model experiment was completed and its test errors reviewed before the user requested 3B and 7B. The added models are a post-review extension on the same test set, not a fresh untouched held-out model-selection result. The original prompt and cleaner were retained without tuning. Original predictions, metadata, and settings were preserved and hash-checked; original reports are archived in results/original_three_model.')
    interrupted = ROOT/'results/interrupted_extension/README.json'
    if interrupted.exists():
        partial = json.loads(interrupted.read_text())
        para(f"Execution disclosure: the first 7B test attempt was interrupted after {partial['completed_snippets']} snippets before summary metrics were saved. Its partial prediction file is preserved in results/interrupted_extension and excluded from the reported measurements. The full 50-snippet run was restarted with identical frozen settings. The tables summarize completed runs only; this restart does not make the previously inspected test set fresh.")

    heading('2. Assignment requirements and how they were met')
    bullets([
        'Compare at least three models: the expanded comparison has GLiNER plus four Qwen2.5 sizes: 0.5B, 1.5B, 3B, and 7B.',
        'Include a specialist and a local LLM: GLiNER is the specialist; all four Qwen models are local LLMs.',
        'Use the supplied data: original dev.jsonl and test.jsonl were copied unchanged.',
        'Use the supplied scorer without changing it: original score.py was copied unchanged and run for all ten evaluations.',
        'Create a reusable harness: src/predict.py loads a selected model, predicts each snippet, validates output, writes JSONL, and records measurements.',
        'Report quality and efficiency: results include overall and per-label precision, recall, F1, median/p95 latency, peak sampled RAM, and estimated cost.',
        'Document reproducibility: README, dependency lock, exact checkpoint revisions, settings record, raw responses, and scorer outputs are included.',
        'Provide a short recommendation and AI disclosure: NOTE.md contains both; this report provides a longer explanation.',
        'Provide a submission archive: local-ner-bakeoff-submission.zip contains the deliverables without the environment or model cache.'
    ])
    para('The initial comparison used 0.5B and 1.5B through Transformers. At the user\'s request, 3B and 7B were added using the official Q4_K_M quantized checkpoints through llama.cpp b11935 CPU x64. Float32 7B weights would exceed the laptop\'s RAM, so the added runs use quantization. Ollama is not required or installed for this project.')

    heading('3. Data and annotation rules')
    para('The supplied synthetic dataset has 10 development snippets and 50 test snippets. Test contains 108 distinct gold entity/label pairs: 26 PERSON, 25 ORG, 42 LOCATION, and 15 EVENT. Five models each produced 10 dev and 50 test predictions: 300 evaluated predictions overall, including 250 test predictions. The last 100 test predictions belong to the post-review extension.')
    bullets([
        'PERSON: names of individual people; exclude job titles and honorifics.',
        'ORG: named companies, agencies, schools, institutions, clubs, teams, and media outlets.',
        'LOCATION: named countries, cities, districts, rivers, landmarks, venues, and buildings.',
        'EVENT: named festivals, tournaments, summits, holidays, storms, and similar events.',
        'Use the longest name; do not separately tag names nested inside it.',
        'Do not tag generic events, nationalities, dates, numbers, job titles, or film/book/song titles.',
        'Strip possessives; repeated mentions of the same entity/label pair count once; some snippets contain no entities.'
    ])
    para('Input JSONL records contain id, text, and gold entities. The model adapters receive only text. Gold entities are used for scoring and subsequent error analysis, never for deciding which predictions to retain.')
    code(json.dumps(json.loads((ROOT/'data/dev.jsonl').read_text(encoding='utf-8').splitlines()[0]), indent=2), 'json')
    para('Prediction records contain only id and entities. The original data and scorer were checked against SHA-256 hashes from the supplied copies; the checks passed. The original ASSIGNMENT.md is also preserved.')

    heading('4. Machine and software environment')
    bullets([
        'Operating system: Windows 11; recorded platform string Windows-11-10.0.26200-SP0.',
        'CPU: Intel Core i7-11800H @ 2.30 GHz; 16 logical CPUs reported.',
        'Installed RAM: 16,905,965,568 bytes, approximately 15.75 GiB.',
        'GPU present: NVIDIA GeForce RTX 3050 Laptop GPU, approximately 4 GB VRAM, plus Intel integrated graphics. Neither GPU was used for these model runs.',
        'Python: 3.13.5, in a project-local .venv environment.',
        'Key libraries: gliner 0.2.29, transformers 4.57.6, torch 2.14.1, psutil 7.2.2, sentencepiece 0.2.2.',
        'Common execution: CPU, four compute threads, one snippet at a time. Original models: float32 without quantization. Added 3B/7B models: mixed GGUF Q4_K_M through llama.cpp, 2,048-token context, GPU layers zero.'
    ])
    para('The initial python command on PATH pointed to a non-working Windows Store alias. We located the installed Python313 interpreter and used it to create .venv. Dependencies and model weights were downloaded; inference itself stayed local. requirements.lock.txt records the complete installed dependency versions.')

    heading('5. Models, exact sizes, and settings')
    table(['Model', 'Parameters', 'Type', 'Quantization'], [[r['model_name'], f"{r['parameters']:,}", 'Specialist NER' if r['model']=='gliner' else 'Local instruction LLM', r['quantization']] for r in results])
    para('Parameter counts use the base pretrained model weights: 3,085,938,688 for Qwen 3B and 7,615,616,512 for Qwen 7B. llama.cpp reports 3,397,103,616 stored tensor elements for the 3B GGUF because the tied embedding is stored again as an output tensor. Metrics retain that separate stored_tensor_parameters value; it is not used to inflate the model parameter count.')
    para('GLiNER uses threshold 0.5 and person/organization/location/event labels mapped to assignment labels. All four Qwen models receive the identical system prompt and use greedy decoding, at most 384 generated tokens, and Qwen ChatML formatting. Transformers uses do_sample=False. llama.cpp uses temperature=0, seed=0, repeat_penalty=1.1 over the full context, and cache_prompt=False. Runtime-level numerical and sampling implementation differences remain. The 3B/7B context is 2,048 tokens, sufficient for these short snippets and the output cap; no JSON grammar is enabled.')
    para('The exact primary model repository revisions are pinned in models.lock.json:')
    code(json.dumps(locks, indent=2), 'json')
    heading('The actual shared LLM system prompt', 3)
    code(PROMPT)
    para('The prompt describes the task but generation is not constrained by a JSON schema. That distinction matters: instructions to return JSON do not guarantee valid JSON or supported labels.')

    heading('6. What we did, in order')
    bullets([
        'Read the assignment, supplied scorer, development data, and pasted roadmap. Kept document guidance separate from the user request.',
        'Copied ASSIGNMENT.md, score.py, dev.jsonl, and test.jsonl into the project without modifying the supplied files.',
        'Checked the local Python installation, hardware, available storage, and whether Ollama was running.',
        'Created .venv, installed the dependencies, and downloaded the three model checkpoints to the local Hugging Face cache.',
        'Implemented shared parsing/validation, model adapters, JSONL output, timing, memory sampling, metadata logging, scoring/report compilation, and submission verification.',
        'Added four unit tests for the parser and validation rules; they passed.',
        'Ran all three models on the 10-snippet development set. These runs checked functionality; no prompt, threshold, or alias search was performed.',
        'Saved the fixed settings to results/frozen_settings.json before the final test runs.',
        'Ran GLiNER, Qwen 0.5B, and Qwen 1.5B sequentially on all 50 test snippets, once each.',
        'Scored predictions using the unmodified original scorer, and compiled machine-readable metrics and the HTML results table.',
        'Inspected test errors after inference completed and wrote NOTE.md. No extraction settings were changed based on those errors.',
        'Verified original-file hashes, all test prediction IDs, metrics lengths, positive timings, dependency consistency, and archive integrity; then built the ZIP.',
        'For this follow-up request, added this expanded report and its generator. This documentation work does not rerun or alter model predictions.'
        ,'Subsequently added the user-requested 3B/7B Q4_K_M extension: pinned and checksum-verified the runtime and model files, retained the prompt/cleaner, ran dev before test, and recorded extension_frozen_settings.json.'
        ,'Kept all original prediction/metrics files and the original frozen settings unchanged; preserved their SHA-256 hashes in extension_original_hashes.json and archived the previous reports.'
        ,'Resolved a runtime API validation issue during synthetic warm-up before any new evaluation records were processed: this llama.cpp build requires repeat_last_n=2048 instead of -1. Recorded the explicit full-context window. Also normalized 3B parameter-count metadata while retaining the raw GGUF tensor count; no prediction or timing was changed by that metadata correction.'
    ])
    total = sum(sum(a['seconds'] for a in m['audit']) for m in metrics.values())
    para(f'The ten measured evaluation loops totalled {total:.2f} seconds ({total/60:.2f} minutes). This is measured inference/validation time, not total project wall-clock time: downloads, installation, model loads, warm-ups, development work, and documentation are excluded.')

    heading('7. How the pipeline works')
    code('JSONL input -> id/text selection -> local model -> parse response\n-> normalize/validate entities -> prediction JSONL + per-snippet audit\n-> original score.py -> aggregate results -> recommendation/report')
    heading('Model loading and inference', 3)
    para('src/predict.py selects one of five model configurations. Original adapters use GLiNER.from_pretrained or AutoModelForCausalLM in evaluation/inference mode. The new src/llama_backend.py starts a hidden local CPU llama-server child process bound only to loopback, loads the GGUF, sends raw ChatML completion requests, and shuts the child down when the worker exits. The original memory sampler includes descendants, so it captures the llama-server rather than just the Python client. Each model runs separately, with loading and two synthetic warm-ups excluded from latency.')
    heading('Output parsing and cleaning', 3)
    bullets([
        'Try to parse the complete response as JSON first.',
        'If surrounding prose or fences are present, attempt to identify a unique decodable JSON object. Ambiguous multiple objects are rejected.',
        'Require an object containing an entities array; a parse/schema failure becomes an empty prediction and a logged error.',
        'Require each entity to contain string text and label fields; reject malformed entries.',
        'Trim text, strip trailing possessive endings, and uppercase labels.',
        'Map ORGANIZATION and ORGANISATION to ORG. LOC is not an accepted alias in this experiment.',
        'Reject unsupported labels and names that do not appear as whole spans in the input, using a case-insensitive check.',
        'Restore matched source casing and deduplicate repeated text/label pairs.',
        'Do not consult gold annotations, repair semantic label errors, remove all nested predictions, or automatically remove honorifics.'
    ])
    para('The cleaner is a rule-based validation layer, not another NER model. Its decisions are part of the evaluated pipeline. For example, an otherwise correct location returned with label LOC is rejected, which can reduce measured recall. A malformed response can lose credit for every entity in that snippet.')
    heading('Run protection and audit trail', 3)
    para('Duplicate input IDs are rejected. The prediction command refuses to overwrite an existing prediction or metrics file. The orchestration script reuses completed runs; it does not silently overwrite partial predictions. Each metrics JSON stores the model revision, parameter count, dtype, device, dependency versions, input and prompt hashes, hardware metadata, raw LLM response, parse error, rejected-entry count, and duration for each snippet.')
    para('frozen_settings.json preserves the original experiment. extension_frozen_settings.json records the added models, runtime, settings, and previous test exposure. extension_original_hashes.json checks old predictions, metrics, and settings before/after the extension. These are audit records, not a full cryptographic proof of source-code immutability; interrupted prediction loops are not automatically resumed.')

    heading('8. How quality, speed, memory, and cost were measured')
    para('Quality uses exact entity text/label matching from the supplied score.py. Text is case-insensitive after whitespace trimming; repeated entity/label pairs count once. Precision is correct predictions divided by all predictions; recall is recovered gold entities divided by all gold entities; F1 is their harmonic mean. Micro F1 pools counts across labels; it is not the arithmetic mean of the four label F1 scores.')
    code('Precision = TP / (TP + FP)\nRecall = TP / (TP + FN)\nF1 = 2 * Precision * Recall / (Precision + Recall)')
    para('Latency uses time.perf_counter around model inference, response parsing, and cleaning. Model loading, two synthetic warm-ups per model run, and prediction-file I/O are excluded. Median is the middle duration. P95 uses nearest-rank ceil(0.95*n); on 50 snippets this is the 48th sorted duration. The mean is used to estimate throughput and cost.')
    para('The 3B/7B measurements include Python-to-loopback HTTP transport to the local child server. The original models execute in-process. Final test measurements started after all checkpoint downloads finished; the early 3B dev functional check overlapped the remaining download. These distinctions are additional reasons to compare the implemented pipelines rather than infer pure model-size effects.')
    para('Memory is peak sampled resident set size (RSS) of the worker and its child processes at roughly 20 ms intervals during measured inference. It includes loaded weights and runtime memory. It is approximate CPU RAM, not GPU VRAM, incremental allocation, or the maximum memory used while loading the model.')
    para('The recorded cost assumption is a Linux AWS Lightsail 32 GB RAM / 8 vCPU public-IPv4 host at $164/month, amortized across 730 hours: $0.2246575/hour. The project documentation records the source check on 27 October 2026. No cloud host was provisioned, no cloud throughput was measured, and these figures are not actual charges.')
    code('estimated USD per 1,000 snippets\n= mean seconds per snippet * 1,000 / 3,600 * (164 / 730)')
    para('Source recorded by the project: https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-bundles.html . The estimate excludes idle time, downloads, loading, storage, taxes, management, and cloud/laptop performance differences. Using one host price is not an optimization of instance size for each model.')

    heading('9. Development results')
    dev_rows = []
    for r in results:
        key = r['model']
        line = next(s for s in (ROOT/f'results/{key}_dev_score.txt').read_text().splitlines() if s.startswith('MICRO-ALL'))
        vals = line.split()
        d = metrics[key+'_dev']
        dev_rows.append([key, vals[1], vals[2], vals[3], str(d['parse_failures'])])
    table(['Model alias', 'Precision', 'Recall', 'Micro F1', 'Parse/schema failures out of 10'], dev_rows)
    para('Development revealed GLiNER over-extraction and output-compliance problems in the smaller Qwen model. The project retained its preset baseline settings rather than optimizing them. This is important when interpreting the later model ranking: it is a comparison of the implemented baseline pipelines, not the best achievable settings for each model.')

    heading('10. Final test results')
    table(['Model', 'Precision', 'Recall', 'Micro F1', 'PERSON F1', 'ORG F1', 'LOCATION F1', 'EVENT F1'], [[r['model'], *[f'{r[k]:.4f}' for k in ['ALL_precision','ALL_recall','ALL_f1','PERSON_f1','ORG_f1','LOCATION_f1','EVENT_f1']]] for r in results])
    table(['Model', 'Median s', 'P95 s', 'Mean s', 'Peak RAM MiB', 'Estimated USD / 1,000', 'Parse/schema failures'], [[r['model'],f"{r['median_seconds']:.4f}",f"{r['p95_seconds']:.4f}",f"{r['mean_seconds']:.4f}",f"{r['peak_rss_mib']:.1f}",f"{r['estimated_usd_1000']:.4f}",str(r['parse_failures'])] for r in results])
    table(['Model', 'Correct entities TP', 'False positives FP', 'Missed entities FN', 'Measured test processing s'], [[r['model'], *next(line.split()[4:7] for line in (ROOT/f"results/{r['model']}_score.txt").read_text().splitlines() if line.startswith('MICRO-ALL')), f"{sum(a['seconds'] for a in metrics[r['model']]['audit']):.2f}"] for r in results])
    para('Test rejected-entry counts: ' + '; '.join(f"{r['model']}: {sum(a['rejected'] for a in metrics[r['model']]['audit'])}" for r in results) + '. These count entries rather than snippets and do not enumerate entries inside an unparseable response. Parse/schema failure counts are shown separately in the table.')

    heading('11. Interpretation and recommendation')
    para(f"Cost-sensitive recommendation: {best_cost['model']} at estimated ${best_cost['estimated_usd_1000']:.4f}/1,000 snippets. Overall F1 recommendation among measured configurations: {best_quality['model']} at {best_quality['ALL_f1']:.4f}. The added models were selected after the original test review and use a different runtime/precision; confirm the expanded ranking on fresh held-out data.")
    para('In the original three-model experiment GLiNER led with F1 0.9004 and recovered 104 of 108 gold entities. Its recorded results are unchanged. The five-model tables above include the later extension rather than replacing those measurements.')
    para('Qwen 7B is the closest challenger: micro F1 0.8889, precision 0.9778, and recall 0.8148. It produced only 2 false positives versus GLiNER\'s 19, but missed 20 entities versus GLiNER\'s 4. Its PERSON F1 (0.9600) and ORG F1 (0.8936) exceeded GLiNER\'s, while LOCATION F1 (0.9231) and EVENT F1 (0.6087) were lower. Prefer GLiNER for balanced micro F1, recall, and low compute cost; evaluate 7B for precision-first customers. The small overall F1 difference has not been established as statistically significant on 50 snippets.')
    para('In the original comparison, Qwen 1.5B improved substantially over Qwen 0.5B in overall F1 and produced valid output on every test snippet. However, its higher precision (0.8657 versus GLiNER\'s 0.8455) came with much lower recall (0.5370 versus 0.9630). A customer who attaches unusually high cost to false positives needs an evaluation weighted to that business objective.')
    para('The original 1.5B model did not improve every label over 0.5B: EVENT F1 was 0.2857 versus 0.3000. The added 3B model also had lower micro F1 (0.6380) than 1.5B (0.6629), although its precision was higher. Size alone did not guarantee a higher score, and the extension additionally changes quantization/runtime. GLiNER\'s EVENT F1 was lower than its LOCATION F1, so its performance is not uniform across labels either.')
    para('Qwen 0.5B\'s LOCATION F1 of zero means there were no accepted correct LOCATION predictions in the evaluated pipeline. It should not be interpreted as proof that the model cannot recognize locations: dev outputs showed the unsupported LOC label, and strict validation plus malformed JSON reduced its score. This result combines semantic ability and compliance with the required output contract.')

    heading('12. Concrete observed failures')
    bullets([
        't11: "He said that they would announce the results later this week." GLiNER tagged He as PERSON; gold contains no entities. This is generic-expression over-extraction.',
        't34: "Oakridge Primary School will host the district science fair on 12 March." GLiNER correctly found Oakridge Primary School / ORG but also tagged district science fair / EVENT. The generic fair is not a named event in the gold annotation.',
        't38: "The Great Barrier Reef suffered another bleaching event, scientists in Cairns said." GLiNER correctly found Great Barrier Reef and Cairns, but added bleaching event / EVENT and scientists / ORG. Gold contains only the two locations.',
        'Boundary mismatch: t28 contains "The Straits Orchestra". GLiNER included the initial The in its ORG prediction, whereas gold expects Straits Orchestra. Exact matching counts this as a false positive plus a false negative.',
        'Malformed smaller-LLM dev response: d01 had an incorrectly closed JSON array/object structure. The parser logged the failure and emitted an empty prediction instead of guessing a correction.',
        'Smaller-LLM dev label error: d04 returned Asia Cup as ORG and Colombo as LOC. The former is a semantic mistake; the latter is rejected by the strict supported-label policy.'
    ])
    para('These failures matter differently in production: over-extraction can create spurious records, omitted entities lose useful information, and malformed output can cause the whole snippet to lose its predictions. All observed test mismatches are preserved in results/error_examples.json and the original scorer logs.')
    para('The added 7B model illustrates missed named events: in t10, "Mary-Ann Fernandez will lead the Kochi Muziris Biennale team next year," it labelled Kochi Muziris Biennale as ORG rather than EVENT. In t12, "Typhoon Meilan made landfall near Da Nang early on Friday," it omitted Typhoon Meilan / EVENT. These observed errors help explain its lower EVENT recall despite high overall precision. No prompt or cleaning changes were made after inspecting these new errors.')

    heading('13. Everything included in the project')
    inventory = [
        ('ASSIGNMENT.md','Original assignment requirements, preserved.'),
        ('README.md','Installation, execution, measurement methodology, sources, and a concise results overview.'),
        ('NOTE.md','The approximately one-page recommendation, observed failures, next steps, limitations, and AI disclosure.'),
        ('FULL_REPORT.md / FULL_REPORT.html','This comprehensive report in readable text and browser formats.'),
        ('score.py','Original employer-supplied scoring program; unchanged.'),
        ('data/dev.jsonl / data/test.jsonl','Original 10- and 50-snippet gold datasets.'),
        ('requirements.txt','Top-level dependency ranges used for initial environment setup.'),
        ('requirements.lock.txt','Exact installed dependency versions from pip freeze.'),
        ('models.lock.json','Exact repository names and primary model commit revisions.'),
        ('.gitignore','Excludes the environment, downloaded cache, bytecode, and ZIP files from version control.'),
        ('src/__init__.py','Marks src as an importable Python package.'),
        ('src/common.py','Shared label definitions, exact LLM prompt, JSON parser, and gold-independent cleaner.'),
        ('src/predict.py','Model loading, prediction CLI, warm-up, per-snippet timing, RAM sampler, and metadata/audit output.'),
        ('src/report.py','Compiles aggregate metrics using the scorer functions and exports CSV, JSON, HTML, and error examples.'),
        ('scripts/run_all.py','Orchestrates checkpoint locking, dev runs, settings recording, sequential test runs, scoring, and result generation.'),
        ('scripts/verify_submission.py','Verifies supplied-file hashes, prediction IDs/order, metrics coverage, positive timings, and successful original-scorer execution.'),
        ('scripts/package.ps1','Creates the submission archive while excluding runtime/cache/bytecode.'),
        ('scripts/build_full_report.py','Rebuilds this report from the recorded outputs; it does not run inference.'),
        ('scripts/interview_material.py','Interview-study chapters embedded in the same consolidated report.'),
        ('scripts/verify_archive.py','Checks ZIP integrity, all payload hashes, current report inclusion, navigation, and study-question coverage.'),
        ('tests/test_validation.py','Four unit tests for fenced JSON, ambiguous/invalid JSON, cleaning without gold, and substring rejection.'),
        ('predictions/gliner_dev.jsonl / gliner.jsonl','GLiNER development and final test predictions.'),
        ('predictions/qwen05b_dev.jsonl / qwen05b.jsonl','Smaller Qwen development and final test predictions.'),
        ('predictions/qwen15b_dev.jsonl / qwen15b.jsonl','Larger Qwen development and final test predictions.'),
        ('predictions/*.metrics.json','Ten corresponding measurement and raw-response audit files.'),
        ('results/*_dev_score.txt / *_score.txt','Ten original-scorer outputs, including false positives and false negatives.'),
        ('predictions/qwen3b*.jsonl / qwen7b*.jsonl','Added 3B/7B dev and test predictions, with four matching metrics files.'),
        ('src/llama_backend.py','Local hidden CPU server lifecycle, raw Qwen ChatML rendering, and quantized-model metadata.'),
        ('scripts/setup_larger_models.py','Downloads and verifies the pinned official runtime and GGUF files; handles both 7B shards.'),
        ('scripts/run_extension.py','Runs only the added two models and verifies the original output hashes.'),
        ('runtime.lock.json','Pinned llama.cpp release/download/archive SHA-256 and local executable location.'),
        ('results/extension_frozen_settings.json / extension_original_hashes.json','Extension settings/disclosure and preservation checks.'),
        ('results/original_three_model','Archived original result tables, note, frozen settings, and full reports.'),
        ('results/interrupted_extension','Preserved interrupted 13-snippet 7B attempt and exclusion/restart disclosure; excluded from final scoring and timing summaries.'),
        ('results/*_llama_server_*.log','Local server initialization and inference diagnostic logs for the added runs.'),
        ('tests/test_llama_backend.py','Verifies the shared system prompt and Qwen ChatML delimiters.'),
        ('results/frozen_settings.json','Settings record saved before final test evaluation.'),
        ('results/results.csv / results.json / results.html','Aggregate model comparison in spreadsheet-friendly, machine-readable, and browser formats.'),
        ('results/error_examples.json','Test snippets with observed false-positive and false-negative entity pairs.'),
        ('local-ner-bakeoff-submission.zip','The distributable project submission archive.')
    ]
    bullets([f'{name}: {desc}' for name, desc in inventory])
    para('The local working directory also contains .venv and .cache/huggingface for execution. They are deliberately excluded from the archive. A recipient installs the locked dependencies and downloads the pinned models. Python bytecode and the Git repository metadata are not included. The work was not pushed to a remote repository, emailed, or deployed.')

    heading('14. Commands to use and verify the project')
    para('Run these from the project directory in PowerShell. The first command uses the installed interpreter because this laptop\'s python PATH alias did not work. On another computer, use that machine\'s Python 3.13 installation.')
    code('& "$env:LOCALAPPDATA\\Programs\\Python\\Python313\\python.exe" -m venv .venv\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements.lock.txt\n$env:HF_HOME = "$PWD\\.cache\\huggingface"\n.\\.venv\\Scripts\\python.exe scripts/setup_larger_models.py\n.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v\n.\\.venv\\Scripts\\python.exe scripts/run_all.py\n.\\.venv\\Scripts\\python.exe scripts/verify_submission.py', 'powershell')
    para('To reproduce just the added-model phase with original predictions present, use scripts/run_extension.py after setup_larger_models.py. It preserves old outputs and records the post-review extension disclosure. Downloaded runtime binaries and model weights remain outside the ZIP in .cache.')
    para('On a completed project, run_all.py reuses predictions that have metrics files. It can rescore and rebuild reports without reproducing inference. To intentionally rerun inference, preserve the existing predictions/results elsewhere first, keep the pinned revisions, and disclose the new run. Never silently replace the clean test results while experimenting with changed settings.')
    code('.\\.venv\\Scripts\\python.exe score.py data/test.jsonl predictions/gliner.jsonl --errors\n.\\.venv\\Scripts\\python.exe -m src.report\n.\\.venv\\Scripts\\python.exe scripts/build_full_report.py\n.\\scripts\\package.ps1', 'powershell')

    heading('15. Validation completed and what it establishes')
    bullets([
        'Five parser/cleaner/prompt unit tests passed.',
        'pip check reported no broken dependency requirements.',
        'The supplied scorer and both datasets matched their recorded original SHA-256 hashes.',
        'All 250 test prediction records matched the 50 expected IDs and order for each model, with no missing or unknown IDs.',
        'Each test run had 50 timing audit records and positive measured durations.',
        'The original scorer executed successfully for all five final prediction files.',
        'The initial submission ZIP passed zipfile integrity testing and excluded .venv, .cache, and __pycache__.'
    ])
    para('These checks establish structural completeness, preservation of the supplied evaluation files, successful local execution, and consistency of the recorded runs. They do not prove that every prediction is correct, that all future model outputs will parse, or that the project is production-ready. In particular, the verifier does not independently recompute every recorded timing or enforce source-code immutability.')

    heading('16. Limitations and next work')
    bullets([
        'Only 50 synthetic test snippets; no production-domain validation or statistical confidence intervals.',
        'The 3B/7B extension uses a previously inspected test set; it is exploratory rather than a fresh model-selection holdout.',
        'The extension changes runtime and quantization: comparisons confound parameter count with implementation and precision.',
        'Only one final sequential timing run per model; results may depend on CPU load, thermal state, and text/output lengths.',
        'No prompt/threshold optimization; conclusions concern these baselines rather than the models\' maximum capability.',
        'Strict LOC rejection and malformed JSON materially hurt the smaller LLM; investigate schema-constrained generation and a documented alias policy on dev.',
        'The cleaner is not a complete semantic validator: pronouns, generic terms, titles, wrong labels, and some nested predictions can still survive.',
        'The JSON recovery procedure is practical rather than a proof of unique intended structure; more adversarial parser tests would improve confidence.',
        'RSS sampling can miss brief peaks, counts shared resident pages approximately, and excludes model-loading peaks.',
        'Cloud cost is an extrapolation using a recorded price assumption, not actual cloud performance or billing.',
        'Primary checkpoints and package versions are recorded, but full operating-system, driver, and transitive model-resource freezing is not a hermetic environment.',
        'There is no training pipeline, service deployment, concurrency/load benchmark, monitoring, or production recovery mechanism.'
    ])
    para('With another day, prioritize a small dev-only GLiNER threshold/label-description study, JSON-schema output and LOC normalization for LLMs, fresh held-out data with more EVENT and negative examples, repeat-run latency measurements, and a quantized larger model on the available GPU. Any inference changes after viewing this test set should be evaluated on a fresh held-out set or clearly labelled exploratory.')

    heading('17. How to explain the project in a follow-up discussion')
    para('A concise explanation: "I built a local evaluation harness comparing a specialist NER model and four Qwen sizes. I preserved the supplied data and scorer, checked functionality on dev, and measured quality, latency, RAM, and assumed compute cost. The initial three models used float32 PyTorch; the later user-requested 3B/7B extension used Q4_K_M llama.cpp because of laptop memory limits. I kept the original prompt and validator, preserved the earlier outputs, and disclosed that the test set had already been reviewed. The expanded ranking needs confirmation on fresh data."')
    bullets([
        'Be able to explain why pretrained inference is different from training a model.',
        'Explain why the test split cannot be used to choose prompts or thresholds.',
        'Explain micro F1, precision/recall trade-offs, exact matching, and the effect of malformed responses.',
        'Explain why mean latency determines throughput/cost while median and p95 describe typical and tail experience.',
        'Explain why these CPU results do not establish GPU or cloud performance.',
        'Explain that output cleaning uses input text only and never the gold entity list.',
        'Acknowledge the strict LOC policy and limited optimization rather than claiming the LLMs cannot perform NER.'
    ])
    para('AI disclosure: Codex generated the harness, set up the environment, executed the local experiments and validation, and drafted the documentation and analysis. The user should review the submission and understand the implementation before presenting it as their work.')

    from scripts.interview_material import append_interview
    append_interview(heading, para, bullets, code, table, results, ROOT)
    nav = '<nav aria-label="Contents"><h2>Contents</h2><ol>' + ''.join(f'<li><a href="#{anchor}">{html.escape(title)}</a></li>' for title, anchor in contents) + '</ol></nav>'
    web.insert(2, nav)
    md.insert(2, '## Contents\n\n' + '\n'.join(f'- [{title}](#{anchor})' for title, anchor in contents))
    (ROOT/'FULL_REPORT.md').write_text('\n\n'.join(md) + '\n', encoding='utf-8')
    css = 'body{font:16px/1.65 system-ui,sans-serif;color:#202a35;max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:30px}h2{margin-top:38px;border-bottom:1px solid #d8dee6;padding-bottom:8px}h3{font-size:18px}pre{background:#f3f5f7;padding:18px;overflow:auto;font-size:13px;line-height:1.5}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d8dee6;padding:9px;text-align:left}th{background:#edf1f5}.scroll{overflow-x:auto;margin:24px 0}li{margin:7px 0}@media print{body{font-size:11px;max-width:none;margin:0}h2{break-after:avoid}pre{white-space:pre-wrap}table{font-size:10px}}'
    (ROOT/'FULL_REPORT.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full Local NER Project Report</title><style>' + css + '</style></head><body>' + '\n'.join(web) + '</body></html>', encoding='utf-8')
    print('Created FULL_REPORT.md and FULL_REPORT.html from recorded results.')


if __name__ == '__main__':
    main()
