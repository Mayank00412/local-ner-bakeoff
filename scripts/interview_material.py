"""Interview-study chapters embedded in the single consolidated report."""
import inspect


def append_interview(heading, para, bullets, code, table, results, root):
    rows = {r['model']: r for r in results}
    g, q = rows['gliner'], rows['qwen7b']

    heading('18. Study route and the facts to remember')
    para('This is the interview-study part of the same report. Sections 1–17 contain the project evidence, implementation, results, and history. Sections 18–28 help you explain and defend that work. Suggested answers are practice material, not evidence that you personally performed every step. Review the actual source and preserve the AI disclosure.')
    bullets([
        'First reading, 20 minutes: read the objective, final tables, recommendations, limitations, and the one-minute explanation below.',
        'Second reading, 45 minutes: trace a single snippet through the pipeline; work through the precision/recall calculations; read the actual parser and cleaner.',
        'Third reading, 30 minutes: answer the questions without looking, then check the evidence files and correct your explanation.',
        'Final rehearsal, 20 minutes: give the five-minute walkthrough, run the safe demonstration commands, and explain three limitations without prompting.'
    ])
    table(['Fact', 'What to remember'], [
        ['Task', 'Exact entity extraction for PERSON, ORG, LOCATION, EVENT'],
        ['Data', '10 dev snippets; 50 test snippets; 108 gold test entity/label pairs'],
        ['Models', 'GLiNER; Qwen2.5 0.5B, 1.5B, 3B, 7B'],
        ['Completed runs', '10 runs; 300 predictions total; 250 test predictions'],
        ['Interrupted attempt', '13 partial 7B predictions preserved, excluded; full restart disclosed'],
        ['Best micro F1', 'GLiNER 0.9004; 7B 0.8889 is close'],
        ['Best precision', '7B 0.9778: 88 correct out of 90 predictions'],
        ['Best recall', 'GLiNER 0.9630: 104 recovered out of 108 gold entities'],
        ['Original precision/runtime', 'Float32 PyTorch; CPU'],
        ['Added precision/runtime', '3B/7B Q4_K_M GGUF; llama.cpp CPU'],
        ['Most important caveat', 'Added models used a previously inspected test set; validate on fresh data']
    ])

    heading('19. Your opening explanation and presentation plan')
    heading('A one-minute explanation to practice', 3)
    para('“This project compares a specialist NER model with four sizes of Qwen on a local four-label extraction task. The supplied data and scorer were preserved. A shared harness runs the models, validates outputs without gold annotations, and records quality, latency, RAM, and a compute-cost estimate. GLiNER led balanced micro F1 at 0.900 and was cheapest and fastest. Qwen 7B was close at 0.889 and had much higher precision, but lower recall. The original models used float32 PyTorch; 3B and 7B were later added as quantized llama.cpp pipelines because of laptop memory limits. I would confirm that expanded ranking on fresh data because the original test errors had already been reviewed. Codex generated the implementation and ran the experiments; I am studying the code and evidence so I can explain the submission accurately.”')
    heading('A five-minute walkthrough', 3)
    bullets([
        'Minute 1 — problem: explain the customer question, four labels, exact matching, dev/test roles, and why inference was local.',
        'Minute 2 — design: show common.py, predict.py, and llama_backend.py. Explain text-only adapters, parsing, validation, separate workers, and the audit files.',
        'Minute 3 — experiment: explain fixed baseline settings, CPU hardware, warm-ups, timing, memory sampling, original versus extension phases, and checkpoint pins.',
        'Minute 4 — results: compare GLiNER and 7B using TP/FP/FN counts, then explain the 0.5B formatting failures and the 3B runtime trade-off.',
        'Minute 5 — recommendation and trust: state a business-dependent choice, two concrete errors, the small synthetic dataset, quantization/runtime confounding, and the next experiment.'
    ])
    para('For a 20–30 minute interview, use this opening for about five minutes, spend ten minutes on source and measurement choices, and leave time for error analysis and limitations. The aim is to demonstrate understanding; do not memorize every decimal or present a script you cannot explain.')

    heading('20. Technical concepts explained in plain language')
    concepts = [
        ('Named entity recognition', 'Finding names in text and assigning a type. This assignment evaluates a set of text/label pairs, rather than token-level BIO tags or character offsets.'),
        ('Pretrained inference versus training', 'Inference applies downloaded weights to new inputs. Training changes weights using an optimization objective and examples. This project performed inference only; there was no optimizer, training loop, or fine-tuning dataset.'),
        ('Specialist model', 'A model intended for a narrower task. Here GLiNER accepts a set of entity label names and returns extracted spans. Its usefulness must still be established by measurement, not assumed from its category.'),
        ('Instruction LLM', 'A language model adapted to follow instructions. Qwen generates a sequence of output tokens, including the requested JSON. Generating correct JSON and identifying correct entities are separate requirements.'),
        ('Zero-shot usage', 'The model is asked to perform the task without assignment-specific weight training or demonstrations in the prompt. It does not mean the model had no prior training on language or entity-related tasks.'),
        ('Token and tokenizer', 'A token is a unit of model text representation; it is not necessarily one word. The tokenizer converts text into token IDs. Input prompt length and generated output length both affect LLM work.'),
        ('Chat template / ChatML', 'The formatting that marks system, user, and assistant messages. It is part of the actual model input. Keeping the words of a prompt the same is insufficient if role formatting changes.'),
        ('Autoregressive decoding', 'Generating the next token from the preceding context, then repeating. It helps explain why an LLM can spend substantial time generating repeated JSON keys, while a specialist returns entity predictions directly.'),
        ('Greedy decoding', 'Choosing the highest-scoring next token after configured processing such as the repetition penalty. Transformers uses do_sample=False; the new backend uses temperature=0. This reduces sampling variability but is not a universal guarantee of bitwise reproducibility across hardware.'),
        ('Repetition penalty', 'A modification to token scores intended to discourage repetition. The original generation configuration uses 1.1; the larger-model backend explicitly records 1.1 over its full context window. This is distinct from deduplicating entity pairs after generation.'),
        ('Context window versus output cap', 'The context window limits the token history the model can use; the output cap limits how many new tokens it may generate. The new context is 2,048 tokens and the output cap is 384, not 384 input words.'),
        ('Float32', 'A 32-bit floating-point representation. Weight storage is roughly four bytes per scalar before accounting for buffers, framework overhead, temporary allocations, and memory sharing.'),
        ('Quantization', 'Representing weights more compactly, potentially changing both memory use and numerical behavior. Q4_K_M is a mixed GGUF quantization choice; not every stored value is exactly four bits. Smaller files do not imply identical model quality.'),
        ('GGUF and llama.cpp', 'GGUF is a model file format containing tensors and metadata. llama.cpp is the inference runtime used to execute the larger quantized models. GGUF is not a different NER label scheme or a training method.'),
        ('RSS and VRAM', 'RSS is resident physical RAM associated with a process. VRAM is GPU memory. The study used CPU, so its reported memory is sampled process-tree RSS, not a GPU benchmark.'),
        ('Warm-up and throughput', 'Warm-up runs happen before measurement to reduce first-request effects. Throughput is completed work per unit time; for this sequential experiment it is derived from mean per-snippet duration.'),
        ('Reproducibility and validity', 'Reproducibility asks whether another person can repeat the procedure. Validity asks whether the procedure answers the intended question. Model pins help reproducibility, but they do not solve a tiny test set or runtime confounding.')
    ]
    for name, explanation in concepts:
        heading(name, 3)
        para(explanation)

    heading('21. Worked scoring examples you should calculate yourself')
    para('Use actual counts, not the rounded displayed percentages. Let G be the gold set of entity/label pairs and P the prediction set. TP=|G intersection P|, FP=|P minus G|, FN=|G minus P|. The scorer ignores repeated identical pairs within a snippet.')
    code('Precision = TP / (TP + FP)\nRecall = TP / (TP + FN)\nMicro F1 = 2*TP / (2*TP + FP + FN)')
    code('GLiNER: TP=104, FP=19, FN=4\nPrecision = 104 / 123 = 0.845528...\nRecall = 104 / 108 = 0.962963...\nF1 = 208 / 231 = 0.900433...\n\nQwen 7B: TP=88, FP=2, FN=20\nPrecision = 88 / 90 = 0.977778...\nRecall = 88 / 108 = 0.814815...\nF1 = 176 / 198 = 0.888889...')
    para('A correct text with the wrong label is not a partial hit: it contributes one false positive for the predicted pair and one false negative for the gold pair. In t10, 7B predicted Kochi Muziris Biennale / ORG while gold expects the same text / EVENT. Exact matching also treats The Straits Orchestra and Straits Orchestra as different names, as observed in GLiNER t28.')
    para('A case-only difference can still match. A repeated mention does not create a second gold pair. A no-entity snippet can produce false positives if the model tags generic words. The scorer does not enumerate true negatives or report a conventional accuracy over all possible non-entities, so do not call F1 “accuracy.”')
    para('Macro F1 would give each label equal weight by averaging label F1s. The assignment uses micro F1, pooling entity counts. LOCATION has 42 gold pairs while EVENT has 15, so strong LOCATION behavior can have more influence on the micro result. Always inspect the label-specific scores alongside the overall result.')
    para('One more recovered entity can matter on a small dataset, but the change in F1 depends on TP, FP, and FN together. The 0.0115 absolute F1 gap between GLiNER and 7B is about 1.15 percentage points, not proof of a statistically reliable advantage and not 1.15% greater accuracy.')

    heading('22. Turn metrics into a customer decision')
    para('Do not answer “which model is best?” without specifying the objective. For equal-weight entity F1 and these measured costs, GLiNER is the sensible default. If false positives trigger expensive manual reviews or incorrect records, 7B may be preferable. If missed entities are costly, GLiNER\'s recall advantage matters more. Validate any business-specific decision on representative data.')
    para('Illustrative business loss can be defined as C_FP*FP + C_FN*FN. These are hypothetical penalty units, not measured customer costs or cloud charges. The following calculations use the observed counts and ignore runtime, label-specific penalties, and downstream dependencies.')
    table(['Hypothetical penalties', 'GLiNER loss', '7B loss', 'Lower loss'], [
        ['FP=1; FN=1', '19+4 = 23', '2+20 = 22', '7B'],
        ['FP=10; FN=1', '190+4 = 194', '20+20 = 40', '7B'],
        ['FP=1; FN=10', '19+40 = 59', '2+200 = 202', 'GLiNER']
    ])
    para('Notice that 7B has one fewer total FP+FN at equal penalties, even though GLiNER has higher micro F1. F1 is not the same objective as a linear business loss. From these counts, 7B has lower linear loss when 17*C_FP > 16*C_FN, before compute cost or other factors. This is a worked illustration, not a production decision rule fitted to new customer data.')
    para(f"Using mean latency, GLiNER processes approximately {3600/g['mean_seconds']:,.0f} snippets/hour and 7B approximately {3600/q['mean_seconds']:,.0f} snippets/hour in this local sequential setup. The mean-duration ratio is about {q['mean_seconds']/g['mean_seconds']:.1f}x. That does not predict parallel service throughput or GPU performance.")
    para(f"Worked GLiNER cost: {g['mean_seconds']:.6f} seconds/snippet × 1,000 ÷ 3,600 × ($164 ÷ 730) = ${g['estimated_usd_1000']:.6f}/1,000. For 7B the same formula gives ${q['estimated_usd_1000']:.6f}/1,000. These are throughput-based extrapolations using a recorded host-price assumption, not actual expenses.")
    para('Why not use median to calculate cost? In a sequential batch, total time is the sum of all request durations. Mean multiplied by count reproduces that total; median generally does not. Median and p95 describe the distribution of user experience, while the mean is appropriate for this throughput calculation.')

    heading('23. Explain the code from input to output')
    heading('Step A — registry and orchestration', 3)
    para('MODELS in src/predict.py maps short aliases to repository names. The orchestrator launches a separate Python worker for a model/split. Checkpoint commits come from models.lock.json. run_extension.py preserves old file hashes, runs new dev checks before test, and records the separate phase. It is intentionally sequential to avoid loading multiple models together.')
    heading('Step B — reading and overwrite protection', 3)
    para('The worker parses JSONL into rows and rejects duplicate input IDs. It checks whether the target prediction or metrics file already exists. A refusal to overwrite protects recorded results; it is not an automatic resume feature. The gold entities remain in the parsed input rows but only row[text] is sent to predict. Do not claim the file contains no gold; explain that the adapter never receives it.')
    heading('Step C — loading and warm-up', 3)
    para('GLiNER and small Qwen models load through their Python libraries. GGUF models start a hidden local llama-server on a loopback port and wait for its health endpoint. The server loads local files; snippets are not sent to an external inference provider. Each worker performs two synthetic warm-ups before starting the measured loop.')
    heading('Step D — model call and extraction', 3)
    para('GLiNER returns entity dictionaries, mapped to assignment labels. Qwen returns decoded text. Only the newly generated tokens are decoded for the PyTorch models, excluding the prompt. llama.cpp receives the same Qwen ChatML string and returns completion content. Both LLM paths feed the same JSON parser and cleaner.')
    heading('Step E — JSON and entity validation', 3)
    para('parse_output first attempts complete JSON decoding, then attempts unique-object recovery if needed. It requires an entities list. clean_entities checks types, normalizes labels/text, strips possessive suffixes, verifies source-text spans, and deduplicates. These checks prevent some malformed or invented entries from entering the prediction file; they do not verify that a span is semantically a named entity.')
    heading('Step F — logging and measurement', 3)
    para('The timer includes the adapter call, JSON parsing, and cleaning. The memory thread samples worker-plus-descendant RSS. Predictions are written and flushed per snippet; raw response, duration, parse error, and rejected count are accumulated for the final metrics file. Parse/schema ValueErrors yield empty entities; unrelated runtime failures generally abort rather than silently become successful predictions. The ValueError counter is not a fully typed error taxonomy.')
    heading('Step G — scoring and reporting', 3)
    para('The original score.py is invoked with --errors. src/report.py also uses its normalization/scoring functions to compile higher-precision aggregate numbers and per-label metrics. CSV/JSON are machine-readable; HTML is the comparison view. build_full_report.py combines this evidence with the explanations and these study chapters; it does not run the models again.')
    heading('Step H — verification and packaging', 3)
    para('The verifier checks original file hashes, IDs and order, timing coverage, input/prompt hashes, model revisions, scorer execution, and preserved original experiment hashes. The ZIP contains code, data, results, reports, locks, and provenance, while excluding large downloaded weights, the virtual environment, and bytecode. It is not a container image or a hermetically sealed runtime.')

    heading('24. Likely interview questions with practice answers')
    qa = [
        ('Why did you compare a specialist with LLMs?', 'The assignment asks whether a smaller task-specific approach can replace more general models. I evaluated quality together with speed, RAM, and an explicit cost assumption. The specialist won the measured balanced F1/efficiency comparison; that conclusion comes from the results, not from assuming specialists are always better.'),
        ('Why did you end up with five models?', 'The initial study used GLiNER and 0.5B/1.5B Qwen. The user later requested 3B/7B. I preserved the first outputs and added a separately disclosed phase. The assignment requires at least three, so five satisfies the model-count requirement but also introduces a later model-selection history to disclose.'),
        ('Did you train any of the models?', 'No. These are pretrained checkpoints used for local inference. The project has no optimizer, loss-driven parameter updates, or fine-tuning loop. Changing a prompt, threshold, decoding option, or parser is pipeline configuration, not weight training.'),
        ('Why use CPU when the laptop has a GPU?', 'The benchmark deliberately used CPU for a simple, consistent local execution environment. The GPU has approximately 4 GB VRAM and was unused. I cannot claim GPU performance from these measurements. A quantized GPU experiment would be a useful separate next-day study.'),
        ('Why quantize only the larger models?', 'The 7B base weights at float32 require about 30.46 GB decimal, or 28.37 GiB, before overhead—more than 15.75 GiB installed RAM. Quantized checkpoints make local inference practical. This choice means the comparison is between runnable pipelines, not a clean causal experiment isolating parameter count.'),
        ('Why is quantized 3B faster than float32 1.5B?', 'Parameter count is not the only determinant of latency. The added model changes weight precision, runtime kernels, memory traffic, and process architecture. The observed speed is real for this implementation, but I cannot attribute it solely to quantization or infer a universal model-size speed rule without controlled runs.'),
        ('Why did 3B score below 1.5B?', 'The observed F1s are 0.6380 and 0.6629. Size does not guarantee better instruction following or exact extraction on every dataset. Quantization and runtime also differ, so this experiment cannot identify the cause. I would compare equivalent precisions/backends on fresh data rather than invent an explanation.'),
        ('What does micro F1 mean here?', 'Pool TP, FP, and FN across all labels and snippets, then compute 2TP/(2TP+FP+FN). GLiNER has 104 TP, 19 FP, and 4 FN, giving 208/231, approximately 0.9004. It is not a simple average of label F1s and is not token accuracy.'),
        ('Why might the highest F1 model not be best for a customer?', 'Different errors have different costs. GLiNER recovers more entities, while 7B produces far fewer false positives. A precision-first workflow may prefer 7B despite lower F1 and higher compute cost. I would establish customer error costs and label requirements, then validate on representative data.'),
        ('Can you claim GLiNER is significantly better than 7B?', 'No. The observed micro F1 gap is about 0.0115 on only 50 synthetic snippets, with no confidence interval or significance analysis. The expanded phase also uses a previously reviewed test set. I can report the point estimates and recommend validation, not statistical certainty.'),
        ('Did you tune on test?', 'The original pipeline settings were fixed before its test runs. Test errors were later inspected for analysis. The 3B/7B models were then added at the user\'s request, keeping the prompt and cleaner unchanged. I explicitly label those results post-review/exploratory; I do not describe them as a fresh untouched holdout.'),
        ('What was the development split actually used for?', 'It checked that models loaded, outputs were usable, and the harness/scorer worked. No prompt or threshold search was performed. Therefore the results are fixed baselines rather than optimized configurations, which is a limitation I disclose.'),
        ('Why is the 0.5B LOCATION F1 zero?', 'The evaluated pipeline had no accepted correct LOCATION predictions. The dev responses showed LOC, which the cleaner rejects because it is outside the accepted labels/aliases. Nine test responses were also malformed. That does not establish that the model has no ability to recognize locations.'),
        ('Why not fix LOC before test?', 'That would have been a reasonable documented dev-only improvement, but this implementation retained a preset conservative label policy. I should acknowledge that the choice depresses the smaller LLM baseline, rather than defend it as optimal. A revised mapping should be tested on dev and evaluated on fresh holdout data.'),
        ('Does output cleaning leak gold information?', 'No. It checks only the input text, types, labels, suffixes, and duplicates. Gold annotations are used by the scorer and later analysis. Input text matching can remove invented strings without looking at gold, but it cannot decide whether a real span has the correct semantic category.'),
        ('What happens to invalid JSON?', 'If a unique valid object cannot be identified or the entities schema is invalid, the worker records an error and emits an empty list for that snippet. This can create multiple false negatives. Raw responses are retained, so formatting failures are visible rather than silently hidden.'),
        ('Does your parser guarantee robust recovery?', 'No. It handles common surrounding prose/fences and rejects ambiguous multiple decoded objects, but it is not a formal proof of the model\'s intended structure. An inner object in a malformed outer structure is a possible edge case. More adversarial tests or constrained generation would improve it.'),
        ('Why require predictions to appear in the source?', 'NER here is extraction, not paraphrasing. This rejects hallucinated strings without needing gold. It can also reject alternative spellings or transformed text, and a whole-span match still does not establish namedness. The same policy is part of the measured pipeline for every model.'),
        ('Why preserve the original scorer?', 'The assignment specifies the evaluation contract. Changing it would make reported numbers less comparable. If I want a relaxed or boundary-tolerant metric, I would report it separately as an additional analysis while keeping the mandated exact-match score.'),
        ('What if a prediction has the correct name but wrong label?', 'It is a false-positive predicted pair and a false-negative gold pair. For example, EVENT versus ORG for Kochi Muziris Biennale does not match. This is why a model can appear semantically close yet lose exact-match credit.'),
        ('How did you measure latency fairly?', 'Each model loaded once per run, used two synthetic warm-ups, then processed snippets sequentially. The timer covers inference plus parsing/cleaning and excludes file output and loading. CPU thread count is recorded. Backend differences, background load, and a single timing run remain limitations.'),
        ('How is p95 calculated with 50 snippets?', 'Sort all 50 durations and take nearest rank ceil(0.95*50)=48, meaning zero-based index 47. It is an empirical tail summary from a small sample, not a guarantee that 95% of future customer requests will be below it.'),
        ('Why is mean used for cost, not median?', 'Mean times count equals measured sequential processing time. Median is not additive and can underrepresent slow requests. Cost per 1,000 extrapolates that mean duration at the assumed hourly host price; p95 and median are separate distribution summaries.'),
        ('Did memory measurement include the local server?', 'Yes. MemorySampler sums RSS for the Python worker and its descendants. llama-server is a child, so its working set is included. Sampling can miss brief peaks and may double-count shared pages; loading peaks and GPU memory are not measured.'),
        ('Are the cost estimates real cloud costs?', 'No. They use local throughput and a recorded $164/month host-price assumption amortized over 730 hours. No cloud instance ran. Different processors, idle time, storage, startup, retries, and operations can change actual costs. The same instance price also does not optimize host size per model.'),
        ('What happened to the interrupted 7B attempt?', 'It stopped after 13 predictions before final metadata was written. The partial file and disclosure are preserved, excluded from the summary, and the complete run was restarted with identical settings. The saved 13 entity predictions matched the completed restart, but this is not a full latency reproducibility study.'),
        ('How do you prevent accidental overwriting?', 'The worker refuses an existing prediction or metrics path, while orchestration reuses completed runs. This helps preserve evidence. It does not automatically detect every source edit or resume a partial run; the interruption required explicit preservation and restart.'),
        ('Why record a commit SHA instead of just a model name?', 'A repository name or mutable branch can point to different files later. Primary checkpoint revisions, GGUF file hashes, dependency versions, and runtime archive hashes identify what was used. They improve reproducibility but do not freeze the entire OS or every transitive model resource.'),
        ('Why do the 3B parameter counts differ between metadata fields?', 'The canonical base-model count is 3,085,938,688. The GGUF runtime reports 3,397,103,616 stored tensor elements, a difference of one embedding-sized tensor. The report distinguishes model parameter count from stored tensors rather than treating duplicated storage as a larger trained model.'),
        ('What did the tests actually prove?', 'Five tests exercised parsing, ambiguity handling, input-only cleaning, substring rejection, and ChatML formatting. Integration verification checked files, IDs, hashes, revisions, timings, and scorer execution. They do not prove semantic correctness, full parser robustness, production reliability, or universal model quality.'),
        ('What is the first experiment you would run next?', 'Collect a fresh representative holdout because the current test has been inspected. On dev, test a small GLiNER threshold/description study and LLM schema-constrained output with a documented alias policy. Then lock configurations and evaluate on the fresh set, including named EVENT and no-entity cases.'),
        ('What would you add before production?', 'Representative validation, explicit error-handling/retry policy, per-request audit checkpoints, resource/time limits, batching/concurrency tests, monitoring, and a business-specific quality target. Those are proposed next steps, not capabilities already delivered by this assignment harness.'),
        ('How did you use AI?', 'Codex generated the code, performed environment setup and local execution, and drafted reports. I should disclose that clearly, review the source, reproduce small checks, and explain the decisions. I should not claim independent authorship or understanding of code I have not studied.')
    ]
    for i, (question, answer) in enumerate(qa, 1):
        heading(f'Q{i}. {question}', 3)
        para(answer)

    heading('25. A safe live demonstration during the interview')
    para('Use saved predictions to demonstrate the pipeline evidence without waiting for a large model to load. Open the project root in a terminal. Explain what each command establishes before running it. A new model inference demonstration is optional and should be rehearsed; it is not necessary to reproduce a 50-snippet benchmark live.')
    code('.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v\n.\\.venv\\Scripts\\python.exe score.py data/test.jsonl predictions/gliner.jsonl --errors\n.\\.venv\\Scripts\\python.exe score.py data/test.jsonl predictions/qwen7b.jsonl --errors\n.\\.venv\\Scripts\\python.exe scripts/verify_submission.py', 'powershell')
    bullets([
        'Open src/common.py: point out the four labels, shared prompt, parser, and cleaner. State that LOC is not accepted in these reported runs.',
        'Open one predictions/*.metrics.json: find model revision, quantization, prompt/input hashes, raw output, a duration, and any parse error.',
        'Open results/results.html: compare overall scores, per-label performance, latency, memory, and phase/runtime labels.',
        'Open results/extension_frozen_settings.json: show the explicit prior-test-exposure disclosure and fixed generation settings.',
        'Open results/interrupted_extension/README.json: explain the interrupted attempt and why it was excluded.',
        'Finish with NOTE.md: make a customer-dependent recommendation supported by the measured counts.'
    ])
    heading('Tiny validation example to trace by hand', 3)
    code('Input text: "Singapore\'s harbour is busy."\nRaw entities:\n  {"text": "Singapore\'s", "label": "location"}\n  {"text": "Singapore", "label": "LOCATION"}\n  {"text": "Paris", "label": "LOCATION"}\n\nCleaner output:\n  {"text": "Singapore", "label": "LOCATION"}\n\nWhy: normalize label; strip possessive; deduplicate; reject Paris\nbecause it is absent from input. No gold annotation was consulted.')
    para('This is a validation illustration, not an additional measured model prediction. If asked to change the LOC mapping during the interview, demonstrate it in a separate exploratory branch/output path, explain why the old report must not be silently replaced, and reserve performance claims for a new evaluation.')

    heading('26. Stronger experiment design for the next day')
    bullets([
        'Define the customer objective first: balanced F1, maximum precision at a minimum recall, label-specific requirements, throughput, and acceptable hardware.',
        'Create a new held-out set before selecting improved configurations; keep the currently reviewed test set for exploratory analysis.',
        'On dev, vary one documented factor at a time: a small GLiNER threshold/description study; then LLM schema constraints or alias normalization. Record each trial and avoid unlimited dev overfitting.',
        'For parameter-size comparisons, use comparable runtimes and quantization levels across Qwen sizes. Separately benchmark the most practical deployable pipeline per model.',
        'Repeat timing runs after controlled warm-up, record background load and output lengths, and distinguish cold start, sequential steady state, batching, concurrency, and GPU experiments.',
        'Measure memory with a clearly defined scope; add load-time peak and, for GPU experiments, appropriate device memory measurements rather than reusing CPU RSS.',
        'Estimate uncertainty with paired snippet-level resampling: resample snippet IDs with replacement, aggregate TP/FP/FN, recompute both micro F1s, and examine their difference. Do not treat entities within one snippet as independent observations. This analysis was not performed in the submitted results.',
        'Add more named events, negative snippets, ambiguous organization/location names, honorifics, possessives, repeated mentions, and title exclusions. Keep gold annotation rules consistent and review disagreements.',
        'Only after local validation, benchmark a plausible deployment host and incorporate idle time, cold starts, retry policy, and expected utilization into a more realistic cost model.'
    ])

    heading('27. Self-assessment and statements to avoid')
    bullets([
        'Can you compute GLiNER precision, recall, and F1 from 104/19/4 without reading the result row?',
        'Can you explain why 7B can have fewer total errors but a lower F1?',
        'Can you show exactly where gold stops and model input begins in predict.py?',
        'Can you explain both parse_output and clean_entities using a malformed response?',
        'Can you distinguish greedy decoding from a guarantee of valid JSON?',
        'Can you explain why the 3B/7B comparison is exploratory and why runtime confounding matters?',
        'Can you reproduce the small tests and original scorer output without rerunning the models?',
        'Can you identify the interrupted attempt, the completed restart, and which files are actually scored?',
        'Can you explain what AI produced and what you have personally reviewed?'
    ])
    para('Avoid these claims: “we trained five models”; “90% accuracy”; “LLMs cannot recognize locations”; “larger models are always better”; “GLiNER is significantly better”; “all five used the same precision/runtime”; “the test set was untouched throughout”; “the cloud cost was measured”; “the parser cannot fail”; or “the ZIP includes the model weights.” Replace them with the specific, qualified statements supported by this report.')

    heading('28. Actual shared validation code and final submission map')
    para('The following is the actual code from src/common.py, included for study. Read it with the limitations above. It is not pseudocode and does not modify the reported experiment.')
    from src.common import parse_output, clean_entities
    code(inspect.getsource(parse_output) + '\n\n' + inspect.getsource(clean_entities), 'python')
    para('The final submission has one authoritative consolidated report: FULL_REPORT.html and its Markdown equivalent FULL_REPORT.md. README.md provides setup instructions and NOTE.md remains the assignment-required short recommendation. Archived earlier reports are explicitly historical evidence, not competing current reports. Data, scorer, source, tests, all completed prediction/audit files, model/runtime locks, errors, and interruption disclosure accompany them in the ZIP.')
    para('The packaged SUBMISSION_MANIFEST.json lists SHA-256 checksums for the payload files other than the manifest itself. It detects accidental file changes; it is not a signature proving authorship. Downloaded weights, runtime binaries, .venv, .cache, bytecode, and Git metadata are excluded. The recipient downloads the pinned runtime/checkpoints to reproduce inference. No project email, repository push, or deployment was performed.')
    para('Final preparation: read the short note first; study the tables and worked examples; trace the code; rehearse the questions; then run the small verification commands. Use the report as a guide to understanding and honest discussion, not as a substitute for understanding the submission.')
