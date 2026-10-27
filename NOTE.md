# Recommendation — five-model local NER bake-off

The comparison includes GLiNER small v2.5 and Qwen2.5 0.5B, 1.5B, 3B, and
7B, running locally on an i7-11800H laptop. Original models used float32
PyTorch; 3B/7B used official Q4_K_M checkpoints and llama.cpp to fit 16 GB RAM.
The latter are a user-requested post-review extension: test errors had already
been inspected. The shared prompt and cleaner stayed unchanged; this is not
a fresh holdout or a parameter-count-only comparison. The first 7B attempt was
interrupted after 13 snippets and excluded; its full restart used identical
settings. Original results and settings were preserved.

**Recommendation.** For cost-sensitive use I would choose GLiNER: micro F1
**0.900**, median **0.236 s**, p95 **0.283 s**, peak RAM **1,139 MiB**, and
estimated compute cost **$0.0148/1,000 snippets**. It remains the best overall
F1 configuration, although its small lead over 7B is not established as
statistically significant on 50 snippets.

Qwen 7B achieved F1 **0.889**, median **13.705 s**, p95 **17.700 s**, peak RAM
**5,363 MiB**, and **$0.820/1,000**. It is the stronger option when quality
means avoiding false positives: precision was **0.978** versus GLiNER's
**0.846** (2 versus 19 false positives). GLiNER had higher recall, **0.963**
versus **0.815** (4 versus 20 missed entities). For balanced exact-match F1,
choose GLiNER; for precision-first use, evaluate 7B against customer error
costs. Its PERSON/ORG F1s were higher, but LOCATION/EVENT F1s were lower.

Other test F1s were 0.338 (0.5B), 0.663 (1.5B), and 0.638 (3B). Quantized
3B had median 3.796 s and estimated cost $0.244/1,000, illustrating why runtime
and precision must be considered alongside size. Complete metrics are in
`results/results.html` and `results/results.csv`.

**Observed failure pattern: GLiNER over-extracts generic expressions.**

- t11: “He said that they would announce the results later this week.”
  Prediction: **He / PERSON**. Gold: no entities.
- t34: “Oakridge Primary School will host the district science fair on
  12 March.” It found the school but added **district science fair / EVENT**;
  gold contains only **Oakridge Primary School / ORG**.
- t38: “The Great Barrier Reef suffered another bleaching event, scientists
  in Cairns said.” It found the two locations but added **bleaching event /
  EVENT** and **scientists / ORG**. Gold contains only the locations.

These errors can create spurious records despite high recall.

**Next day and limitations.** Confirm the ranking on fresh, larger held-out
data, especially named EVENT cases and negative snippets. On dev, test GLiNER
threshold/label descriptions, LLM schema-constrained JSON, and a documented
`LOC` alias policy. The 0.5B pipeline had 9 malformed responses and no accepted
LOCATION predictions, so its score combines recognition and output compliance.
Repeat timing runs and test GPU execution. Evidence is limited to 50 synthetic
snippets, fixed baseline settings, exact matching, and sampled RSS. Costs
extrapolate local mean throughput at the recorded $164/month AWS Lightsail
32 GB host assumption, amortized over 730 hours; they are not cloud measurements
and exclude loading, idle time, retries, storage, and hardware differences.

**AI disclosure:** Codex generated the harness, executed local experiments
and validation, and drafted documentation and analysis.
