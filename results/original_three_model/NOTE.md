# Recommendation — local NER bake-off

Three fixed configurations ran locally on an i7-11800H laptop: GLiNER small
v2.5 (166M parameters), Qwen2.5-0.5B-Instruct (494M), and
Qwen2.5-1.5B-Instruct (1.544B). All used CPU float32, without quantization.
Dev was a functional check; no prompt or threshold search was performed.
Settings were frozen before each model's single 50-snippet test run. I inspected
test errors afterwards to write this note; no inference settings changed.

**Recommendation.** For a cost-sensitive customer I would choose GLiNER:
micro F1 **0.900**, median **0.236 s**, p95 **0.283 s**, and peak RAM
**1,139 MiB**. Estimated compute cost is **$0.0148 per 1,000 snippets** under
the stated host-price assumption. Qwen 0.5B achieved F1 **0.338**, median
**9.849 s**, and **$0.687/1,000**; Qwen 1.5B achieved F1 **0.663**, median
**19.602 s**, and **$1.399/1,000**. Full precision, recall, per-label F1,
memory, and latency results are in `results/results.html` and `results.csv`.

For a customer prioritizing quality under the assignment's equal-weight
entity F1 metric, I would also choose GLiNER among these configurations.
It recalled 104 of 108 gold entities. However, Qwen 1.5B had slightly higher
precision (0.866 versus 0.846), at much lower recall (0.537 versus 0.963).
A customer who strongly penalizes false positives needs a separate operating
point evaluation. GLiNER's EVENT F1 was 0.833, so the overall score should not
be treated as uniform reliability across labels.

**Observed failure pattern: over-extracting generic expressions.** GLiNER
sometimes recognizes a semantic category rather than a proper name:

- t11: “He said that they would announce the results later this week.” It
  predicted **He / PERSON**; gold contains no entities.
- t34: “Oakridge Primary School will host the district science fair on
  12 March.” It correctly found the school but also predicted **district
  science fair / EVENT**; gold contains only **Oakridge Primary School / ORG**.
- t38: “The Great Barrier Reef suffered another bleaching event, scientists
  in Cairns said.” It correctly found **Great Barrier Reef** and **Cairns**,
  but added **bleaching event / EVENT** and **scientists / ORG**. Gold contains
  only the two locations. Such errors could create spurious customer records
  or alerts even when recall is excellent.

**Next day.** On dev, test explicit namedness instructions or label descriptions
and a small GLiNER threshold sweep. For the LLMs, test schema-constrained JSON
and a documented `LOC` alias mapping. Qwen 0.5B had 9/50 malformed responses
and no accepted LOCATION predictions; its score reflects both recognition and
strict output compliance. Any revised pipeline needs a new held-out set, with
more EVENT examples and negative snippets. Repeat latency runs and compare a
quantized larger LLM on GPU.

**Limitations.** Fifty synthetic snippets provide weak evidence about production
domains, statistical uncertainty, and rare cases. These are baseline settings,
not optimized model capabilities. Exact matching penalizes boundary differences
and does not award partial credit. Memory is sampled RSS; latency is one local
sequential run after warm-up. Costs extrapolate local mean throughput onto a
$164/month AWS Lightsail 32 GB CPU host, amortized over 730 hours; they are not
cloud measurements and exclude idle time, loading, storage, and hardware
differences. The identical host price is not an instance-size optimization.

**AI disclosure:** Codex generated the harness, executed the local experiments
and validation, and drafted the documentation and analysis.
