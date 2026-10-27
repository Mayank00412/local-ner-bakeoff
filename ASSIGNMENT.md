# Case Assignment: Local NER Bake-off

**Role:** ML Engineering Intern
**Effort:** about 4-5 hours. Please do not spend more than one working day.
**Compute:** your own machine only. No cloud accounts or paid APIs are needed or expected.

---

## 1. Why we are asking this

We build ML pipelines for media and enterprise customers. A recurring question is whether a cheaper, smaller model can do a job that we would otherwise give to an expensive one. Named Entity Recognition (NER) is a good small example: you can solve it with a large language model, a small language model, or a purpose-built specialist model, and the trade-offs between quality, speed, and cost are real.

This exercise is a miniature version of that question. We care more about how you think and report than about squeezing out the highest score.

## 2. The task

Compare **at least three models** on a small NER task, all running locally, and tell us which one you would choose and why.

1. **Pick your models.** At least one must be a *specialist* (non-LLM) NER model, and at least one must be a *local LLM*. The third is your choice. See the suggestions in section 5.
2. **Build a small harness** that runs each model over the dataset and writes predictions in the format in section 4.
3. **Score** each model with the provided `score.py`.
4. **Measure** latency, memory, and estimate cost (section 6).
5. **Write a short note** with your recommendation (section 7).

## 3. Data

All data is **synthetic**: the people and organisations are made up, and some places and events are real. No real individuals are described.

| File | Snippets | Use it for |
|---|---|---|
| `data/dev.jsonl` | 10 | Developing and tuning your prompts and settings |
| `data/test.jsonl` | 50 | Final reporting only |

Each line looks like this:

```json
{"id": "d01", "text": "Priya Nair visited the Gardens by the Bay on Saturday to open the Lantern Festival.",
 "entities": [{"text": "Priya Nair", "label": "PERSON"},
              {"text": "Gardens by the Bay", "label": "LOCATION"},
              {"text": "Lantern Festival", "label": "EVENT"}]}
```

**Please do not tune anything on `test.jsonl`.** Develop on `dev`, then run once on `test` and report that. If you do look at test errors to understand failures (we expect you to), say so in your note and treat any change you make afterwards as no longer a clean result.

### Labels

| Label | What to tag | Notes |
|---|---|---|
| `PERSON` | Names of individual people | Exclude titles and honorifics: tag "Daniel Koh" in "Minister Daniel Koh", not the whole phrase |
| `ORG` | Companies, agencies, institutions, schools, clubs, teams, media outlets | |
| `LOCATION` | Countries, cities, districts, rivers, mountains, landmarks, venues, buildings | |
| `EVENT` | Named events: festivals, tournaments, summits, holidays, storms | Tag only named events, not generic ones like "the match" |

Further rules:

- Tag the **longest name** only. Do not also tag a place or person inside a longer name ("Busan International Film Festival" is one EVENT).
- Nationalities and demonyms ("Jordanian") are not tagged unless part of an organisation name.
- Titles of films, books, and songs are **not** tagged, even if they contain a place or person name.
- Dates, numbers, and job titles are not tagged.
- Repeated mentions of the same entity in one snippet count once.
- Possessives are stripped: tag "Singapore", not "Singapore's".
- Some snippets contain no entities.

## 4. Prediction format

One JSON object per line, one line per snippet:

```json
{"id": "t01", "entities": [{"text": "Daniel Koh", "label": "PERSON"}, {"text": "Tuas", "label": "LOCATION"}]}
```

Score with:

```bash
python score.py data/test.jsonl predictions_<model>.jsonl --errors
```

`score.py` uses exact entity matching: the text must match (case-insensitive) and the label must match. Do not modify it when reporting results. If you think the metric is unfair for some cases, say so in your note.

## 5. Model suggestions (a menu, not a requirement)

The local model landscape changes fast, so use these as starting points. You are free to choose others, and your reasoning for the choice is part of what we read.

**Specialist models**
- GLiNER: a small zero-shot NER model that takes your label names as input and runs on CPU.
- spaCy pipelines, or a fine-tuned BERT NER model from Hugging Face (note that many of these use a fixed label set that differs from ours, so you will need to map labels).

**Local LLMs** (for example via Ollama or llama.cpp)
- Qwen3 (4B or 8B), Gemma 3 (4B or 12B), Llama 3.x (3B or 8B), Phi-4-mini.
- Larger models such as gpt-oss-20b are fine if your machine can run them, but are not expected.

Record the exact model name, size, and quantisation you used.

## 6. What to measure

For each model, on `test.jsonl`:

- **Quality:** precision, recall, and F1 (overall and per label) from `score.py`.
- **Latency:** median and p95 seconds per snippet, excluding model load and a short warm-up.
- **Memory:** peak RAM or VRAM while running, approximately.
- **Hardware:** CPU, RAM, GPU if any, and OS. We will not penalise slow absolute numbers on modest laptops.
- **Estimated cost per 1,000 snippets:** pick a plausible hourly price for a cloud machine that could run the model (state your assumption and where the number came from), then convert your measured throughput into a cost per 1,000 snippets. Briefly explain what this estimate leaves out.

## 7. Deliverables

1. **Code and README.** Everything needed to reproduce your results, with clear run instructions and the library and model versions you used.
2. **Results table** with one row per model: F1 (overall), per-label F1, median latency, p95 latency, peak memory, estimated cost per 1,000 snippets.
3. **A short note (about one page)** covering:
   - Which model you would choose for a customer who cares about cost, and which for one who cares mainly about quality.
   - One concrete failure pattern you observed, with two or three example snippets.
   - What you would test next if you had another day.
   - Limitations of your experiment (we expect an honest answer to "how much should we trust these numbers?").
4. **AI-assistant disclosure.** You may use AI coding assistants. Tell us in a sentence where you used them. In the follow-up conversation you should be able to explain every part of your submission.

## 8. How we will read it

We are looking for sound evaluation practice, reliable handling of messy model output, sensible reasoning about cost, clear and honest writing, and the quality of the choices you made. We are not looking for fine-tuning, elaborate frameworks, or the highest possible F1. A simple, correct, well-explained submission is better than a complex one.

## 9. Submitting

Send a zip file or a link to a private repository. We will schedule a 20-30 minute follow-up conversation in which you walk us through your approach and results.

If anything in the brief is unclear, make a reasonable assumption, write it down in your README, and carry on.
