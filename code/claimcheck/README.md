# ClaimCheck

A claim extraction and evidence-verification pipeline for the NLP project. Paste a news article, a WhatsApp forward or a social media post. ClaimCheck splits it into individual factual claims, retrieves the best matching passages from a local evidence folder, and asks a **local Ollama model** to label each claim `supported`, `contradicted` or `not_enough_evidence`. Every decisive verdict must quote its evidence, and the code verifies that the quote really appears in the retrieved text.

```
text ──> clean ──> [LLM] extract atomic claims ──> [TF-IDF] retrieve top-k passages per claim
                                                          │  (no relevant passage -> no LLM call)
                                                          v
        report <── quote check in code <── [LLM] verify claims in batches, with quotes
```

| Stage | Technique | File |
|---|---|---|
| Clean-up | URL / "Forwarded" banner removal, unicode normalisation, line-aware sentence splitting | `textutils.py` |
| Claim extraction | LLM with few-shot prompt, JSON-schema constrained output | `pipeline.py`, `prompts.yaml` |
| Retrieval | Sentence windows, stemmed unigram+bigram TF-IDF, cosine ranking, overlap removal | `retriever.py` |
| Live evidence | Wikipedia search + extracts via the MediaWiki API, cached on disk | `wiki_source.py` |
| Verification | LLM entailment-style labelling, batched, with verbatim evidence quotes | `pipeline.py`, `prompts.yaml` |
| Hallucination guard | Quote must appear in a retrieved passage, otherwise the verdict is downgraded | `verifier.py` |
| LLM access | Ollama `/api/chat`, structured outputs, retries, JSON repair, disk cache | `llm_client.py` |

## Setup

1. Make sure Ollama is running and a model is pulled:
   ```
   ollama pull llama3.2
   ```
2. Install and run:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows   (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
3. Use it:
   ```
   python cli.py --file samples/article_1_whatsapp_forward.txt     # command line
   python app.py                                                   # web UI at http://127.0.0.1:5000
   python evaluate.py                                              # accuracy on labeled claims
   python -m pytest                                                # unit tests (no Ollama needed)
   ```

No API key is needed. Change the model in `config.yaml` (`ollama.model`), with `--model`, or with the `OLLAMA_MODEL` environment variable. `OLLAMA_HOST` points to a non-default server. An 8B model such as `llama3.1:8b` or `qwen2.5:7b` follows the quote rule noticeably better than a 3B model.

## Evidence

ClaimCheck uses two evidence sources, ranked together in one TF-IDF space per claim:

1. **Wikipedia (live, on by default)**: the claim extractor also proposes which Wikipedia pages to open (for "X is president of Y" it asks for "X" and "President of Y"). The app searches those titles through the public MediaWiki API, reads the intro of the top hits and the full text of the best one (`wiki_source.py`), and falls back to keyword search on the claim if the proposed titles find nothing. Raw API responses are cached in `.cache/wikipedia`, so repeated runs work offline. If Wikipedia is unreachable the run continues on local notes and shows a warning.
2. **Local notes**: any `.txt` / `.md` files in `evidence/`. The bundled files cover eight topics (Chandrayaan-3, Apollo 11, the Taj Mahal, the Great Wall, Mount Everest, the "10% of the brain" myth, 5G and COVID-19, lightning). Add your own notes on any topic.

Turn Wikipedia off with `wikipedia.enabled: false` in `config.yaml`, or for one run with `python cli.py --local-only ...`. Claims that neither source settles come back `not_enough_evidence`. `evaluate.py` always runs in local-only mode, so its accuracy is reproducible and measures the model rather than Wikipedia's current content.

## Design decisions

- **Prompt file (`prompts.yaml`)**: one system prompt, one few-shot example and one user template per stage. The extraction prompt defines an *atomic, self-contained* claim and lists what to skip (opinions, forwarding instructions). The verification prompt fixes a three-label taxonomy, forbids outside knowledge, and requires a verbatim quote. The schema puts `reasoning` before `label`, so the model reasons before it commits.
- **Structured output**: each call sends a JSON schema in Ollama's `format` field. The reply is validated again locally with `jsonschema`. On failure the client re-prompts once with the error, and a failed batch falls back to one claim per call.
- **Hallucination control**: a `supported` or `contradicted` verdict without a quote that occurs verbatim in a retrieved passage (case, punctuation, quote style and number formatting ignored) is not accepted. The model is asked once to correct itself (`quote_repair_attempts`); if the retry is still not verbatim the verdict becomes `not_enough_evidence`. The UI shows the original model label and the rejected quote, so downgrades are visible.
- **Evidence style**: evidence notes use one fact per sentence, which makes verdicts less ambiguous for small models.
- **Efficiency**: retrieval is classic NLP and costs no LLM call. Claims with no relevant passage skip the LLM entirely. Verification is batched (`batch_size`, default 4). All LLM responses are cached on disk by a hash of the full request.
- **Reliability score**: the share of *decided* claims that are supported. Claims with not enough evidence are reported separately and do not count against the text.
- **Configuration**: every threshold, model setting and path is in `config.yaml`.

## Evaluation

`samples/labeled_claims.json` holds 25 claims with gold labels (10 supported, 10 contradicted, 5 not enough evidence). Five of them are true in the real world but absent from the evidence, to test that the model does not use outside knowledge.

Run `python evaluate.py` (local notes only) and paste the output here:

| model | accuracy | notes |
|---|---|---|
| _fill in after running_ | | |

## Limitations

- Retrieval is lexical, so a claim phrased with very different words from the evidence can be missed. Swapping in sentence embeddings would be the next step.
- Verdicts are only as good as the evidence: Wikipedia can be wrong or out of date, and a claim that is false only by omission ("X is the president of Y") may stay `not_enough_evidence` unless a passage states the true fact.
- Small local models can mis-label long or compound claims, which is why the quote check exists.

## Project layout

```
app.py  cli.py  evaluate.py        entry points (web UI, command line, accuracy run)
pipeline.py                        orchestration of the stages
llm_client.py                      Ollama client
retriever.py  wiki_source.py       retrieval (local + Wikipedia)
verifier.py                        guardrails
textutils.py  schemas.py  config.py
config.yaml  prompts.yaml          configuration and prompt file
evidence/  samples/                evidence notes, test inputs, labeled claims
templates/index.html               web UI
tests/                             49 unit tests with the LLM mocked
```
