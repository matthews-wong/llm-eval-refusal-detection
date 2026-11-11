# llm-eval-refusal-detection

Classify a model response as `answer`, `refusal`, `truncated` or `empty`, then
score the classifier against labelled responses. The aim is to measure two
failure modes separately:

- **over-refusal**: a benign prompt that got a refusal
- **under-refusal**: a prompt that should be declined but got an answer

Stdlib Python, fully offline. Responses are recorded fixtures, no model call is
made.

## Usage

```sh
make validate
python3 -m refusal_eval data/responses.jsonl
```

## Layout

- `refusal_eval/` detector, metrics and report CLI
- `data/responses.jsonl` labelled fixture responses
- `tests/` unit tests
