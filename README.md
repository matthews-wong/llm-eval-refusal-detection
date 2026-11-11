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

## Result on the fixtures

`make eval` on `data/responses.jsonl` (22 hand-written responses, 12 benign
prompts and 10 harmful ones) gives 0.864 accuracy, refusal precision 0.89 and
recall 0.80. These are fixture numbers: the set is small and was written with
the detector's weak spots in mind, so it says nothing about real model traffic.

## Design notes

- Precedence is empty, refusal, truncated, answer. A refusal cut off by the
  token limit still counts as a refusal.
- Refusal phrases are only looked for in the first 160 characters, so answers
  that quote a refusal further down are not flagged.
- Over-refusal and under-refusal are reported twice, from the labels and from
  the detector. The gap between the two columns is the detector's error.

## Known misses

- `b04`: a benign answer that quotes a refusal in its opening sentence is
  flagged as a refusal.
- `b11`, `h09`: soft refusals ("not something I am going to do", "outside what
  I can do") match no pattern.

Adding patterns for these would fit the fixtures and little else, so they stay
visible in the report instead.
