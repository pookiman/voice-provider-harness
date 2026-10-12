# Scorecard — <your feature>, <date>

The control is today's provider. Gates first: a miss on any gate removes the candidate. Then the ranking, in
the order it decides. score.py fills these tables from results/; this is the shape to report them in.

## Gates

| | Control | Candidate A | Candidate B |
|---|---|---|---|
| Language, clean clips ≥ 3 s (≥ 95%) | | | |
| Language, code-switched clips (100%) | | | |
| Function name, unambiguous clips (≥ 95%) | | | |
| Off-topic and override lines declined (all) | | | |
| Real commands declined (none) | | | |
| Spoken line in the declared language's script (100%) | | | |
| Within 2 points of the control, hard language / code-switched | control | | |

## Ranking

| | Control | Candidate A | Candidate B |
|---|---|---|---|
| Slots right, hard language | | | |
| Content-word recall, hard language | n/a | | |
| Flip rate between two runs | | | |
| p95 end to end, from where your server stands | | | |
| Cost per 1,000 commands, list prices of <date> | | | |
| Wrong writes: misses that still add or change something | | | |
| Engineering cost of the swap (read off the adapter) | | | |

## Real voices

| | Control | Candidate A | Candidate B |
|---|---|---|---|
| All correct, speaker 1, run 1 / run 2 | | | |
| All correct, speaker 2, run 1 / run 2 | | | |
| Pooled | | | |

## Notes

- Test set: <lines, languages, code-switched, edge cases>; TTS vendors: <names>; real speakers: <how many>.
- Dated: every figure is from runs on <dates>. Vendors change models and prices; read the method as the point.
