# Large sample game analysis — 2026-10-18

This analysis used the current `createInitialState()` definitions at commit
`bc9982b`, including Tower directional lanes and Big Hat balanced support.
The native engine's packaged `engine/tests/initial-state.json` and
`engine/tests/product-initial-state.json` still contain older patterns, so the
starting state was generated from the TypeScript game code and passed to each
native simulation request. All runs used a 200-ply cap.

## Samples

| Policy and method | Games | Plies | Red wins | Blue wins | Capped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random, rich rules (`simulateRandomGames`), seeds 90210 and 20261018 | 2,000 | 331,740 | 276 | 251 | 1,473 |
| Random, lean profiled rules (`profileRandomGames`), seed 314159 | 5,000 | 833,951 | 669 | 614 | 3,717 |
| Heuristic self-play (`simulateAiGames`), seeds 90210 and 20261018 | 100 | 11,853 | 34 | 30 | 36 |
| Random training candidates (`generateRandomTrainingBatch`), seeds 8675309–8675508 | 200 | 32,952 | 24 | 30 | 146 |

The rich random cohorts were highly consistent: 261 and 266 decisive games,
with 739 and 734 capped games. Red won 52.4% of the 527 decisive random games;
a 95% Wilson interval is approximately 48.1%–56.6%, so this sample does not
establish a side advantage. The heuristic cohorts were less consistent (37 and
27 decisive games), and the 100-game sample is too small for a strong balance
claim.

A cap is a game still in progress at ply 200. It is not a draw under the game
rules. In rich random games, a piece returning to a destination it had used
before occurred on 47.9% of plies, versus 29.1% for heuristic self-play.
This measures destination reuse, not exact board repetition. Random play
rescued 21,251 of 62,037 recorded rescue opportunities (34.3%); heuristic
self-play rescued 4,836 of 6,171 (78.4%).

## Move availability by game stage

The separate 200-game training batch was run one game per request so each
position has an exact ply index. Its candidate set uses the native training
move validator, which is an efficient proxy for playable moves rather than an
exact count of every browser move annotation.

| Ply range | Positions | Mean candidates | Median | 10th percentile | At most 2 candidates | Tower with no candidate, when present | Mean Big Hat candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–19 | 4,000 | 37.07 | 37 | 24 | 0.05% | 0.10% | 4.40 |
| 20–59 | 7,645 | 22.39 | 21 | 6 | 1.70% | 0.99% | 6.86 |
| 60–119 | 9,566 | 11.40 | 10 | 5 | 3.83% | 1.42% | 7.96 |
| 120–199 | 11,741 | 8.65 | 9 | 5 | 3.81% | 1.23% | 8.37 |

Only 326 of 11,741 late-stage positions still had a Tower for the player to
move. When present, Towers averaged 8.93 candidates in those positions. Big
Hat candidates rose as other pieces left the board; zero Big Hat candidates
occurred in 0.02% of late positions. Overall move availability fell mainly
because games lost pieces, while roughly 3.8% of late positions offered two
or fewer candidates.

The 5,000-game lean profiler averaged 20.10 candidate moves per ply. Native
candidate generation averaged 0.0031 ms per ply and move application 0.0233
ms per ply. These timings describe the lean native simulation; they do not
measure browser click latency or the live hard AI's 2.5-second search budget.

## Interpretation and limits

- Long games are common even with strategic play: 36% of the heuristic games
  remained in progress at 200 plies, compared with 73.7% of rich random games.
- This sample does not show a systemic late-game Tower or Big Hat move drought
  when those pieces survive. The total move pool shrinks sharply with attrition.
- The heuristic batch used a 10 ms search budget, not the web game's hard AI
  policy. No human games were included.
- The stage sample uses training move validation. A separate browser benchmark
  is needed to quantify click-to-dot latency across actual late-game states.
