# Wavefielder Signature Nucleation Investigation

This is an experimental analysis only. It does not change normal Wave Field
rules, piece definitions, wave functions, board geometry, signs/phases, or
superposition.

## Ground Truth Used

The search mirrors the current TypeScript implementation:

- piece types: `pawn`, `rook`, `spy`, `king`
- display names: Round Hat, Tower, Triangle Hat, Big Hat
- Chebyshev-ring distance and decay from `src/field/kernels.ts`
- component definitions from `src/field/componentDefinitions.ts`
- piece strengths, home energy, and friendly/hostile scaling from
  `src/game/constants.ts`
- signed superposition from `src/field/evaluateField.ts`

The default active components are:

- Round Hat: checkerboard
- Tower: tower grid + pull-gap-push ring
- Triangle Hat: checkerboard only; its second diamond-core component has
  coefficient `0` in `DEFAULT_COMPONENTS`
- Big Hat: ring mode + horizontal-versus-vertical mode

## Recommended Signature Definition

The most faithful local signature is the vector of production contribution
values produced by a single target piece centered at the candidate nucleation
site, sampled over a local Chebyshev square and excluding the origin.

For the first pass the script uses radius 3, giving the 7x7 local neighborhood
minus the center. Excluding the center avoids confusing piece identity with
home energy at the occupied square; nucleation asks whether an empty square's
surrounding field looks like the field around a piece.

The report uses several metrics because no single one preserves all important
distinctions:

- cosine similarity: shape match independent of magnitude
- scaled normalized RMSE: residual after the best scalar fit
- raw normalized RMSE: actual magnitude-aware error
- sign agreement: whether positive/negative local structure matches
- magnitude ratio: norm of generated field divided by norm of target signature

Cosine alone is not enough. Some candidates score well by shape but are 2x to
4x too strong, which would make accidental generation harder to control unless
a magnitude constraint is added.

## Search Method

Script: `scripts/wavefielder-signature-search.mjs`

Run:

```sh
npm run analyze:wavefielder -- --beam=220 --random=2500 --max-pieces=6 --placement-radius=3 --signature-radius=3 --scaled=false
```

The search:

1. Builds candidate atoms from legal existing piece types at relative offsets
   around an empty target.
2. Uses actual signed superposition. `ownerSign: 1` is same-polarity/red in the
   script's coordinates; `ownerSign: -1` is opposite-polarity/blue.
3. Searches 1 through 6 pieces with beam search.
4. Separately samples random configurations with the same piece counts and
   radius to estimate accidental-match rates.

This is not a proof of global optimality. It is a local, reproducible sparse
search.

## Scale-I Findings

The strongest simple result is Triangle Hat from two Round Hats:

```text
. . . . .
. . R . .
. . * . .
. . R . .
. . . . .
```

Relative coordinates:

```json
[
  { "piece": "Round Hat", "ownerSign": 1, "x": 0, "y": -2 },
  { "piece": "Round Hat", "ownerSign": 1, "x": 0, "y": 2 }
]
```

Metrics:

- cosine: `0.8683`
- scaled normalized RMSE: `0.4961`
- raw normalized RMSE: `0.5291`
- sign agreement: `1.0000`
- magnitude ratio: `1.0522`

This is the best current evidence for a native generative relation. It is
simple, memorable, same-polarity, magnitude-close, and sign-perfect in the
sampled radius.

Other notable Scale-I candidates:

### Round Hat

Best simple two-piece candidate:

```json
[
  { "piece": "Triangle Hat", "ownerSign": 1, "x": 0, "y": -2 },
  { "piece": "Triangle Hat", "ownerSign": 1, "x": 0, "y": 2 }
]
```

Metrics:

- cosine: `0.8821`
- scaled normalized RMSE: `0.4710`
- raw normalized RMSE: `1.4303`
- sign agreement: `0.5000`
- magnitude ratio: `2.2327`

This is a shape match but not a good nucleation rule by itself: sign agreement
is poor and the magnitude is more than twice the target.

### Tower

Best six-piece local candidate:

```json
[
  { "piece": "Triangle Hat", "ownerSign": 1, "x": 0, "y": 2 },
  { "piece": "Triangle Hat", "ownerSign": 1, "x": 1, "y": -3 },
  { "piece": "Round Hat", "ownerSign": -1, "x": 1, "y": -1 },
  { "piece": "Round Hat", "ownerSign": 1, "x": 0, "y": -2 },
  { "piece": "Round Hat", "ownerSign": -1, "x": -1, "y": 1 },
  { "piece": "Round Hat", "ownerSign": 1, "x": -1, "y": 3 }
]
```

Metrics:

- cosine: `0.7475`
- scaled normalized RMSE: `0.6643`
- raw normalized RMSE: `0.7606`
- sign agreement: `0.7045`
- magnitude ratio: `1.1180`

Tower does not currently show a compelling small natural construction.

### Big Hat

Best six-piece local candidate:

```json
[
  { "piece": "Tower", "ownerSign": 1, "x": 3, "y": 1 },
  { "piece": "Tower", "ownerSign": 1, "x": -3, "y": 1 },
  { "piece": "Tower", "ownerSign": 1, "x": 0, "y": 1 },
  { "piece": "Triangle Hat", "ownerSign": 1, "x": 1, "y": -1 },
  { "piece": "Tower", "ownerSign": 1, "x": -3, "y": -2 },
  { "piece": "Round Hat", "ownerSign": -1, "x": -2, "y": 1 }
]
```

Metrics:

- cosine: `0.8269`
- scaled normalized RMSE: `0.5623`
- raw normalized RMSE: `0.5815`
- sign agreement: `0.9375`
- magnitude ratio: `0.9751`

This is moderately interesting mathematically, but it is not especially simple
or memorable.

## Accidental-Match Rates

For the Scale-I run above, 2,500 random configurations per piece count were
sampled. At 6 pieces, the random cosine 99th percentiles were:

- Round Hat: `0.6351`
- Tower: `0.5653`
- Triangle Hat: `0.6452`
- Big Hat: `0.6072`

No random 6-piece sample reached cosine `0.90` or `0.95`.

This suggests a useful separation for a local radius-3 signature search:
random play is usually far below the optimized candidates. However, cosine
thresholding alone is unsafe because some high-cosine candidates have poor sign
agreement or wrong magnitude. A native-feeling nucleation detector should
probably require:

- high cosine
- high sign agreement
- bounded magnitude ratio
- target square empty
- optionally local extremum or stability under one-square perturbations

Those constraints follow from the field math. They are less arbitrary than a
crafting recipe, but the exact thresholds would be game-design choices.

## Exact Reproduction

No exact reproduction was found in this local search. Exact equality also looks
unlikely for small configurations because the kernels combine shifted decaying
Chebyshev patterns with piece-specific strengths and asymmetric component
scales. The script does not prove impossibility.

## Scale-II Probe

Scale-II pieces do not exist in the current implementation, so any Scale-II
signature requires an extension. The script uses a deliberately simple
extension:

```text
K_scale(delta) = K_base(trunc(delta / scale)) / scale
```

This is a reasonable mathematical probe, not a Wave Field rule.

A smaller probe was run with:

```sh
node scripts/wavefielder-signature-search.mjs --beam=120 --random=1000 --max-pieces=6 --placement-radius=2 --signature-radius=2
```

Best 6-piece Scale-II cosine scores:

- Round Hat II: `0.6659`, raw error `3.5536`, magnitude ratio `4.1403`
- Tower II: `0.7560`, raw error `0.9775`, magnitude ratio `1.4819`
- Triangle Hat II: `0.7184`, raw error `2.5346`, magnitude ratio `3.1556`
- Big Hat II: `0.6869`, raw error `0.9256`, magnitude ratio `1.2602`

No strong or self-similar Scale-I-to-Scale-II construction appears in this
probe. In particular, the clean Triangle-Hat-from-two-Round-Hats relation did
not obviously lift into a recursive higher-order construction under this
dilation rule.

## Assessment

Field-signature nucleation appears partially native to the current Wave Field
mathematics:

- The Round Hat / Triangle Hat relationship is real and simple because the
  Triangle Hat's active component is currently just the checkerboard mask with
  different strength and scaling. Two Round Hats at distance 2 naturally
  approximate a Triangle Hat signature with perfect local sign agreement and
  near-target magnitude.
- Tower and Big Hat do not show equally clean small constructions in this
  first pass.
- Random accidental matches were rare at high cosine thresholds in the sampled
  local space.
- Scale-II/self-similar nucleation is not yet supported by strong evidence; it
  depends on a not-yet-implemented scaling definition.

The promising path is not a general crafting grammar yet. It is a small
emergent relation plus a reproducible search framework for finding more.

## Broadened Resonance Search

Second-stage script:

```sh
npm --silent run analyze:wavefielder:resonance -- --max-pieces=6 --beam=150 --random=450 --top=12 --scale=false
```

This pass treats the four piece types as reusable species rather than fixed
inventory. It searches every same-polarity source composition from 2 to 6
pieces, compares each source/target pair against composition-matched random
placements, and annotates candidates with local extremum status, one-cell
perturbation robustness, and a simple symmetry score.

The same-polarity restriction is intentional for this pass: it asks what a
single growing population can synthesize without relying on opposite-polarity
pieces as ingredients. Mixed-polarity searches remain useful, but they answer a
different tactical question.

### Resonance Matrix Highlights

The strongest cross-species relations from this pass were:

| Sources | Target | Cosine | Sign | Mag. ratio | Random percentile | Robust min | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| 2 Round | Triangle | 0.8683 | 1.0000 | 1.0522 | >=99th | 0.4034 | simplest strong cross-species relation |
| 3 Round | Triangle | 0.8570 | 1.0000 | 1.0282 | >=99th | 0.3666 | still simple, near magnitude-correct |
| 4 Round | Triangle | 0.8937 | 1.0000 | 1.2865 | >=99th | 0.4788 | stronger but less minimal |
| 5 Round | Triangle | 0.9190 | 1.0000 | 1.6003 | >=99th | 0.6738 | robust, magnitude high |
| 3 Round + 3 Tower | Big Hat | 0.8157 | 0.9792 | 1.0063 | >=99th | 0.6830 | best clean no-Big source for Big Hat |
| 6 Triangle | Big Hat | 0.8143 | 0.9583 | 1.0089 | >=99th | 0.6849 | Big Hat can be approximated by many Triangles |
| 1 Round + 4 Tower + 1 Triangle | Big Hat | 0.8226 | 0.9375 | 1.0520 | >=99th | 0.6931 | strong but complex |

Tower remained the weakest target. Its best same-polarity six-piece result was:

```text
3 Round + 3 Tower -> Tower
cosine 0.7729, sign 0.8864, magnitude ratio 1.6520, robust min 0.5872
```

That is selective relative to random placements, but it is not a clean
low-complexity construction.

### Pareto-Optimal Simple Candidates

The useful simple frontier is:

```text
2 Round -> Triangle
cos 0.8683, sign 1.0000, mag 1.0522

. . R . .
. . . . .
. . * . .
. . . . .
. . R . .
```

```text
2 Triangle -> Triangle
cos 0.9093, sign 1.0000, mag 1.5342
```

This is a self-resonance rather than new species generation. It may matter for
growth or reinforcement but should not be confused with a cross-species rule.

```text
3 Round + 3 Tower -> Big Hat
cos 0.8157, sign 0.9792, mag 1.0063
```

This is the best evidence that Big Hat is synthesizable without already having
a Big Hat, but it is a six-piece scaffold, not a compact recipe-like motif.

### Random Baselines and Selectivity

The broadened search reports random percentiles per matched composition. The
notable candidates above all landed at or beyond the 99th percentile of random
placements using the same source composition and radius.

This matters: several resonances with cosine only `0.5-0.8` are still
structurally unusual. They are not near-perfect kernel replicas, but they are
not ordinary random field accidents either.

### Local Extremum and Robustness

The Round-to-Triangle relation is strong and simple but not a local absolute
extremum at the target under the current measurement. Its one-cell perturbation
minimum cosine was `0.4034`, so it is recognizable but position-sensitive.

The strongest Big Hat scaffold was more robust:

```text
3 Round + 3 Tower -> Big Hat
robust min cosine 0.6830
```

This supports the interpretation that Big Hat is a broader organizing pattern,
not a tiny two-piece nucleation event.

## Resonance Graph

Using a conservative "strong edge" filter:

- cosine >= `0.75`
- sign agreement >= `0.85`
- magnitude ratio between `0.5` and `1.8`

The native graph looks like this:

```text
Round -> Triangle       strong, simple, native
Round -> Round          self-resonance
Triangle -> Triangle    self-resonance
Round + Tower -> Big    scaffolded path
Tower + Big -> Big      self/scaffold reinforcement
Triangle x many -> Big  scaffolded path
```

What did not appear:

```text
Triangle -> Tower       no clean path found
Tower -> Round          no clean path found
Big -> Round/Tower      no clean path found
```

So the graph is not a neat cycle like Round -> Triangle -> Tower -> Round. It
is more like:

```text
Round family -> Triangle family
Round/Tower/Triangle mixtures -> Big Hat
Tower is hard to synthesize
```

## Big Hat Analysis

Big Hat is not impossible to approximate. In fact, six-piece sources can match
it with good shape, sign, magnitude, and robustness. But the simple searches do
not show a compact two- or three-piece Big Hat nucleation motif unless Big Hats
are already present as source pieces.

Current interpretation:

- Big Hat behaves more like a higher-order/organizing piece than a basic unit.
- Its kernel can be synthesized by larger scaffolds.
- It is not naturally reachable from Round alone under the current strong-edge
  thresholds.
- If Big Hat is generatable, it probably wants either a larger construction or
  a lower-dimensional/focusing signature rather than full-kernel replication.

## Alternative Scale-II Hypotheses

Scale-II pieces do not exist in normal Wave Field. The second script therefore
tests hypotheses only:

```sh
npm --silent run analyze:wavefielder:resonance -- --base=false --diagnostic=false --max-pieces=4 --beam=55 --random=100 --top=4
```

Best 2-4 piece results by scaling hypothesis:

| Hypothesis | Target | Sources | Cosine | Sign | Mag. ratio | Note |
|---|---:|---|---:|---:|---:|---|
| coordinate dilation, fixed amp | Tower II | 1 Round + 1 Tower + 2 Big | 0.7650 | 0.8333 | 1.8203 | moderate |
| coordinate dilation, fixed amp | Triangle II | 2 Round + 2 Big | 0.6861 | 0.9750 | 1.4935 | sign-strong |
| coordinate dilation, half amp | Tower II | 4 Round | 0.7144 | 0.7333 | 1.1111 | magnitude plausible |
| support-expanded | Tower II | 1 Round + 1 Tower + 2 Big | 0.8143 | 0.7538 | 1.7925 | best non-footprint Tower II |
| bilinear half amp | Tower II | 4 Round | 0.8217 | 0.7895 | 1.5595 | good shape, weaker sign |
| bilinear half amp | Big II | 4 Triangle | 0.6624 | 0.8611 | 1.0539 | moderate |
| 2x2 footprint aggregation | Round II | 4 Round | 0.9009 | 1.0000 | 4.0869 | strong shape, too strong |
| 2x2 footprint aggregation | Tower II | 3 Tower + 1 Triangle | 0.8754 | 0.9241 | 4.0711 | strong shape, too strong |
| 2x2 footprint aggregation | Triangle II | 4 Round | 0.8951 | 1.0000 | 2.0032 | strong, high magnitude |
| 2x2 footprint aggregation | Big II | 1 Round + 2 Big | 0.9559 | 0.8462 | 2.4534 | very high shape, high magnitude |

The 2x2-footprint model is the only scale hypothesis that produces very high
cosines across several targets with only 2-4 small pieces. However, its
magnitude ratios are inflated. This looks more like field focusing/aggregation
than clean scaled-kernel reproduction.

No robust recursive grammar was found. There is no evidence yet that
"configuration C of Scale-I pieces makes Piece-II, and the analogous C of
Scale-II pieces makes Piece-III" follows naturally from the current kernels.

## Diagnostic Local Signatures

Lower-dimensional signatures were tested for distinguishability:

| Signature | Samples | Max absolute pairwise cosine | Interpretation |
|---|---:|---:|---|
| ring 1 | 8 | 0.9701 | poor: Round and Triangle nearly identical |
| rings 1-2 | 24 | 0.9701 | still poor |
| axes only | 12 | 0.8889 | better but still conflates Round/Triangle |
| diagonals only | 12 | 1.0000 | unusable: Round and Triangle identical |
| full radius 3 | 48 | 0.9701 | full signature still sees Round/Triangle as close |

This is a major mathematical fact: with the current default components,
Triangle Hat's active field is essentially the Round Hat checkerboard family at
different strength/scaling. A local cellular-automata-style diagnostic can
distinguish Big Hat and Tower more easily than it can distinguish Round from
Triangle unless magnitude, center/home behavior, or inactive/alternate
components are brought into the signature.

Good diagnostic candidates should therefore include:

- sign pattern, but not sign pattern alone;
- magnitude ratio or fitted amplitude;
- one or two rings beyond nearest neighbors;
- possibly center/home response if the variant gives empty nucleation sites a
  principled way to evaluate it.

## Bootstrapping Populations

Using the strong-edge filter above, no single-species seed reached all four
piece types in this pass.

Single-species implications:

- Round-only seeds can reach Triangle, but not Tower.
- Triangle-only seeds reinforce Triangle and can help scaffold Big Hat at high
  counts, but do not reach Round or Tower.
- Tower-only seeds do not reach Round or Triangle.
- Big-only seeds do not reach Round, Tower, or Triangle.

Small mixed seeds that reached all four under the discovered graph:

```text
1 Round + 1 Tower
1 Round + 2 Tower
2 Round + 1 Tower
1 Round + 1 Tower + 1 Triangle
1 Round + 1 Tower + 1 Big
```

This suggests that Round and Tower are the most important starter species if
the goal is broad reachability. Round provides the clean Triangle pathway;
Tower participates in the stronger Big Hat scaffolds. Tower itself remains the
hard missing target, so a Wavefielder seed population may need to include
Towers unless a better Tower resonance is found.
