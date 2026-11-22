# Model equality (33 chat endpoints)

## The task

- **Endpoints.** 33 text-only chat endpoints, ep-01 to ep-33: no logprobs; `temperature` and `seed` are ignored, system messages are rejected, the native `tools` field is silently stripped.
- **Budget.** 100 answers per endpoint, at most 4 parallel requests, at most 256 output tokens per answer.
- **Disguises vs variants.** An endpoint may serve a model behind a disguise (provider system prompt, persona, decoding watermark, decoding strategy such as temperature 1.0): that is still the *same* model. A different base model, an fp8 quantization or a LoRA adapter is a *different* model.
- **Labels.** One line per endpoint, `endpoint,family,model`: same family letter = same base model family; model = letter + number (A1, A2, …), same label = exactly the same model.
- **Score.** 0.1 × MCC(family, all pairs) + 0.9 × MCC(model, pairs within the true family), each clipped at 0. Public leaderboard on a subset of the pairs, final ranking on the rest; one submission per 30 minutes.
- **Practice.** ref-01 to ref-09, free, truth in `practice_answers.csv`: one small model as 5 disguised copies (plain, system prompt, persona, watermark, temperature 1.0) and 4 variants (fp8, LoRA 0.3 / 0.7 / 1.5).
- **References in the task description.** MET (Gao et al., ICLR 2025, Hamming-kernel MMD), RUT (Zhu et al., ICLR 2026, rank test against a reference model), AgentProv (tool-choice probes).

## Layout

| Path | Contents |
|---|---|
| `task_template.py`, `query_reference.py`, `score_practice.py`, `submission.py`, `pyproject.toml`, `uv.lock`, `src/` | Organisers' starter files. `submission.py` and the collectors read `API_KEY` from the environment. |
| `practice_answers.csv`, `my_practice_answers.csv`, `reference_prompts.txt` | Practice truth, a practice guess, the first hand-written probe prompts. |
| `probes/probes.py` | Probe definitions: short questions (F, V), random sequences (R), knowledge (K), stories (G) and more. |
| `probes/collect.py` | Collector with resume, 4 parallel requests and a hard budget guard (`--go --approve-budget N`). |
| `probes/analyze.py` | Per-probe pair tests with calibrated permutation z, practice scoring, the submission format check. |
| `probes/combine.py` | Combines the channels (self-description, style, length, tools) into families and models: reliability gate, `--tool-stat freq|shape`, `--diagnose`. |
| `tool_probe/tool_probe.py` | Tool-choice probe adapted from AgentProv to plain text (12 items × 4 equivalent function names, few-shot). |
| `extra_probes/preference_probes.py` | Preference, forced-choice, continuation and arithmetic probes. |
| `extra_probes/number_fingerprint.py`, `extra_probes/native_tools.py`, `extra_probes/api_params.py` | Number fingerprint with repeats; native-tools and parameter checks of the API. |
| `reference_models/` | Local reference-model likelihoods (RUT-style, inference only): `setup.sh` (own venv, Qwen2.5 0.5B/1.5B/3B), `get_llama.sh`, `score.py` (log-likelihood of every saved answer under a local model, optional fp8), Slurm jobs. |
| `analysis/` | Offline analyses of the saved answers (see below). They expect two copies of this folder next to them: `data/` with the practice run and `eprun/` with the scored run. |
| `submissions/` | All submission files (ours, and teammates' as `team_*.csv`). |
| `results/` | Probe weights, practice guesses and the round-2 plan. |

## Results

**Families: solved with local reference models.** Each endpoint's saved answers were scored under real checkpoints (`reference_models/score.py`, inference only). Every family fits one base model far better than the others (tool answers, mean log-prob per token, closer to 0 = better):

| Family | Base model | Fit under it | Endpoints (11 each) |
|---|---|---|---|
| A | Qwen2.5-3B-Instruct | −0.06 (others −0.30 … −0.56) | 01 02 04 12 14 17 18 23 24 25 33 |
| B | Qwen2.5-1.5B-Instruct | −0.11 (others −0.33 … −0.66) | 03 06 07 **08** 20 21 22 26 28 30 32 |
| C | a Llama model, closest to Llama-3.2-1B-Instruct | −0.24 (Qwen −0.61 … −1.02) | 05 09 10 11 13 15 16 19 27 29 31 |

ep-08 (persona "made by Anthropic") was in A by writing style, but its answers fit Qwen2.5-1.5B (−0.10) and not 3B (−0.55); moved to B, every family has 11 endpoints. Earlier, self-description plus writing style had given the same split except for ep-08; a test submission with A and B merged dropped the score (0.05), confirming A ≠ B.

**Models within a family.** Until the reference models, every label set scored 0.10 on the public pairs (full family marks, model part ≤ 0). Moving ep-08 into B and grouping it with the other disguised copies of the 1.5B model gave **0.45** (`submission_ref.csv`).

Within a family, the tool-choice temperature fitted against the base model's own option probabilities (`reference_models/tools_fit.py`, `analysis/analyze_toolfit.py`) separates endpoints: sharp ones behave like the base at default decoding, ep-28 is far out (T 2.0, picks the base's favourite only 55 % of the time).

| Family | Sharp (T ≤ 0.78) | Flatter | Far out |
|---|---|---|---|
| A | 01 0.62, 12 0.65, 17/24/25 0.73, 04 0.78 | 18 0.88, 14 0.98, 33 1.02, 02 1.06, 23 1.28 | — |
| B | 32 0.67, 06 0.70 | 07/08/21 0.88, 30 0.91, 20/22/26 0.95, 03 1.10 | 28 2.02 |

**A fixed rule** (`analysis/rule.py`, the same for every family, no hand-picking): base-model group = sharp endpoints (T within 0.15 of the family's sharpest) + disguise marks (other or no maker claimed) + the least self-consistent endpoint (temperature copy); everyone else is its own model. On practice, with the practice base SmolLM2-360M-Instruct as reference, it scores **model MCC 0.721** (6 right "same" calls, 0 wrong), against 0.49 for the tool probe alone. Applied unchanged to the scored set: `submission_rule.csv`.

| Submission (`submissions/`) | Labels | Public |
|---|---|---|
| `submission_combined.csv` | 3 families, almost all "same" within each | 0.10 |
| `submission_v2.csv` | plus 03 and 21 split off in B | 0.10 |
| `test_fam_AB_merged.csv` | families A and B merged (test) | 0.05 |
| `submission_final_10pairs.csv` | mostly "different", copy groups A1 {02, 08, 23}, B1 {20, 30}, C1 {05, 11, 29, 31} | 0.10 |
| `submission_final.csv` | plus A2 {01, 18, 24} | 0.10 |
| `test_B_only.csv` | B core together, everything else separate | 0.10 |
| `submission_ref.csv` | reference-model families (ep-08 in B); A1 {02, 23, 33}, A2 {01, 18, 24}, B1 {08, 20, 30}, C1 {05, 11, 29, 31} | **0.45** |
| `submission_rule.csv` | the fixed rule: A1 {01 02 12 17 24 25 33}, B1 {06 08 20 30 32}, C1 {11 31}, rest separate | not yet submitted |

## Methods tried

"Practice" = checked against the practice truth; "scored" = truth-free checks on the 33 endpoints.

1. **Length test** (paired per-prompt log length, calibrated permutation z). Reproducible on practice, noise on the scored endpoints. Dropped.
2. **Tool probe** (`tool_probe.py`). Calibrated and reproducible; on practice it separates only the strong LoRAs (model MCC 0.49). On the scored set the clear outliers (08, 28, 05, 11) are mostly the endpoints with odd self-descriptions, i.e. it reacts to disguises. Llama-style endpoints often continue the few-shot numbering, so 30–50 % of family C's answers can't be read.
3. **Temperature-robust "shape" statistic** (`--tool-stat shape`). No better than frequencies.
4. **Directional test** (`analysis/anchor.py`: project each endpoint onto the line toward the strongest outlier). Practice: LoRA 0.3 rises with more answers, copies stay flat. Scored: in B, ep-03 rose from z 2.4 to 4.9 between 20 and 40 answers, as predicted; ep-21 stayed near 3.
5. **Knowledge test** (100 one-word questions per request). Failed its pre-set practice check. Dropped.
6. **Preference, forced-choice, continuation, arithmetic probes** (`preference_probes.py`). No signal on practice.
7. **MET Hamming-kernel MMD** on practice probes (`analysis/met_practice.py`): disguises dominate. On the MET paper's own Llama-3-8B data (`analysis/met_power.py`, data read with `met_zip.py`): at 40 samples, int8 and nf4 are detected only 2–10 % of the time, so quantization is essentially invisible in text at this budget.
8. **Text similarity / embeddings** (character n-gram TF-IDF). Practice copies are as far apart as copies and variants. No use.
9. **Formatting habits**, **mutual nearest neighbours** (`mnn.py`), **supervised pair classifier** on the 36 practice pairs (`paircls.py`, leave-one-out MCC 0.05), **rare shared phrases** (`rare.py`, backwards on practice). No skill.
10. **Random-sequence openings.** In A, 01, 18 and 24 share the opening of their "30 random numbers" (pair p ≈ 0.007–0.02). Used as group A2.
11. **Disguise marks** (`disguise.py`). On practice each disguise leaves a mark (system prompt: different claimed maker; persona: no maker; temperature: low self-consistency; watermark: odd tool answers). Scored candidates: A 02, 08, 23; B 20, 30, 28; C 05, 11, 29, 31.
12. **Twins** (`twins.py`: closest core partner of each disguised endpoint by tool choices, split-half stability). Only 28 ↔ 03 is stable (82 %).
13. **API checks.** `/models` and answer metadata carry nothing useful; `temperature`, `seed` and native tools have no effect.
14. **Local reference models** (`reference_models/`, `analysis/analyze_ll.py`). Log-likelihood of every saved answer under Qwen2.5 0.5B / 1.5B / 3B and Llama-3.2 1B / 3B: identifies each family's base model and moved ep-08 to B (see Results). This is what lifted the public score from 0.10 to 0.45.
15. **fp8 check** (`score_fp8.sbatch`, `analysis/analyze_fp8.py`: same scoring with a self-made fp8 copy). Inconclusive: every endpoint fits the fp8 copy slightly worse, none stands out.
16. **Tool-choice temperature** (`tools_fit.py`, `analysis/analyze_toolfit.py`): the base model's probability for each of the 4 equivalent names, conditioned on the endpoint's own text, then a fitted temperature per endpoint. Separates sharp (base-like) endpoints from flatter ones and from ep-28.
17. **Fixed rule** (`analysis/rule.py`), validated on practice (model MCC 0.721) before being applied to the scored set.

## How to run

```bash
uv sync                                         # or: pip install -r requirements.txt
read -s API_KEY; export API_KEY
python probes/collect.py --target ref --repeats 5 --go                # practice endpoints are free
python probes/collect.py --target ep --repeats 2 --go --approve-budget 40   # budget = new answers per endpoint
python tool_probe/tool_probe.py collect --target ep --n 10 --go --approve-budget 10
python probes/combine.py --target ep --diagnose
(cd reference_models && bash setup.sh && sbatch score.sbatch)                     # local likelihoods (no API calls)
python analysis/analyze_ll.py reference_models/out                  # from model_equality/
(cd reference_models && sbatch toolfit.sbatch)                      # tool-choice probabilities, families A and B
(cd reference_models && bash get_smol.sh && sbatch practice.sbatch) # same pipeline on practice (SmolLM2)
python analysis/rule.py ref 360M                                    # practice check of the fixed rule
python analysis/rule.py ep                                          # writes submissions/submission_rule.csv
```

Every collector refuses to spend scored queries without an explicit `--approve-budget`.
