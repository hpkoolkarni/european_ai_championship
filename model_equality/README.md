# Model equality (33 chat endpoints)

## The task

- **Endpoints.** 33 text-only chat endpoints, ep-01 to ep-33: no logprobs; `temperature` and `seed` are ignored, system messages are rejected, the native `tools` field is silently stripped.
- **Budget.** 100 answers per endpoint, at most 4 parallel requests, at most 256 output tokens per answer.
- **Disguises vs variants.** An endpoint may serve a model behind a disguise (provider system prompt, persona, decoding watermark, decoding strategy such as temperature 1.0): that is still the *same* model. A different base model, a quantization configuration or a LoRA adapter is a *different* model.
- **Labels.** One line per endpoint, `endpoint,family,model`: same family letter = same base model family; model = letter + number (A1, A2, …), same label = exactly the same model.
- **Score.** 0.1 × MCC(family, all pairs) + 0.9 × MCC(model, pairs within the true family), each clipped at 0. Public leaderboard on a subset of the pairs, final ranking on the rest; one submission per 30 minutes.
- **Practice.** ref-01 to ref-09, free, truth in `practice_answers.csv`: one small model (SmolLM2-360M-Instruct) as 5 disguised copies (plain, system prompt, persona, watermark, temperature 1.0) and 4 variants (fp8, LoRA 0.3 / 0.7 / 1.5).
- **References in the task description.** MET (Gao et al., ICLR 2025), RUT (Zhu et al., ICLR 2026, rank test against a reference model), AgentProv (tool-choice probes).

## Layout

| Path | Contents |
|---|---|
| `task_template.py`, `query_reference.py`, `score_practice.py`, `submission.py`, `pyproject.toml`, `uv.lock`, `src/` | Organisers' starter files. `submission.py` and the collectors read `API_KEY` from the environment. |
| `practice_answers.csv`, `reference_prompts.txt` | Practice truth, the first hand-written probe prompts. |
| `probes/` | Text probes (`probes.py`), collector with resume and a hard budget guard (`collect.py`), pair tests and the submission format check (`analyze.py`), channel combiner (`combine.py`). |
| `tool_probe/tool_probe.py` | Tool-choice probe adapted from AgentProv to plain text (12 items × 4 equivalent function names, few-shot). |
| `reference_models/` | Local reference models, inference only: `setup.sh` (own venv, Qwen2.5 0.5B/1.5B/3B), `get_llama.sh`, `get_smol.sh`, `score.py` (log-likelihood of every saved answer), `tools_fit.py` (base-model probability of every tool choice, optional simulated fp8), Slurm jobs. |
| `analysis/` | Offline analyses: `analyze_ll.py` (families), `analyze_toolfit.py` (tool-choice temperature), `residual.py` (calibrated residual test against the exact base), `fp8_toolfit.py`, `rule.py`, `disguise.py` (disguise marks), `collide.py` (shared answer openings). |
| `submissions/` | Every scored submission (ours, and teammates' as `team_*.csv`). |

## Results

**Families: solved with local reference models.** Each endpoint's saved answers were scored under real checkpoints (`reference_models/score.py`). Every family fits one base model far better than the others; on all saved answers the fit of each family to its base is on the same scale, so each is the exact base:

| Family | Base model | Fit, all answers (mean log-prob per token) | Endpoints (11 each) |
|---|---|---|---|
| A | Qwen2.5-3B-Instruct | −0.29 … −0.58 | 01 02 04 12 14 17 18 23 24 25 33 |
| B | Qwen2.5-1.5B-Instruct | −0.34 … −0.61 | 03 06 07 **08** 20 21 22 26 28 30 32 |
| C | Llama-3.2-1B-Instruct | −0.33 … −0.65 (Qwen −0.8 … −1.5) | 05 09 10 11 13 15 16 19 27 29 31 |

ep-08 (persona "made by Anthropic") was in A by writing style, but its answers fit Qwen2.5-1.5B and not 3B; moved to B. A test with A and B merged dropped the score (0.05), confirming A ≠ B.

**Models within a family: still the open problem.** Best public score **0.45** (`submission_ref.csv`): small groups of disguised copies, everything else its own model.

**Residual test against the exact base** (`reference_models/tools_fit.py`, `analysis/residual.py`). For each tool answer, the base model's probability of each of the 4 equivalent names given the endpoint's own text; per endpoint a temperature, then observed vs expected choice counts, with the family's shared offset removed and calibrated by 200 simulated answer sets (54 tool answers per endpoint):

| Family | Off the base (z) | Consistent with the base |
|---|---|---|
| A | 04 (9.7), 23 (6.8), 02 (3.4), 01 (3.1) | 12 14 17 18 24 25 33 |
| B | 28 (23), 06 (9.8), 03 (5.5), 21 (4.6), 08 (3.8), 22 (3.7) | 07 20 26 30 32 |
| C | 10 (27), 05 (10), 11 (6.6) | 09 13 15 16 19 27 29 31 |

- In B, 28, 03 and 21 share one deviation pattern (correlation 0.52–0.73) with graded size, like one LoRA at three strengths.
- A provider system prompt or persona also moves tool choices: in A, only 02 and 23 do not say "I am Qwen" (the default system prompt), and their deviations correlate 0.70; in C, 11 claims another maker and 05 shares its pattern (0.62).
- **fp8** (`analysis/fp8_toolfit.py`): a simulated fp8 copy (per row or per tensor) moves the tool-choice probabilities by 13–16 %, but no endpoint fits either simulation better than bf16, so the server's quantization is not reproduced; a real fp8 copy would look only slightly off-base (simulation: chi/dof 0.9–1.2 against 0.4–0.7 for exact copies).

**What the public scores say.** The more endpoints are merged into one "base" model, the lower the score: 15 same-model pairs 0.45, 19 pairs 0.20, 32 pairs 0.40, 96 pairs 0.10. Endpoints that look like the base in the tool test are not one model; the true same-model groups are small, and family A has at least two groups with several endpoints (merging them cost 0.25). Practice, with a single multi-endpoint model, does not show this structure.

| Submission (`submissions/`) | Labels | Public |
|---|---|---|
| `submission_combined.csv` | 3 families, almost all "same" within each | 0.10 |
| `submission_v2.csv` | plus 03 and 21 split off in B | 0.10 |
| `test_fam_AB_merged.csv` | families A and B merged (test) | 0.05 |
| `submission_final_10pairs.csv` | mostly "different", copy groups A1 {02, 08, 23}, B1 {20, 30}, C1 {05, 11, 29, 31} | 0.10 |
| `submission_final.csv` | plus A2 {01, 18, 24} | 0.10 |
| `test_B_only.csv` | B core together, everything else separate | 0.10 |
| `team_submission_S6.csv`, `team_submission_S7.csv`, `team_submission_final.csv` | teammates' label sets | 0.10 |
| **`submission_ref.csv`** | reference-model families (ep-08 in B); A1 {02, 23, 33}, A2 {01, 18, 24}, B1 {08, 20, 30}, C1 {05, 11, 29, 31} | **0.45** |
| `submission_rule.csv` | fixed rule (`analysis/rule.py`, practice model MCC 0.72): A1 {01 02 12 17 24 25 33}, B1 {06 08 20 30 32}, C1 {11 31} | 0.40 |
| `submission_v3.csv` | residual test: one base model per family (A 9, B 6, C 10 endpoints), off-base endpoints separate | 0.10 |
| `submission_v4.csv` | `submission_ref` with A1 and A2 merged into {02, 18, 23, 24, 33}, 01 separate | 0.20 |

## Methods tried

"Practice" = checked against the practice truth; "scored" = truth-free checks on the 33 endpoints.

1. **Length test** (paired per-prompt log length, calibrated permutation z). Reproducible on practice, noise on the scored endpoints.
2. **Tool probe** (`tool_probe.py`). Calibrated and reproducible; on practice it separates only the strong LoRAs (model MCC 0.49). Practice copies mostly repeat the few-shot example, so only 12–28 of their tool items can be read.
3. **Knowledge, preference, forced-choice, continuation and arithmetic probes.** No signal on practice.
4. **MET Hamming-kernel MMD** on practice probes: disguises dominate. On the MET paper's own Llama-3-8B data, at 40 samples int8 and nf4 are detected only 2–10 % of the time.
5. **Text similarity, formatting habits, nearest neighbours, a supervised pair classifier on the practice pairs, rare shared phrases.** No skill.
6. **Shared answer openings** (`collide.py`). In A, 01, 18 and 24 share the opening of their "30 random numbers" (pair p ≈ 0.007–0.02): group A2. Over all 37 prompts the best pairs are barely above a typical pair.
7. **Disguise marks** (`disguise.py`). Claimed maker, tool-choice entropy, self-consistency: candidates A 02, 08, 23; B 20, 30, 28; C 05, 11, 29, 31.
8. **API checks.** `/models` and answer metadata carry nothing useful; `temperature`, `seed` and native tools have no effect.
9. **Local reference models** (`score.py`, `analyze_ll.py`). Identify each family's exact base; lifted the score from 0.10 to 0.45.
10. **Tool-choice temperature** (`analyze_toolfit.py`) and the **fixed rule** (`rule.py`), validated on practice before use.
11. **Residual test against the exact base** (`residual.py`) and **simulated fp8** (`fp8_toolfit.py`): see Results.

## How to run

```bash
uv sync                                         # or: pip install -r requirements.txt
read -s API_KEY; export API_KEY
python probes/collect.py --target ref --repeats 5 --go                      # practice endpoints are free
python probes/collect.py --target ep --repeats 2 --go --approve-budget 40   # budget = new answers per endpoint
python tool_probe/tool_probe.py collect --target ep --n 54 --go --approve-budget 14
(cd reference_models && bash setup.sh && bash get_llama.sh && sbatch score.sbatch)   # local likelihoods, no API calls
python analysis/analyze_ll.py reference_models/out
(cd reference_models && sbatch toolfit.sbatch)                      # tool-choice probabilities, all three families
(cd reference_models && FP8=t sbatch toolfit.sbatch)                # the same under a simulated fp8 copy
python analysis/residual.py ABC                                     # calibrated residual test
python analysis/fp8_toolfit.py fp8t
python analysis/rule.py ep                                          # writes submissions/submission_rule.csv
```

Every collector refuses to spend scored queries without an explicit `--approve-budget`.
