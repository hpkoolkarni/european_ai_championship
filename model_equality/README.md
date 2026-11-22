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
| `me/me_probes.py` | Probe definitions: short questions (F, V), random sequences (R), knowledge (K), stories (G) and more. |
| `me/me_collect.py` | Collector with resume, 4 parallel requests and a hard budget guard (`--go --approve-budget N`). |
| `me/me_analyze.py` | Per-probe pair tests with calibrated permutation z, practice scoring, the submission format check. |
| `me/me_combine.py` | Combines the channels (self-description, style, length, tools) into families and models: reliability gate, `--tool-stat freq|shape`, `--diagnose`. |
| `me2/me2_tools.py` | Tool-choice probe adapted from AgentProv to plain text (12 items × 4 equivalent function names, few-shot). |
| `me3/me3.py` | Preference, forced-choice, continuation and arithmetic probes. |
| `me3/nums.py`, `me3/native_tools.py`, `me3/params.py` | Number fingerprint with repeats; native-tools and parameter checks of the API. |
| `me4/` | Local reference-model likelihoods (RUT-style, inference only): `setup.sh` (own venv, Qwen2.5 0.5B/1.5B/3B), `get_llama.sh`, `score.py` (log-likelihood of every saved answer under a local model, optional fp8), Slurm jobs. |
| `analysis/` | Offline analyses of the saved answers (see below). They expect two copies of this folder next to them: `data/` with the practice run and `eprun/` with the scored run. |
| `me/out/` | Our submissions and probe weights. `submissions/` holds teammates' submissions. |

## Results

**Families: solved.** Self-description plus writing style give three families; tool choices confirm them (median pair z ≥ 5.8 between every two families, 98 % of cross pairs above the cut). A test submission with two families merged dropped the score, which confirms the split.

| Family | Endpoints |
|---|---|
| A (Qwen-style) | 01 02 04 08 12 14 17 18 23 24 25 33 |
| B (Qwen-style) | 03 06 07 20 21 22 26 28 30 32 |
| C (Llama-style) | 05 09 10 11 13 15 16 19 27 29 31 |

**Models within a family: not solved.** Every label set we submitted (mostly "same", mostly "different", groups from the directional test, groups from random-sequence openings) scored 0.10 on the public pairs, i.e. full family marks and a model part of 0 or below.

| Submission (`me/out/`) | Labels | Public |
|---|---|---|
| `submission_combined.csv` | 3 families, almost all "same" within each | 0.10 |
| `submission_v2.csv` | plus 03 and 21 split off in B | 0.10 |
| `test_fam_AB_merged.csv` | families A and B merged (test) | 0.05 |
| `submission_final_10pairs.csv` | mostly "different", copy groups A1 {02, 08, 23}, B1 {20, 30}, C1 {05, 11, 29, 31} | 0.10 |
| `submission_final.csv` | plus A2 {01, 18, 24} | 0.10 |
| `test_B_only.csv` | B core together, everything else separate | 0.10 |

## Methods tried

"Practice" = checked against the practice truth; "scored" = truth-free checks on the 33 endpoints.

1. **Length test** (paired per-prompt log length, calibrated permutation z). Reproducible on practice, noise on the scored endpoints. Dropped.
2. **Tool probe** (`me2_tools.py`). Calibrated and reproducible; on practice it separates only the strong LoRAs (model MCC 0.49). On the scored set the clear outliers (08, 28, 05, 11) are mostly the endpoints with odd self-descriptions, i.e. it reacts to disguises. Llama-style endpoints often continue the few-shot numbering, so 30–50 % of family C's answers can't be read.
3. **Temperature-robust "shape" statistic** (`--tool-stat shape`). No better than frequencies.
4. **Directional test** (`analysis/anchor.py`: project each endpoint onto the line toward the strongest outlier). Practice: LoRA 0.3 rises with more answers, copies stay flat. Scored: in B, ep-03 rose from z 2.4 to 4.9 between 20 and 40 answers, as predicted; ep-21 stayed near 3.
5. **Knowledge test** (100 one-word questions per request). Failed its pre-set practice check. Dropped.
6. **Preference, forced-choice, continuation, arithmetic probes** (`me3.py`). No signal on practice.
7. **MET Hamming-kernel MMD** on practice probes (`analysis/met_practice.py`): disguises dominate. On the MET paper's own Llama-3-8B data (`analysis/met_power.py`, data read with `met_zip.py`): at 40 samples, int8 and nf4 are detected only 2–10 % of the time, so quantization is essentially invisible in text at this budget.
8. **Text similarity / embeddings** (character n-gram TF-IDF). Practice copies are as far apart as copies and variants. No use.
9. **Formatting habits**, **mutual nearest neighbours** (`mnn.py`), **supervised pair classifier** on the 36 practice pairs (`paircls.py`, leave-one-out MCC 0.05), **rare shared phrases** (`rare.py`, backwards on practice). No skill.
10. **Random-sequence openings.** In A, 01, 18 and 24 share the opening of their "30 random numbers" (pair p ≈ 0.007–0.02). Used as group A2.
11. **Disguise marks** (`disguise.py`). On practice each disguise leaves a mark (system prompt: different claimed maker; persona: no maker; temperature: low self-consistency; watermark: odd tool answers). Scored candidates: A 02, 08, 23; B 20, 30, 28; C 05, 11, 29, 31.
12. **Twins** (`twins.py`: closest core partner of each disguised endpoint by tool choices, split-half stability). Only 28 ↔ 03 is stable (82 %).
13. **API checks.** `/models` and answer metadata carry nothing useful; `temperature`, `seed` and native tools have no effect.
14. **Local reference models** (`me4/`, `analysis/analyze_ll.py`, `analysis/analyze_fp8.py`). Log-likelihood of every saved answer under Qwen2.5 0.5B / 1.5B / 3B (and Llama-3.2 1B / 3B, fp8 copies), to find different base sizes inside a family and fp8 variants.

## How to run

```bash
uv sync                                         # or: pip install -r requirements.txt
read -s API_KEY; export API_KEY
python me/me_collect.py --target ref --repeats 5 --go                # practice endpoints are free
python me/me_collect.py --target ep --repeats 2 --go --approve-budget 40   # budget = new answers per endpoint
python me2/me2_tools.py collect --target ep --n 10 --go --approve-budget 10
python me/me_combine.py --target ep --diagnose
(cd me4 && bash setup.sh && sbatch score.sbatch)                     # local likelihoods (no API calls)
python analysis/analyze_ll.py me4/out                  # from model_equality/
```

Every collector refuses to spend scored queries without an explicit `--approve-budget`.
