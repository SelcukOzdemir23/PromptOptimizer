# EvoPrompt Optimizer

> **Project HZ-2026-001** — Computational Intelligence Midterm Project  
> **Neuroevolution-based LLM Prompt Optimization for News Classification**

Automatically optimizes natural language prompts for Large Language Models using genetic algorithms. The LLM's parameters remain frozen; only the system instruction evolves through selection, crossover, and mutation to maximize classification accuracy on the AG News dataset.

---

## Quick Start

```bash
# 1. Clone & enter project
cd PromptOptimizer

# 2. Activate virtual environment
source venv/bin/activate

# 3. Verify API connectivity
python tests/test_api_health.py

# 4. Run the full evolution pipeline
python src/main.py
```

**Prerequisites:**
- A valid Google Gemini API key in `.env` ([get one free](https://aistudio.google.com/app/apikey))
- Python 3.10+

---

## Configuration Variables Explained

All variables are read from the `.env` file. Below is a detailed explanation of each parameter, its effect, and how to tune it.

### 🔑 API & Model

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | _(required)_ | Your Google Gemini API key. Get it from [AI Studio](https://aistudio.google.com/app/apikey). |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | The Gemini model to use. Must be an available model name. |

#### GEMINI_MODEL — Choosing a Model

| Model | Speed | Cost | Best For |
|-------|-------|------|----------|
| `gemini-2.5-flash-lite` | ⚡ Fastest | 💰 Cheapest | **Default — best for this project** |
| `gemini-2.5-flash` | Fast | Low | Higher accuracy, slower evolution |
| `gemini-2.5-pro` | Slow | Higher | Maximum reasoning quality |
| `gemini-flash-latest` | Fast | Low | Auto-updates to newest Flash |

**Recommendation:** Use `gemini-2.5-flash-lite`. The evolution process makes hundreds of API calls; this model is fast, cheap, and sufficient for 4-class news classification.

---

### 🧬 Evolution Parameters

#### POPULATION_SIZE

```
POPULATION_SIZE=20
```

Number of candidate prompts in each generation.

| Value | Effect |
|-------|--------|
| **Low (5–10)** | Faster per generation, but risks premature convergence (all prompts become similar). May miss the optimal prompt. |
| **Default (20)** | Good balance. Enough diversity to explore different phrasings without excessive API calls. |
| **High (40–100)** | Better exploration, higher chance of finding an optimal prompt. **Much slower and more expensive** — each individual requires 50 API calls for fitness evaluation. |

**Trade-off:** `POPULATION_SIZE × GENERATIONS × MINI_BATCH_SIZE` = total API calls. A population of 20 over 10 generations with batch size 50 = ~10,000 calls minimum.

---

#### GENERATIONS

```
GENERATIONS=10
```

Number of evolution iterations (generations) to run.

| Value | Effect |
|-------|--------|
| **Low (3–5)** | Quick run, but the prompt may not have converged to its best form. You'll see early improvements but miss fine-tuning. |
| **Default (10)** | Enough for the prompt to stabilize. Accuracy typically plateaus around generation 6–8. |
| **High (20–50)** | Diminishing returns after ~15 generations. Useful for monitoring convergence behavior. Significantly increases cost. |

**Tip:** Watch the `accuracy_curve.png` output. If the curve is still rising at the last generation, consider increasing this value.

---

#### MUTATION_PROBABILITY

```
MUTATION_PROBABILITY=0.2
```

Probability (0.0–1.0) that an individual's prompt text will be mutated. Mutation replaces one word with a WordNet synonym.

| Value | Effect |
|-------|--------|
| **Low (0.05–0.1)** | Conservative. Prompts change slowly. Good if the initial prompt is already strong. Risk: slow exploration, may get stuck in a local optimum. |
| **Default (0.2)** | Balanced. Enough mutation to explore variations without destroying good prompts. |
| **High (0.4–0.8)** | Aggressive. Many words get swapped, creating very different prompts. Good for early exploration but can break well-performing prompts. |

**Analogy:** Think of mutation as "trying synonyms." Too little = never try new words. Too much = can't settle on a good sentence.

---

#### CROSSOVER_PROBABILITY

```
CROSSOVER_PROBABILITY=0.8
```

Probability (0.0–1.0) that two parent prompts will exchange parts to create offspring.

| Value | Effect |
|-------|--------|
| **Low (0.1–0.3)** | Rare recombination. Evolution relies mostly on mutation. Slower to combine good ideas from different prompts. |
| **Default (0.8)** | **Recommended.** High crossover lets the algorithm combine the best parts of two good prompts (e.g., "You are an expert classifier" + "Reply with only the category"). |
| **High (0.9–1.0)** | Almost always recombines. Can be effective but may break apart a perfectly good prompt structure. |

**Rule of thumb:** `CROSSOVER_PROBABILITY` should generally be higher than `MUTATION_PROBABILITY` (0.8 vs 0.2 is a classic ratio).

---

#### TOURNAMENT_SIZE

```
TOURNAMENT_SIZE=3
```

Number of individuals competing in each tournament selection. The best from the tournament is selected as a parent.

| Value | Effect |
|-------|--------|
| **Low (2)** | Weak selection pressure. Even mediocre prompts have a chance to reproduce. Maintains diversity but slower convergence. |
| **Default (3)** | Good balance. The best 1 out of 3 has a reasonable chance while still allowing decent prompts to participate. |
| **High (5–10)** | Strong selection pressure. Only the very best prompts reproduce. Fast convergence but risks losing diversity and getting stuck. |

---

#### MINI_BATCH_SIZE

```
MINI_BATCH_SIZE=50
```

Number of news headlines sampled from the development pool to evaluate each individual's fitness (accuracy).

| Value | Effect |
|-------|--------|
| **Low (10–20)** | Very fast per individual. But accuracy estimate is noisy — a prompt might score 80% on one batch and 40% on another. Leads to unreliable selection. |
| **Default (50)** | Reasonable accuracy estimate (~7% margin of error at 50% accuracy). Good speed/quality trade-off. |
| **High (100–500)** | More accurate fitness evaluation. Better selection decisions. **But** each individual requires more API calls, making evolution much slower and more expensive. |

**Impact on cost:** This is the **most important parameter for API usage**. At `MINI_BATCH_SIZE=50`, each individual costs 50 API calls. At `MINI_BATCH_SIZE=200`, it costs 4× more.

**Formula:** Total API calls ≈ `POPULATION_SIZE × GENERATIONS × MINI_BATCH_SIZE`  
With defaults: 20 × 10 × 50 = **~10,000 API calls**

---

### ⏱️ Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `API_CALL_DELAY` | `4.0` | Seconds to wait between each API call. Prevents hitting rate limits. |
| `API_MAX_RETRIES` | `3` | Number of retry attempts on transient errors (429, 500). |
| `API_BACKOFF_FACTOR` | `2.0` | Exponential backoff multiplier. Wait time = `max(API_CALL_DELAY, BACKOFF_FACTOR ^ attempt)`. |

#### Tuning Rate Limits

| Situation | Change |
|-----------|--------|
| **Getting 429 errors** | Increase `API_CALL_DELAY` to 6–8 seconds |
| **Too slow, no errors** | Decrease `API_CALL_DELAY` to 2–3 seconds (if your quota allows) |
| **Frequent transient failures** | Increase `API_MAX_RETRIES` to 5 |

**Warning:** The free tier of Gemini API has strict rate limits (15 RPM for Flash-Lite). With `API_CALL_DELAY=4.0`, you make ~15 calls/minute, which is at the limit. Do not decrease below 4.0 on the free tier.

---

### 📊 Data Split

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_SPLIT_RATIO` | `0.5` | Fraction of the test dataset held out as final test set. The rest becomes the development pool. |
| `RANDOM_SEED` | `42` | Seed for reproducible data splitting and evolution. |

#### TEST_SPLIT_RATIO

| Value | Effect |
|-------|--------|
| **Low (0.2–0.3)** | Large development pool (more data for evolution), small test set (less reliable final evaluation). |
| **Default (0.5)** | Equal split: 3,800 for evolution, 3,800 for final test. Balanced. |
| **High (0.7–0.8)** | Small development pool, large test set. More reliable final evaluation but less data to evolve on. |

---

## 📁 Project Structure

```
PromptOptimizer/
├── src/
│   ├── config.py            # Central configuration (reads .env)
│   ├── data_loader.py       # AG News loading, splitting, caching
│   ├── llm_interface.py     # Gemini API wrapper (google-genai SDK)
│   ├── evolution.py         # DEAP genetic algorithm engine
│   ├── visualizer.py        # Charts, reports, prompt saving
│   └── main.py              # Full pipeline orchestrator
├── tests/
│   ├── test_api_health.py   # API connectivity & health check
│   └── test_e2e.py          # 23 unit tests for all modules
├── data/
│   ├── raw/                 # (reserved for raw data)
│   └── processed/           # Cached dev_pool.csv and test_set.csv
├── dataset/                 # Source AG News CSV files
├── outputs/                 # Generated results (created at runtime)
│   ├── accuracy_curve.png   # Fitness progression chart
│   ├── best_prompt.txt      # Best evolved prompt (human-readable)
│   ├── best_prompt.json     # Best prompt with metadata
│   └── report.txt           # Full evolution report
├── .env                     # Your configuration (DO NOT commit)
├── .env.example             # Template for .env
├── .gitignore
├── requirements.txt
├── venv/                    # Python virtual environment
└── README.md
```

---

## 🧪 Testing

```bash
# Activate virtual environment
source venv/bin/activate

# Run all unit tests (23 tests)
pytest tests/test_e2e.py -v

# Run API health check
python tests/test_api_health.py

# Run tests with coverage
pytest tests/ -v --cov=src
```

---

## 🚀 Running the Pipeline

```bash
source venv/bin/activate
python src/main.py
```

### What Happens

1. **Config validation** — checks API key and parameters
2. **Data loading** — loads AG News, splits into dev pool + test set
3. **Initial prompt evaluation** — measures accuracy of the human-written seed prompt
4. **Evolution** — runs the genetic algorithm for the configured number of generations
5. **Test evaluation** — evaluates the best evolved prompt on the held-out test set
6. **Output generation** — creates accuracy chart, saves best prompt, writes report

### Expected Runtime

With default parameters (population=20, generations=10, batch=50, delay=4s):
- **Per individual:** ~50 API calls × 4s = ~200s (3.3 minutes)
- **Per generation:** 20 individuals × 3.3 min = ~66 minutes
- **Total:** ~10 generations × 66 min = **~11 hours**

**To speed up:**
- Reduce `POPULATION_SIZE` to 10 → ~5.5 hours
- Reduce `MINI_BATCH_SIZE` to 20 → ~4.5 hours
- Reduce `GENERATIONS` to 5 → ~5.5 hours
- Reduce `API_CALL_DELAY` to 2s (if quota allows) → ~5.5 hours

---

## 📚 Academic References

This project is inspired by:

- **Sprig (April 2026)** — Genetic algorithm for system prompt optimization via editing operations
- **GEPA (2025–2026)** — Genetic Evolutionary Prompt Optimization, showing GAs compete with RL methods
- **RoboPhd (April 2026)** — LLM-guided evolutionary agent optimization

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM API | Google Gemini (new `google-genai` SDK ≥ 1.0.0) |
| Model | Gemini 2.5 Flash-Lite |
| Genetic Algorithm | DEAP 1.4+ |
| Synonym Lookup | NLTK WordNet |
| Data Processing | Pandas, scikit-learn |
| Visualization | Matplotlib |
| Testing | pytest |

---

## ⚠️ Notes

- **API Key Quota:** The free tier has daily/monthly limits. If you hit 429 errors, wait or enable billing.
- **Cost:** With default settings, expect ~10,000 API calls. On the free tier, this is within limits but close.
- **Reproducibility:** Set `RANDOM_SEED` to get the same data split each run. Evolution involves randomness, so results may vary.
- **.env file:** Never commit your `.env` file. Use `.env.example` as a template.
