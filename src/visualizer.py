"""
EvoPrompt Optimizer - Visualization & Reporting Module

Generates:
- Accuracy trend charts (average & best per generation) as PNG
- Best prompt saves in both human-readable (.txt) and machine-readable (.json) formats
- Comprehensive summary report of the full evolution run

All outputs are written to the configured outputs directory.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

import config

# Configure module logger
logger = logging.getLogger(__name__)


# ─── Accuracy Curve Plot ─────────────────────────────────────────────────────

def plot_accuracy_curve(
    generation_stats: dict,
    output_path: Path | None = None,
) -> Path:
    """
    Plot average and best accuracy per generation as a line chart.

    The chart includes grid lines, legend, and is saved as a high-resolution PNG
    suitable for academic reports.

    Args:
        generation_stats: Dictionary with keys 'avg_fitness', 'max_fitness',
                          'min_fitness' (each a list of floats per generation).
        output_path: Destination file path. Defaults to outputs/accuracy_curve.png.

    Returns:
        Path: Path to the saved image file.
    """
    if output_path is None:
        output_path = config.OUTPUTS_DIR / "accuracy_curve.png"

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    generations = range(len(generation_stats["avg_fitness"]))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(generations, generation_stats["max_fitness"],
            label="Best Accuracy", marker="o", linewidth=2, color="#2196F3")
    ax.plot(generations, generation_stats["avg_fitness"],
            label="Average Accuracy", marker="s", linewidth=1.5,
            color="#FF9800", linestyle="--")
    ax.plot(generations, generation_stats.get("min_fitness", []),
            label="Worst Accuracy", marker="^", linewidth=1,
            color="#9E9E9E", linestyle=":")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("EvoPrompt Optimizer — Accuracy Over Generations", fontsize=14,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(generations)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"[VIZ] Accuracy curve saved to {output_path}")
    return output_path


# ─── Best Prompt Save ────────────────────────────────────────────────────────

def save_best_prompt(
    prompt: str,
    fitness: float,
    output_path_txt: Path | None = None,
    output_path_json: Path | None = None,
) -> dict:
    """
    Save the best prompt and its fitness score to disk.

    Creates both a human-readable .txt file and a structured .json file
    with metadata.

    Args:
        prompt: The best evolved prompt string.
        fitness: Accuracy score of the best prompt.
        output_path_txt: Path for text output. Defaults to outputs/best_prompt.txt.
        output_path_json: Path for JSON output. Defaults to outputs/best_prompt.json.

    Returns:
        dict: Metadata dictionary that was saved to JSON.
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path_txt is None:
        output_path_txt = config.OUTPUTS_DIR / "best_prompt.txt"
    if output_path_json is None:
        output_path_json = config.OUTPUTS_DIR / "best_prompt.json"

    timestamp = datetime.now().isoformat(timespec="seconds")

    # Text format
    with open(output_path_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EvoPrompt Optimizer — Best Prompt\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy: {fitness:.4f} ({fitness:.2%})\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {config.GROQ_MODEL}\n\n")
        f.write("-" * 60 + "\n")
        f.write("PROMPT:\n")
        f.write("-" * 60 + "\n")
        f.write(prompt + "\n")
        f.write("-" * 60 + "\n")

    # JSON format (for programmatic access)
    metadata = {
        "prompt": prompt,
        "accuracy": fitness,
        "model": config.GROQ_MODEL,
        "timestamp": timestamp,
        "config": {
            "population_size": config.POPULATION_SIZE,
            "generations": config.GENERATIONS,
            "mutation_probability": config.MUTATION_PROBABILITY,
            "crossover_probability": config.CROSSOVER_PROBABILITY,
            "mini_batch_size": config.MINI_BATCH_SIZE,
            "tournament_size": config.TOURNAMENT_SIZE,
        },
    }
    with open(output_path_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"[VIZ] Best prompt saved to {output_path_txt}")
    logger.info(f"[VIZ] Best prompt metadata saved to {output_path_json}")
    return metadata


# ─── Summary Report ──────────────────────────────────────────────────────────

def generate_report(
    history: dict,
    best_prompt: str,
    best_fitness: float,
    initial_fitness: float,
    test_accuracy: float,
    output_path: Path | None = None,
) -> Path:
    """
    Generate a comprehensive summary report of the evolution run.

    Includes evolution parameters, fitness progression, best prompt,
    and final test set performance. Suitable for academic submission.

    Args:
        history: Per-generation statistics from the evolution loop.
        best_prompt: The evolved prompt with highest fitness.
        best_fitness: Fitness (accuracy) of the best prompt on dev pool.
        initial_fitness: Fitness of the original human-written prompt.
        test_accuracy: Final accuracy on the held-out test set.
        output_path: Destination file path. Defaults to outputs/report.txt.

    Returns:
        Path: Path to the saved report file.
    """
    if output_path is None:
        output_path = config.OUTPUTS_DIR / "report.txt"

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    improvement = best_fitness - initial_fitness
    improvement_pct = (improvement / initial_fitness * 100) if initial_fitness > 0 else 0

    report_lines = [
        "=" * 70,
        "EvoPrompt Optimizer — Evolution Report",
        "Project HZ-2026-001 | Computational Intelligence Midterm",
        "=" * 70,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "--- Configuration ---",
        f"Model:              {config.GROQ_MODEL}",
        f"Population Size:    {config.POPULATION_SIZE}",
        f"Generations:        {config.GENERATIONS}",
        f"Mutation Prob:      {config.MUTATION_PROBABILITY:.2f}",
        f"Crossover Prob:     {config.CROSSOVER_PROBABILITY:.2f}",
        f"Tournament Size:    {config.TOURNAMENT_SIZE}",
        f"Mini-Batch Size:    {config.MINI_BATCH_SIZE}",
        f"API Call Delay:     {config.API_CALL_DELAY:.1f}s",
        "",
        "--- Results ---",
        f"Initial Accuracy:   {initial_fitness:.4f} ({initial_fitness:.2%})",
        f"Best Dev Accuracy:  {best_fitness:.4f} ({best_fitness:.2%})",
        f"Improvement:        {improvement:+.4f} ({improvement_pct:+.1f}%)",
        f"Test Set Accuracy:  {test_accuracy:.4f} ({test_accuracy:.2%})",
        "",
        "--- Fitness Progression ---",
    ]

    for gen in range(len(history["avg_fitness"])):
        report_lines.append(
            f"  Gen {gen:2d}:  avg={history['avg_fitness'][gen]:.4f}  "
            f"max={history['max_fitness'][gen]:.4f}  "
            f"min={history['min_fitness'][gen]:.4f}"
        )

    report_lines += [
        "",
        "--- Best Evolved Prompt ---",
        "-" * 70,
        best_prompt,
        "-" * 70,
        "",
        "--- Academic Notes ---",
        "This project demonstrates neuroevolution applied to LLM prompt",
        "optimization, following the approach of Sprig (April 2026) and",
        "GEPA (2025-2026). The LLM parameters remain frozen; only the",
        "natural language instruction evolves through genetic selection.",
        "",
        "References:",
        "  - Sprig: Genetic algorithm for system prompt optimization",
        "  - GEPA: Genetic Evolutionary Prompt Optimization (2025-2026)",
        "  - RoboPhd: LLM-guided evolutionary agent optimization (April 2026)",
        "",
        "=" * 70,
        "End of Report",
        "=" * 70,
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"[VIZ] Report saved to {output_path}")
    return output_path


# ─── Statistical CSV Exports ─────────────────────────────────────────────────

def save_generation_stats_csv(
    history: dict,
    output_path: Path | None = None,
) -> Path:
    """
    Save per-generation fitness statistics as a CSV file.

    Columns: generation, avg_fitness, max_fitness, min_fitness,
             improvement_from_initial, avg_change, max_change

    Args:
        history: Per-generation stats from the evolution loop.
        output_path: Destination file path. Defaults to outputs/generation_stats.csv.

    Returns:
        Path: Path to the saved CSV file.
    """
    if output_path is None:
        output_path = config.OUTPUTS_DIR / "generation_stats.csv"

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "avg_fitness",
            "max_fitness",
            "min_fitness",
            "avg_fitness_pct",
            "max_fitness_pct",
            "min_fitness_pct",
            "fitness_spread",
        ])

        for i in range(len(history["avg_fitness"])):
            avg = history["avg_fitness"][i]
            mx = history["max_fitness"][i]
            mn = history["min_fitness"][i]
            writer.writerow([
                i,
                f"{avg:.6f}",
                f"{mx:.6f}",
                f"{mn:.6f}",
                f"{avg:.2%}",
                f"{mx:.2%}",
                f"{mn:.2%}",
                f"{mx - mn:.6f}",
            ])

    logger.info(f"[VIZ] Generation stats CSV saved to {output_path}")
    return output_path


def save_experiment_summary_csv(
    history: dict,
    best_prompt: str,
    best_fitness: float,
    initial_fitness: float,
    test_accuracy: float,
    output_path: Path | None = None,
) -> Path:
    """
    Save a single-row CSV experiment summary with all key metrics.

    Useful for comparing multiple runs or appending to a larger dataset.

    Args:
        history: Per-generation stats.
        best_prompt: Best evolved prompt.
        best_fitness: Best fitness on dev pool.
        initial_fitness: Initial human prompt fitness.
        test_accuracy: Accuracy on held-out test set.
        output_path: Destination file path. Defaults to outputs/experiment_summary.csv.

    Returns:
        Path: Path to the saved CSV file.
    """
    if output_path is None:
        output_path = config.OUTPUTS_DIR / "experiment_summary.csv"

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    import csv

    avg_overall = sum(history["avg_fitness"]) / len(history["avg_fitness"])
    max_overall = max(history["max_fitness"])
    min_overall = min(history["min_fitness"])
    improvement = best_fitness - initial_fitness
    convergence_gen = None
    for i in range(len(history["max_fitness"])):
        # Convergence: when max stops improving for 2+ generations
        if i >= 2:
            if history["max_fitness"][i] == history["max_fitness"][i - 1]:
                convergence_gen = i
                break

    rows = [
        ["metric", "value"],
        ["timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["model", config.GROQ_MODEL],
        ["population_size", config.POPULATION_SIZE],
        ["generations", config.GENERATIONS],
        ["mutation_probability", config.MUTATION_PROBABILITY],
        ["crossover_probability", config.CROSSOVER_PROBABILITY],
        ["tournament_size", config.TOURNAMENT_SIZE],
        ["mini_batch_size", config.MINI_BATCH_SIZE],
        ["api_call_delay", config.API_CALL_DELAY],
        ["initial_accuracy", f"{initial_fitness:.6f}"],
        ["best_dev_accuracy", f"{best_fitness:.6f}"],
        ["test_accuracy", f"{test_accuracy:.6f}"],
        ["improvement_absolute", f"{improvement:.6f}"],
        ["improvement_percentage", f"{(improvement / initial_fitness * 100) if initial_fitness > 0 else 0:.2f}%"],
        ["avg_accuracy_overall", f"{avg_overall:.6f}"],
        ["max_accuracy_overall", f"{max_overall:.6f}"],
        ["min_accuracy_overall", f"{min_overall:.6f}"],
        ["final_generation_avg", f"{history['avg_fitness'][-1]:.6f}"],
        ["final_generation_max", f"{history['max_fitness'][-1]:.6f}"],
        ["convergence_generation", convergence_gen if convergence_gen is not None else "N/A"],
        ["total_api_calls_estimate", config.POPULATION_SIZE * config.GENERATIONS * config.MINI_BATCH_SIZE + config.MINI_BATCH_SIZE + len(history.get("test_titles", []))],
        ["best_prompt", best_prompt.replace("\n", " ")],
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    logger.info(f"[VIZ] Experiment summary CSV saved to {output_path}")
    return output_path


def save_population_analysis_csv(
    population: list,
    generation: int,
    output_path: Path | None = None,
) -> Path:
    """
    Save the final population's prompts and fitness scores as CSV.

    Useful for analyzing diversity and prompt variations.

    Args:
        population: Final DEAP population list.
        generation: Generation number this population is from.
        output_path: Destination file. Defaults to outputs/population_analysis.csv.

    Returns:
        Path: Path to the saved CSV file.
    """
    if output_path is None:
        output_path = config.OUTPUTS_DIR / "population_analysis.csv"

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    import csv

    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "fitness",
            "fitness_pct",
            "prompt_length",
            "prompt",
        ])

        for rank, ind in enumerate(sorted_pop, 1):
            writer.writerow([
                rank,
                f"{ind.fitness.values[0]:.6f}",
                f"{ind.fitness.values[0]:.2%}",
                len(str(ind)),
                str(ind).replace("\n", " "),
            ])

    logger.info(f"[VIZ] Population analysis CSV saved to {output_path}")
    return output_path
