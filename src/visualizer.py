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
        f.write(f"Model: {config.GEMINI_MODEL}\n\n")
        f.write("-" * 60 + "\n")
        f.write("PROMPT:\n")
        f.write("-" * 60 + "\n")
        f.write(prompt + "\n")
        f.write("-" * 60 + "\n")

    # JSON format (for programmatic access)
    metadata = {
        "prompt": prompt,
        "accuracy": fitness,
        "model": config.GEMINI_MODEL,
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
        f"Model:              {config.GEMINI_MODEL}",
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
