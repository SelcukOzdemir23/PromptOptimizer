"""
EvoPrompt Optimizer - Main Orchestration

Entry point for the application. Coordinates all modules to run the
full EvoPrompt optimization pipeline:

1. Load & preprocess data
2. Evaluate the human-written initial prompt
3. Run genetic algorithm evolution
4. Evaluate the best evolved prompt on the held-out test set
5. Generate visualization and summary report

Usage:
    cd /home/sozdemir/Masaüstü/PromptOptimizer
    source venv/bin/activate
    python src/main.py
"""

import sys
import logging
from pathlib import Path

# Add src/ to path so sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))

import config
import data_loader
import llm_interface
import evolution
import visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def main() -> None:
    """Run the full EvoPrompt optimization pipeline."""
    print("=" * 70)
    print("EvoPrompt Optimizer")
    print("Project HZ-2026-001 | Computational Intelligence Midterm")
    print("=" * 70)
    print()

    # ── Step 0: Validate Configuration ───────────────────────────────────
    logger.info("Validating configuration...")
    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    logger.info("Configuration valid.")
    print()

    # ── Step 1: Load & Preprocess Data ───────────────────────────────────
    logger.info("Step 1: Loading and preprocessing data...")
    dev_pool, test_set = data_loader.prepare_dataset(force_reprocess=False)
    print()

    dev_titles = dev_pool["title"].tolist()
    dev_labels = dev_pool["label"].tolist()
    test_titles = test_set["title"].tolist()
    test_labels = test_set["label"].tolist()

    # ── Step 2: Evaluate Initial Human-Written Prompt ────────────────────
    logger.info("Step 2: Evaluating initial human-written prompt...")
    initial_prompt = config.INITIAL_PROMPT
    initial_accuracy = llm_interface.evaluate_prompt(
        prompt=initial_prompt,
        titles=dev_titles[:config.MINI_BATCH_SIZE],  # Small sample for speed
        labels=dev_labels[:config.MINI_BATCH_SIZE],
    )
    logger.info(f"Initial prompt accuracy: {initial_accuracy:.4f} ({initial_accuracy:.2%})")
    print()

    # ── Step 3: Run Evolution ────────────────────────────────────────────
    logger.info("Step 3: Initializing population and starting evolution...")
    population = evolution.initialize_population(dev_titles, dev_labels)

    final_population, history = evolution.evolve(
        population=population,
        dev_pool_titles=dev_titles,
        dev_pool_labels=dev_labels,
        generations=config.GENERATIONS,
    )

    # Extract best individual
    best_individual = max(final_population, key=lambda ind: ind.fitness.values[0])
    best_prompt = str(best_individual)
    best_fitness = best_individual.fitness.values[0]

    logger.info(f"Best evolved prompt fitness: {best_fitness:.4f} ({best_fitness:.2%})")
    print()

    # ── Step 4: Evaluate on Final Test Set ───────────────────────────────
    logger.info("Step 4: Evaluating best prompt on held-out test set...")
    test_accuracy = llm_interface.evaluate_prompt(
        prompt=best_prompt,
        titles=test_titles,
        labels=test_labels,
    )
    logger.info(f"Test set accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print()

    # ── Step 5: Generate Outputs ─────────────────────────────────────────
    logger.info("Step 5: Generating visualizations and report...")

    plot_path = visualizer.plot_accuracy_curve(history)
    logger.info(f"Accuracy curve: {plot_path}")

    metadata = visualizer.save_best_prompt(best_prompt, best_fitness)
    logger.info(f"Best prompt saved: {metadata}")

    report_path = visualizer.generate_report(
        history=history,
        best_prompt=best_prompt,
        best_fitness=best_fitness,
        initial_fitness=initial_accuracy,
        test_accuracy=test_accuracy,
    )
    logger.info(f"Report: {report_path}")
    print()

    # ── Final Summary ────────────────────────────────────────────────────
    print("=" * 70)
    print("EVOlUTION COMPLETE — Summary")
    print("=" * 70)
    print(f"Initial Accuracy:  {initial_accuracy:.4f} ({initial_accuracy:.2%})")
    print(f"Best Dev Accuracy: {best_fitness:.4f} ({best_fitness:.2%})")
    print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy:.2%})")
    improvement = test_accuracy - initial_accuracy
    print(f"Improvement:       {improvement:+.4f} ({improvement:+.2%})")
    print()
    print(f"Outputs saved to: {config.OUTPUTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
