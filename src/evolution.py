"""
EvoPrompt Optimizer - Evolutionary Algorithm Engine

Implements the genetic algorithm using DEAP (Distributed Evolutionary
Algorithms in Python). Handles:

- Population initialization with seed prompt + WordNet variants
- Fitness evaluation via mini-batch sampling
- Tournament selection
- One-point crossover for prompt strings
- Word substitution mutation using NLTK WordNet synonyms
- Full evolution loop with per-generation logging

The LLM parameters remain fixed; only the prompt string evolves.
"""

import random
import string
import logging
from typing import Any

import nltk
from nltk.corpus import wordnet
from deap import base, creator, tools, algorithms

import config
import llm_interface

# Configure module logger
logger = logging.getLogger(__name__)

# Ensure WordNet data is downloaded
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


# ─── DEAP Setup ───────────────────────────────────────────────────────────────

# Register fitness and individual types globally
# FitnessWeights: positive = maximization (accuracy)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", str, fitness=creator.FitnessMax)  # type: ignore[reportAttributeAccessIssue]


# ─── Individual Creation ──────────────────────────────────────────────────────

def _str_to_individual(prompt_str: str) -> creator.Individual:  # type: ignore[reportAttributeAccessIssue]
    """
    Convert a prompt string to a DEAP Individual.

    DEAP individuals are normally lists, but we subclass str
    to make individuals directly represent prompt text.

    Args:
        prompt_str: The prompt template string.

    Returns:
        creator.Individual: Individual with string value and fitness.
    """
    ind = creator.Individual(prompt_str)
    ind.fitness.values = (0.0,)
    return ind


# ─── WordNet Synonym Mutation Helper ──────────────────────────────────────────

def _get_synonym(word: str) -> str | None:
    """
    Get a random synonym for a word using WordNet.

    Args:
        word: Word to find a synonym for.

    Returns:
        str | None: A synonym if found, else None.
    """
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():  # type: ignore[reportOptionalMemberAccess]
            syn_word = lemma.name().replace("_", " ")
            if syn_word.lower() != word.lower():
                synonyms.append(syn_word)

    return random.choice(synonyms) if synonyms else None


def _mutate_prompt(individual: creator.Individual,
                   indpb: float = config.MUTATION_PROBABILITY) -> tuple:
    """
    Mutate a prompt by randomly substituting a word with a synonym.

    Only words outside of placeholders ({title}) and common structural
    words are considered for mutation to preserve prompt functionality.

    Args:
        individual: DEAP individual (prompt string).
        indpb: Independent probability of mutation.

    Returns:
        tuple: Modified individual as a single-element tuple (DEAP convention).
    """
    words = individual.split()
    # Skip structural words and placeholder
    skip_words = {"{title}", "you", "are", "a", "an", "the", "is", "to",
                  "of", "in", "into", "for", "and", "or", "with", "by",
                  "only", "just", "reply", "respond", "classify"}

    mutated = False
    for i in range(len(words)):
        if random.random() < indpb and words[i].lower() not in skip_words:
            # Clean word of punctuation
            clean = words[i].strip(string.punctuation)
            synonym = _get_synonym(clean)
            if synonym:
                # Preserve original casing style
                if words[i][0].isupper():
                    synonym = synonym.capitalize()
                elif words[i].isupper():
                    synonym = synonym.upper()
                words[i] = synonym
                mutated = True
                break  # Only one word mutation per individual

    if mutated:
        new_prompt = " ".join(words)
        return (creator.Individual(new_prompt),)

    return (individual,)


# ─── Crossover ────────────────────────────────────────────────────────────────

def _cx_prompt(ind1: creator.Individual,
               ind2: creator.Individual,
               indpb: float = config.CROSSOVER_PROBABILITY) -> tuple:
    """
    One-point crossover between two prompt strings.

    Splits both parents at a random point and recombines to produce
    two children. If crossover doesn't occur (probability check),
    children are copies of parents.

    Args:
        ind1: First parent individual.
        ind2: Second parent individual.
        indpb: Probability of crossover occurring.

    Returns:
        tuple: (child1, child2) as DEAP individuals.
    """
    if random.random() < indpb:
        # Split at word boundaries
        words1 = str(ind1).split()
        words2 = str(ind2).split()

        if len(words1) > 1 and len(words2) > 1:
            point = random.randint(1, min(len(words1), len(words2)) - 1)

            new1 = " ".join(words1[:point] + words2[point:])
            new2 = " ".join(words2[:point] + words1[point:])

            ind1 = creator.Individual(new1)
            ind2 = creator.Individual(new2)

    return ind1, ind2


# ─── Fitness Evaluation ───────────────────────────────────────────────────────

def _evaluate(individual: creator.Individual,
              dev_pool_titles: list[str],
              dev_pool_labels: list[str],
              batch_size: int = config.MINI_BATCH_SIZE) -> tuple[float]:
    """
    Evaluate fitness of a prompt using mini-batch sampling.

    Instead of evaluating on the full development pool (expensive),
    a random sample of batch_size items is used.

    Args:
        individual: DEAP individual (prompt string).
        dev_pool_titles: All titles in the development pool.
        dev_pool_labels: All labels in the development pool.
        batch_size: Number of samples for fitness evaluation.

    Returns:
        tuple: (accuracy,) as a single-element tuple (DEAP convention).
    """
    # Sample mini-batch
    indices = random.sample(range(len(dev_pool_titles)), batch_size)
    batch_titles = [dev_pool_titles[i] for i in indices]
    batch_labels = [dev_pool_labels[i] for i in indices]

    accuracy = llm_interface.evaluate_prompt(
        prompt=str(individual),
        titles=batch_titles,
        labels=batch_labels,
    )

    return (accuracy,)


# ─── Population Initialization ───────────────────────────────────────────────

def initialize_population(
    dev_pool_titles: list[str],
    dev_pool_labels: list[str],
) -> list[creator.Individual]:
    """
    Create the initial population.

    The population includes:
    1. The human-written seed prompt
    2. WordNet-based variations of the seed prompt
    3. Random word substitutions for diversity

    Args:
        dev_pool_titles: Titles for fitness evaluation.
        dev_pool_labels: Labels for fitness evaluation.

    Returns:
        list[creator.Individual]: Initial population.
    """
    population = []

    # 1. Seed prompt
    seed = _str_to_individual(config.INITIAL_PROMPT)
    population.append(seed)

    # 2-3. Generate variations
    for _ in range(config.POPULATION_SIZE - 1):
        variant = _str_to_individual(config.INITIAL_PROMPT)
        mutated = _mutate_prompt(variant, indpb=0.5)  # Higher mutation for init
        population.append(mutated[0])

    # Evaluate initial fitness
    for ind in population:
        fitness = _evaluate(ind, dev_pool_titles, dev_pool_labels)
        ind.fitness.values = fitness

    logger.info(f"[EVO] Initial population: {len(population)} individuals")
    logger.info(f"[EVO] Best initial fitness: "
                f"{max(ind.fitness.values[0] for ind in population):.4f}")

    return population


# ─── Evolution Loop ───────────────────────────────────────────────────────────

def evolve(
    population: list[creator.Individual],
    dev_pool_titles: list[str],
    dev_pool_labels: list[str],
    generations: int = config.GENERATIONS,
) -> tuple[list[creator.Individual], dict[str, list]]:
    """
    Run the main evolution loop.

    Uses DEAP's eaSimple algorithm with custom operators.
    Records per-generation statistics for visualization.

    Args:
        population: Initial population list.
        dev_pool_titles: Titles for fitness evaluation.
        dev_pool_labels: Labels for fitness evaluation.
        generations: Number of generations to evolve.

    Returns:
        tuple: (final_population, hall_of_fame) where hall_of_fame
               contains {'best_individual': str, 'best_fitness': float,
                         'avg_fitness': [...], 'max_fitness': [...]}
    """
    # Create toolbox
    toolbox = base.Toolbox()

    # Register operators
    toolbox.register("select", tools.selTournament, tournsize=config.TOURNAMENT_SIZE)
    toolbox.register("mate", _cx_prompt)
    toolbox.register("mutate", _mutate_prompt, indpb=config.MUTATION_PROBABILITY)

    # Custom evaluate that captures dev_pool context
    def evaluate_with_context(ind):
        return _evaluate(ind, dev_pool_titles, dev_pool_labels)

    toolbox.register("evaluate", evaluate_with_context)

    # Hall of Fame to track best individual
    hof = tools.HallOfFame(1)

    # Statistics recording
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x) if x else 0)
    stats.register("max", max)
    stats.register("min", min)

    logger.info(f"[EVO] Starting evolution: {generations} generations, "
                f"pop_size={len(population)}")

    # Run evolution
    final_pop, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=config.CROSSOVER_PROBABILITY,
        mutpb=config.MUTATION_PROBABILITY,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    # Extract generation history
    history = {
        "gen": list(range(generations)),
        "avg_fitness": logbook.select("avg"),
        "max_fitness": logbook.select("max"),
        "min_fitness": logbook.select("min"),
    }

    best = hof[0]
    logger.info(f"[EVO] Evolution complete. Best fitness: {best.fitness.values[0]:.4f}")
    logger.info(f"[EVO] Best prompt: {str(best)[:100]}...")

    return final_pop, history
