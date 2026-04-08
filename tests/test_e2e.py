"""
EvoPrompt Optimizer - Unit Tests

Tests for all modules: config, data_loader, evolution operators, visualizer.
Run with: pytest tests/ -v
"""

import sys
import json
from pathlib import Path

import pytest

# Add src/ to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


# ─── Config Tests ─────────────────────────────────────────────────────────────

class TestConfig:
    """Tests for config.py module."""

    def test_config_imports(self):
        """Config module loads without errors."""
        import config
        assert hasattr(config, "POPULATION_SIZE")
        assert hasattr(config, "GENERATIONS")
        assert hasattr(config, "GROQ_MODEL")

    def test_population_size_valid(self):
        """Population size is at least 2."""
        import config
        assert config.POPULATION_SIZE >= 2

    def test_generations_valid(self):
        """Generations is at least 1."""
        import config
        assert config.GENERATIONS >= 1

    def test_probability_ranges(self):
        """Mutation and crossover probabilities are between 0 and 1."""
        import config
        assert 0.0 <= config.MUTATION_PROBABILITY <= 1.0
        assert 0.0 <= config.CROSSOVER_PROBABILITY <= 1.0

    def test_class_labels(self):
        """Class labels have 4 AG News categories."""
        import config
        assert len(config.CLASS_LABELS) == 4
        assert config.CLASS_LABELS[1] == "World"
        assert config.CLASS_LABELS[2] == "Sports"
        assert config.CLASS_LABELS[3] == "Business"
        assert config.CLASS_LABELS[4] == "Sci/Tech"

    def test_initial_prompt_has_placeholder(self):
        """Initial prompt contains {title} placeholder."""
        import config
        assert "{title}" in config.INITIAL_PROMPT

    def test_validate_config_with_key(self, monkeypatch):
        """validate_config passes when API key is set."""
        import config
        monkeypatch.setenv("GROQ_API_KEY", "test_key_123")
        monkeypatch.setattr(config, "GROQ_API_KEY", "test_key_123")
        # Should not raise
        config.validate_config()

    def test_validate_config_without_key(self, monkeypatch):
        """validate_config raises ValueError when API key is missing."""
        # Override both env and .env-loaded value
        monkeypatch.setenv("GROQ_API_KEY", "")
        # Force reload config module to pick up the empty key
        import importlib
        import config
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            config.validate_config()


# ─── Data Loader Tests ────────────────────────────────────────────────────────

class TestDataLoader:
    """Tests for data_loader.py module."""

    @pytest.fixture(scope="class")
    def dev_pool(self):
        """Load development pool for tests."""
        import data_loader
        dev, _ = data_loader.prepare_dataset(force_reprocess=False)
        return dev

    @pytest.fixture(scope="class")
    def test_set(self):
        """Load test set for tests."""
        import data_loader
        _, test = data_loader.prepare_dataset(force_reprocess=False)
        return test

    def test_load_dataset(self):
        """Dataset loads with correct columns."""
        import data_loader
        df = data_loader.load_dataset()
        assert "title" in df.columns
        assert "label" in df.columns
        assert len(df) == 7600

    def test_class_labels_mapping(self):
        """Class labels mapping is correct."""
        import data_loader
        mapping = data_loader.get_class_labels()
        assert mapping == {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

    def test_reverse_label_mapping(self):
        """Reverse mapping works correctly."""
        import data_loader
        rev = data_loader.get_label_mapping_reverse()
        assert rev["World"] == 1
        assert rev["Sports"] == 2
        assert rev["Business"] == 3
        assert rev["Sci/Tech"] == 4

    def test_split_preserves_columns(self, dev_pool, test_set):
        """Split DataFrames have correct columns."""
        assert "title" in dev_pool.columns
        assert "label" in dev_pool.columns
        assert "title" in test_set.columns
        assert "label" in test_set.columns

    def test_split_sizes(self, dev_pool, test_set):
        """Split produces expected sizes."""
        assert len(dev_pool) == 3800
        assert len(test_set) == 3800

    def test_split_stratification(self, dev_pool, test_set):
        """Both splits contain all 4 classes."""
        assert set(dev_pool["label"].unique()) == {"World", "Sports", "Business", "Sci/Tech"}
        assert set(test_set["label"].unique()) == {"World", "Sports", "Business", "Sci/Tech"}

    def test_no_missing_values(self, dev_pool, test_set):
        """No NaN values in split data."""
        assert dev_pool.isnull().sum().sum() == 0
        assert test_set.isnull().sum().sum() == 0


# ─── Evolution Operator Tests ─────────────────────────────────────────────────

class TestEvolution:
    """Tests for evolution.py operators."""

    def test_synonym_lookup(self):
        """WordNet synonym lookup returns results for common words."""
        import evolution
        syn = evolution._get_synonym("happy")
        assert syn is not None
        assert isinstance(syn, str)
        assert len(syn) > 0

    def test_synonym_not_found_for_placeholder(self):
        """No synonym for placeholder tokens."""
        import evolution
        result = evolution._get_synonym("{title}")
        assert result is None

    def test_mutate_preserves_placeholder(self):
        """Mutation does not corrupt the {title} placeholder."""
        from deap import base, creator  # type: ignore[reportAttributeAccessIssue]
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # type: ignore[reportAttributeAccessIssue]
        creator.create("Individual", str, fitness=creator.FitnessMax)  # type: ignore[reportAttributeAccessIssue]

        import evolution
        ind = creator.Individual("Classify this: {title} into categories")  # type: ignore[reportAttributeAccessIssue]
        mutated = evolution._mutate_prompt(ind, indpb=1.0)

        assert "{title}" in str(mutated[0])

    def test_crossover_produces_valid_strings(self):
        """Crossover produces valid string individuals."""
        from deap import base, creator  # type: ignore[reportAttributeAccessIssue]
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # type: ignore[reportAttributeAccessIssue]
        creator.create("Individual", str, fitness=creator.FitnessMax)  # type: ignore[reportAttributeAccessIssue]

        import evolution
        ind1 = creator.Individual("Classify the news headline into categories")  # type: ignore[reportAttributeAccessIssue]
        ind2 = creator.Individual("You are an expert, determine the topic for: {title}")  # type: ignore[reportAttributeAccessIssue]

        child1, child2 = evolution._cx_prompt(ind1, ind2, indpb=1.0)

        assert isinstance(str(child1), str)
        assert isinstance(str(child2), str)
        assert len(str(child1)) > 0
        assert len(str(child2)) > 0


# ─── Visualizer Tests ─────────────────────────────────────────────────────────

class TestVisualizer:
    """Tests for visualizer.py module."""

    @pytest.fixture
    def mock_history(self):
        """Sample history data for testing."""
        return {
            "avg_fitness": [0.30, 0.40, 0.50, 0.60],
            "max_fitness": [0.35, 0.50, 0.65, 0.75],
            "min_fitness": [0.25, 0.30, 0.35, 0.45],
        }

    def test_plot_creates_file(self, mock_history, tmp_path):
        """Accuracy curve plot is saved to disk."""
        import visualizer
        output = tmp_path / "test_plot.png"
        result = visualizer.plot_accuracy_curve(mock_history, output)

        assert result.exists()
        assert result.stat().st_size > 0

    def test_save_best_prompt_text(self, tmp_path):
        """Best prompt text file is saved correctly."""
        import visualizer
        output = tmp_path / "test_prompt.txt"
        visualizer.save_best_prompt("Test prompt {title}", 0.75, output)

        assert output.exists()
        content = output.read_text()
        assert "Test prompt {title}" in content
        assert "0.75" in content

    def test_save_best_prompt_json(self, tmp_path):
        """Best prompt JSON file is valid and complete."""
        import visualizer
        output = tmp_path / "test_prompt.json"
        metadata = visualizer.save_best_prompt("Test {title}", 0.80, output_path_json=output)

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["prompt"] == "Test {title}"
        assert data["accuracy"] == 0.80
        assert "config" in data

    def test_generate_report(self, mock_history, tmp_path):
        """Report file contains evolution summary."""
        import visualizer
        output = tmp_path / "test_report.txt"
        visualizer.generate_report(
            history=mock_history,
            best_prompt="Best {title}",
            best_fitness=0.85,
            initial_fitness=0.50,
            test_accuracy=0.80,
            output_path=output,
        )

        assert output.exists()
        content = output.read_text()
        assert "Best {title}" in content
        assert "0.85" in content
        assert "0.50" in content
        assert "0.80" in content
