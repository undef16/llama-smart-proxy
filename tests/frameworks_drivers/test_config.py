"""
Unit tests for config.py module.
"""

import json

import pytest

from src.frameworks_drivers.config import Config, ModelConfig, ServerPoolConfig


class TestServerPoolConfig:
    """Test ServerPoolConfig model."""

    def test_valid_config(self):
        """Test creating a valid ServerPoolConfig."""
        config = ServerPoolConfig(size=3, host="127.0.0.1", port_start=9000, gpu_layers=0, request_timeout=300)
        assert config.size == 3
        assert config.host == "127.0.0.1"
        assert config.port_start == 9000
        assert config.gpu_layers == 0
        assert config.request_timeout == 300

    def test_default_values(self):
        """Test default values for ServerPoolConfig."""
        config = ServerPoolConfig(size=2, host="localhost", port_start=8001, gpu_layers=1, request_timeout=60)
        assert config.size == 2
        assert config.host == "localhost"
        assert config.port_start == 8001
        assert config.gpu_layers == 1
        assert config.request_timeout == 60

    def test_invalid_size(self):
        """Test that size must be greater than 0."""
        with pytest.raises(ValueError):
            ServerPoolConfig(size=0, host="localhost", port_start=8001, gpu_layers=0, request_timeout=300)

        with pytest.raises(ValueError):
            ServerPoolConfig(size=-1, host="localhost", port_start=8001, gpu_layers=0, request_timeout=300)


class TestModelConfig:
    """Test ModelConfig model."""

    def test_valid_config(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(repo="test/repo", variant="Q4_K_M.gguf")
        assert config.repo == "test/repo"
        assert config.variant == "Q4_K_M.gguf"


class TestConfig:
    """Test Config model."""

    def test_valid_config(self, sample_config_data):
        """Test creating a valid Config."""
        config = Config(**sample_config_data)
        assert config.server_pool.size == 3
        assert config.server_pool.host == "127.0.0.1"
        assert config.server_pool.port_start == 9000
        assert len(config.effective_models) == 2
        assert "test-model" in config.effective_models
        assert config.effective_models["test-model"].repo == "test/repo"
        assert config.agents == ["test-agent", "another-agent"]

    def test_minimal_config(self, minimal_config_data):
        """Test config with minimal required fields."""
        config = Config(**minimal_config_data)
        assert config.server_pool.size == 1
        assert config.effective_models == {}
        assert config.agents == []

    def test_invalid_config(self, invalid_config_data):
        """Test that invalid config raises validation errors."""
        with pytest.raises(ValueError):
            Config(**invalid_config_data)


class TestLoadConfig:
    """Test load_config function."""

    def test_load_valid_config(self, config_file, sample_config_data):
        """Test loading a valid config file."""
        config = Config.load(config_file)
        assert isinstance(config, Config)
        assert config.server_pool.size == sample_config_data["server_pool"]["size"]
        assert config.server_pool.host == sample_config_data["server_pool"]["host"]
        assert config.server_pool.port_start == sample_config_data["server_pool"]["port_start"]
        assert len(config.effective_models) == len(sample_config_data["models"])
        assert config.agents == sample_config_data["agents"]

    def test_load_config_with_defaults(self, temp_dir, minimal_config_data):
        """Test loading config with default values."""
        config_path = temp_dir / "minimal_config.json"
        with open(config_path, "w") as f:
            json.dump(minimal_config_data, f, indent=2)

        config = Config.load(str(config_path))
        assert isinstance(config, Config)
        assert config.server_pool.size == 1
        assert config.server_pool.host == "localhost"  # default
        assert config.server_pool.port_start == 8001  # default
        assert config.effective_models == {}
        assert config.agents == []

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.load("nonexistent_config.json")

    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON raises JSON decode error."""
        config_path = temp_dir / "invalid.json"
        with open(config_path, "w") as f:
            f.write("invalid json content {")

        with pytest.raises(json.JSONDecodeError):
            Config.load(str(config_path))

    def test_load_invalid_config_structure(self, temp_dir):
        """Test loading JSON with invalid structure raises validation error."""
        config_path = temp_dir / "invalid_structure.json"
        invalid_data = {"invalid_field": "value"}
        with open(config_path, "w") as f:
            json.dump(invalid_data, f, indent=2)

        with pytest.raises(ValueError):
            Config.load(str(config_path))
