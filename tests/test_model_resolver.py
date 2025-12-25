"""
Unit tests for model_resolver.py module.
"""

import pytest

from src.frameworks_drivers.model_resolver import ModelResolver


class TestModelResolver:
    """Test ModelResolver class."""

    def test_resolve_repo_only(self):
        """Test resolving model identifier with repo only (uses default variant)."""
        repo_id, filename_pattern = ModelResolver.resolve("test/repo")
        assert repo_id == "test/repo"
        assert filename_pattern == "*Q4_K_M.gguf"

    def test_resolve_repo_with_variant(self):
        """Test resolving model identifier with repo and variant."""
        repo_id, filename_pattern = ModelResolver.resolve("test/repo:Q4_K_M")
        assert repo_id == "test/repo"
        assert filename_pattern == "*Q4_K_M.gguf"

    def test_resolve_repo_with_gguf_variant(self):
        """Test resolving model identifier with repo and .gguf variant."""
        repo_id, filename_pattern = ModelResolver.resolve("test/repo:Q4_K_M.gguf")
        assert repo_id == "test/repo"
        assert filename_pattern == "Q4_K_M.gguf"

    def test_resolve_complex_repo_name(self):
        """Test resolving model identifier with complex repo name."""
        repo_id, filename_pattern = ModelResolver.resolve("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
        assert repo_id == "unsloth/Qwen3-0.6B-GGUF"
        assert filename_pattern == "*Q4_K_M.gguf"

    def test_resolve_with_whitespace(self):
        """Test resolving model identifier with leading/trailing whitespace."""
        repo_id, filename_pattern = ModelResolver.resolve("  test/repo:Q4_K_M  ")
        assert repo_id == "test/repo"
        assert filename_pattern == "*Q4_K_M.gguf"

    def test_resolve_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Model identifier cannot be empty"):
            ModelResolver.resolve("")

    def test_resolve_whitespace_only(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Model identifier cannot be empty"):
            ModelResolver.resolve("   ")

    def test_resolve_empty_repo(self):
        """Test that empty repo part raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model identifier format"):
            ModelResolver.resolve(":Q4_K_M")

    def test_resolve_empty_variant(self):
        """Test that empty variant part raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model identifier format"):
            ModelResolver.resolve("test/repo:")

    def test_resolve_multiple_colons(self):
        """Test that multiple colons raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model identifier format"):
            ModelResolver.resolve("repo:variant:extra")

    def test_resolve_only_colon(self):
        """Test that only colon raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model identifier format"):
            ModelResolver.resolve(":")

    def test_default_variant_constant(self):
        """Test that DEFAULT_VARIANT constant is correct."""
        assert ModelResolver.DEFAULT_VARIANT == "Q4_K_M"

    def test_resolve_case_preservation(self):
        """Test that case is preserved in repo and variant names."""
        repo_id, filename_pattern = ModelResolver.resolve("Test/Repo:Custom_Variant")
        assert repo_id == "Test/Repo"
        assert filename_pattern == "*Custom_Variant.gguf"

    def test_resolve_special_characters(self):
        """Test resolving model identifier with special characters."""
        repo_id, filename_pattern = ModelResolver.resolve("org/repo-name_123:Q4_K_M-test.gguf")
        assert repo_id == "org/repo-name_123"
        assert filename_pattern == "Q4_K_M-test.gguf"
