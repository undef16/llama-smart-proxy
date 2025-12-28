import pytest
from pydantic import ValidationError

from src.entities.model import Model


def test_model_creation_valid():
    model = Model(id="test-model", repo="user/repo", variant="v1", backend="llama.cpp")
    assert model.id == "test-model"
    assert model.repo == "user/repo"
    assert model.variant == "v1"
    assert model.backend == "llama.cpp"


def test_model_creation_without_variant():
    model = Model(id="test-model", repo="user/repo", backend="ollama")
    assert model.id == "test-model"
    assert model.repo == "user/repo"
    assert model.variant is None
    assert model.backend == "ollama"


def test_model_invalid_id_empty():
    with pytest.raises(ValidationError):
        Model(id="", repo="user/repo", backend="llama.cpp")


def test_model_invalid_repo_format():
    with pytest.raises(ValidationError):
        Model(id="test-model", repo="invalid", backend="llama.cpp")


def test_model_invalid_backend():
    with pytest.raises(ValidationError):
        Model(id="test-model", repo="user/repo", backend="invalid")
