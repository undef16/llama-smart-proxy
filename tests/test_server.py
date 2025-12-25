import pytest
from pydantic import ValidationError

from src.entities.server import Server


def test_server_creation_valid():
    server = Server(id="server1", host="localhost", port=8080, model_id="model1", status="running", process=1234)
    assert server.id == "server1"
    assert server.host == "localhost"
    assert server.port == 8080
    assert server.model_id == "model1"
    assert server.status == "running"
    assert server.process == 1234


def test_server_creation_without_process():
    server = Server(id="server1", host="localhost", port=8080, model_id="model1", status="stopped")
    assert server.process is None


def test_server_invalid_port_low():
    with pytest.raises(ValidationError):
        Server(id="server1", host="localhost", port=0, model_id="model1", status="running")


def test_server_invalid_port_high():
    with pytest.raises(ValidationError):
        Server(id="server1", host="localhost", port=70000, model_id="model1", status="running")


def test_server_invalid_status():
    with pytest.raises(ValidationError):
        Server(id="server1", host="localhost", port=8080, model_id="model1", status="invalid")
