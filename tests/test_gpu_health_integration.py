import pytest
from unittest.mock import MagicMock

from src.entities.model import Model
from src.entities.server import Server
from src.entities.gpu_assignment import GPUAssignment
from src.entities.gpu import GPU
from src.use_cases.get_health import GetHealth
from src.frameworks_drivers.gpu_monitor import GPUMonitor
from src.frameworks_drivers.model_repository import ModelRepository
from src.frameworks_drivers.config import Config, GPUConfig


class TestGPUHealthIntegration:
    def test_get_health_includes_gpu_pool_status_when_gpu_monitor_initialized(self):
        """Test that health endpoint includes GPU pool status when GPU monitoring is available."""
        # Arrange
        model_repository = ModelRepository()
        gpu_monitor = MagicMock(spec=GPUMonitor)
        gpu_monitor.initialized = True

        # Create config with GPU settings
        gpu_config = GPUConfig(
            enabled=True,
            enable_gpu_monitoring=True,
            allocation_strategy="single-gpu-preferred",
            gpu_allocation_strategy="single-gpu-preferred",
            monitoring_interval=5.0,
            cpu_fallback=True,
            kv_offload=True,
            cache_type_k="f16",
            cache_type_v="f16",
            repack=True,
            no_host=False,
            activation_overhead_factor=0.25
        )
        from src.frameworks_drivers.config import ServerConfig, ServerPoolConfig
        server_config = ServerConfig(host="0.0.0.0", port=8000)
        server_pool_config = ServerPoolConfig(size=2, host="localhost", port_start=8001, gpu_layers=99, request_timeout=300)
        config = Config(
            backend="llama.cpp",
            server=server_config,
            server_pool=server_pool_config,
            gpu=gpu_config
        )

        # Create actual GPU object
        mock_gpu = GPU(
            id=0,
            name="Test GPU",
            total_memory=8.0,
            free_memory=6.0,
            used_memory=2.0,
            utilization=25.0,
            temperature=60,
            power_usage=10,
            assigned_models=["model1"],
            compute_capability="7.5"
        )

        gpu_monitor.get_all_gpus.return_value = [mock_gpu]

        # Add some models and servers to the repository
        model = Model(id="model1", repo="user/repo1", backend="llama.cpp")
        model_repository.models["model1"] = model

        # Create a server with GPU assignment
        server = Server(
            id="server1",
            host="localhost",
            port=8080,
            model_id="model1",
            status="running",
            gpu_assignment=GPUAssignment(
                gpu_ids=[0],
                tensor_splits=[1.0],
                estimated_vram_required=2.0
            )
        )
        model_repository.servers = [server]

        use_case = GetHealth(model_repository, gpu_monitor, config)
        
        # Act
        result = use_case.execute()
        
        # Assert
        assert "gpu_pool_status" in result
        assert result["gpu_pool_status"]["total_gpus"] == 1
        assert result["gpu_pool_status"]["total_memory"] == 8.0
        assert result["gpu_pool_status"]["used_memory"] == 2.0
        assert result["gpu_pool_status"]["free_memory"] == 6.0
        assert result["gpu_pool_status"]["utilization_average"] == 25.0
        assert "allocation_strategy" in result["gpu_pool_status"]
        
    def test_get_health_no_gpu_pool_status_when_gpu_monitor_not_initialized(self):
        """Test that health endpoint does not include GPU pool status when GPU monitoring is not available."""
        # Arrange
        model_repository = ModelRepository()
        gpu_monitor = MagicMock(spec=GPUMonitor)
        gpu_monitor.initialized = False # GPU monitor not initialized
        
        # Add some models and servers to the repository
        model = Model(id="model1", repo="user/repo1", backend="llama.cpp")
        model_repository.models["model1"] = model
        
        server = Server(
            id="server1", 
            host="localhost", 
            port=8080, 
            model_id="model1", 
            status="running"
        )
        model_repository.servers = [server]
        
        use_case = GetHealth(model_repository, gpu_monitor)
        
        # Act
        result = use_case.execute()
        
        # Assert
        assert "gpu_pool_status" not in result
        assert "servers" in result
        assert len(result["servers"]) == 1
        
    def test_get_health_includes_server_gpu_assignment_info(self):
        """Test that health endpoint includes GPU assignment information for servers."""
        # Arrange
        model_repository = ModelRepository()
        gpu_monitor = MagicMock(spec=GPUMonitor)
        gpu_monitor.initialized = True
        
        # Create actual GPU object
        mock_gpu = GPU(
            id=0,
            name="Test GPU",
            total_memory=8.0,
            free_memory=6.0,
            used_memory=2.0,
            utilization=25.0,
            temperature=60,
            power_usage=100,
            assigned_models=["model1"],
            compute_capability="7.5"
        )
        
        gpu_monitor.get_all_gpus.return_value = [mock_gpu]
        
        # Add model and server with GPU assignment
        model = Model(id="model1", repo="user/repo1", backend="llama.cpp")
        model_repository.models["model1"] = model
        
        gpu_assignment = GPUAssignment(
            gpu_ids=[0],
            tensor_splits=[1.0],
            estimated_vram_required=2.0
        )
        
        server = Server(
            id="server1", 
            host="localhost", 
            port=8080, 
            model_id="model1", 
            status="running",
            gpu_assignment=gpu_assignment
        )
        model_repository.servers = [server]
        
        use_case = GetHealth(model_repository, gpu_monitor)
        
        # Act
        result = use_case.execute()
        
        # Assert
        assert len(result["servers"]) == 1
        server_data = result["servers"][0]
        assert server_data["gpu_assignment"] is not None
        assert server_data["gpu_assignment"]["gpu_ids"] == [0]
        assert server_data["gpu_assignment"]["estimated_vram_required"] == 2.0