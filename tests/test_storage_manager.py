import pytest
import hashlib
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from tinygrad import Tensor, dtypes
from anytensor.src.anytensor.core import PlaceholderInfo, GraphProgram, ActualTensors
from anytensor.src.anytensor.storage_manager import (
    fetch_safetensors_by_uuid,
    fetch_exported_task_by_uuid,
)


class TestStorageManager:
    @pytest.fixture
    def mock_app_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Create a temporary directory structure that mimics the app directory."""
        # Create mock directory structure
        safetensors_dir = tmp_path / "safetensors"
        tasks_dir = tmp_path / "tasks"
        safetensors_dir.mkdir()
        tasks_dir.mkdir()

        # Patch the APP_DIR to use our temporary directory
        monkeypatch.setattr("anytensor.src.anytensor.storage_manager.APP_DIR", tmp_path)
        return tmp_path

    @pytest.fixture
    def sample_safetensor_file(self, mock_app_dir: Path) -> Dict[str, Any]:
        """Create a sample safetensor file with a known hash."""
        # Create binary content that will have a predictable hash
        content = b"sample_safetensor_content"
        file_hash = hashlib.sha256(content).hexdigest()

        # Write to a file
        file_path = mock_app_dir / "safetensors" / "sample.safetensors"
        with open(file_path, "wb") as f:
            f.write(content)

        return {"path": file_path, "hash": file_hash, "content": content}

    @pytest.fixture
    def sample_task_file(self, mock_app_dir: Path) -> Dict[str, Any]:
        """Create a sample task file with a known hash."""
        # Create a simple exported task
        placeholder = PlaceholderInfo(True, "test_input", (2, 3), dtypes.float32)
        task = GraphProgram(Tensor.ones((2, 3)), [placeholder])

        # Serialize the task
        task_data = task.to_bytes()

        # Get the hash (we'll just use the first 8 chars as the UUID)
        file_hash = hashlib.sha256(task_data).hexdigest()[:8]

        # Write to a file
        file_path = mock_app_dir / "tasks" / "sample_task.pkl"
        with open(file_path, "wb") as f:
            f.write(task_data)

        return {"path": file_path, "hash": file_hash, "task": task, "data": task_data}

    @patch("anytensor.src.anytensor.storage_manager.safe_load")
    def test_fetch_safetensors_by_uuid_with_path(
        self, mock_safe_load: MagicMock, sample_safetensor_file: Dict[str, Any]
    ) -> None:
        """Test importing weights when path is explicitly provided."""
        # Setup mock return value
        expected_tensors: ActualTensors = {"weight1": Tensor.ones((2, 3))}
        mock_safe_load.return_value = expected_tensors

        # Call function with explicit path
        result = fetch_safetensors_by_uuid(
            sample_safetensor_file["hash"],
        )

        # Verify safe_load was called correctly
        mock_safe_load.assert_called_once_with(sample_safetensor_file["path"])
        assert result == expected_tensors

    @patch("anytensor.src.anytensor.storage_manager.safe_load")
    def test_fetch_safetensors_by_uuid_search(
        self, mock_safe_load: MagicMock, sample_safetensor_file: Dict[str, Any]
    ) -> None:
        """Test importing weights by searching for the file by hash."""
        # Setup mock return value
        expected_tensors: ActualTensors = {"weight1": Tensor.ones((2, 3))}
        mock_safe_load.return_value = expected_tensors

        # Call function without explicit path
        result = fetch_safetensors_by_uuid(sample_safetensor_file["hash"])

        # Verify safe_load was called correctly
        mock_safe_load.assert_called_once()
        assert result == expected_tensors

    def test_fetch_safetensors_by_uuid_hash_mismatch(
        self, sample_safetensor_file: Dict[str, Any]
    ) -> None:
        """Test importing weights with hash mismatch in the provided path."""
        wrong_uuid = "wrong_hash_value"

        # Trying to use a file with mismatched hash should raise ValueError
        with pytest.raises(
            ValueError, match="File hash .* does not match provided UUID"
        ):
            fetch_safetensors_by_uuid(wrong_uuid)

    def test_fetch_safetensors_by_uuid_not_found(self, mock_app_dir: Path) -> None:
        """Test importing weights when the file is not found."""
        # Try to import a file that doesn't exist
        with pytest.raises(ValueError, match="No file found with UUID"):
            fetch_safetensors_by_uuid("nonexistent_uuid")

    def test_fetch_exported_task_by_uuid(
        self, sample_task_file: Dict[str, Any]
    ) -> None:
        """Test fetching an exported task by UUID."""
        # Fetch the task using the UUID (hash)
        result = fetch_exported_task_by_uuid(sample_task_file["hash"])

        # Verify the result
        assert isinstance(result, GraphProgram)
        assert len(result.placeholders) == 1
        assert result.placeholders[0].name == "test_input"

    def test_fetch_exported_task_by_uuid_not_found(self, mock_app_dir: Path) -> None:
        """Test fetching a task that doesn't exist."""
        # Try to fetch a task that doesn't exist
        result = fetch_exported_task_by_uuid("nonexistent_uuid")

        # Should return a ValueError
        assert isinstance(result, ValueError)
        assert "No file found matching UUID" in str(result)

    @patch("builtins.open")
    @patch("anytensor.src.anytensor.storage_manager.GraphProgram.from_bytes")
    def test_fetch_exported_task_by_uuid_corrupt_file(
        self, mock_from_bytes: MagicMock, mock_open: MagicMock, mock_app_dir: Path
    ) -> None:
        """Test fetching a task from a corrupt file."""
        # Setup mock to simulate a file existing but being corrupted
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = b"corrupt_data"
        mock_open.return_value = mock_file

        # Mock file hash to match our test UUID
        mock_hash = MagicMock()
        mock_hash.hexdigest.return_value = "test_uuid" + "0" * 24
        hashlib.sha256 = MagicMock(return_value=mock_hash)

        # Simulate a ValueError when trying to parse the file
        error_msg = "Failed to unpickle data"
        mock_from_bytes.return_value = ValueError(error_msg)

        # Create a dummy task file to find
        (mock_app_dir / "tasks").mkdir(exist_ok=True)
        (mock_app_dir / "tasks" / "corrupt.pkl").touch()

        # Fetch the task
        result = fetch_exported_task_by_uuid("test_uuid")

        # Should return the ValueError from GraphProgram.from_bytes
        assert isinstance(result, ValueError)
        assert str(result) == error_msg
