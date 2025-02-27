import pathlib
import hashlib
from anytensor.core import GraphProgram, ActualTensors, APP_DIR
from tinygrad.nn.state import safe_load
import urllib.request
from urllib.parse import urlparse

####
# Tools For Importing Safe Tensors
####


def _validate_url(url: str) -> bool:
    """Validate URL format."""
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


def _download_file(url: str, target_path: pathlib.Path) -> None:
    """Download file from URL to target path."""
    urllib.request.urlretrieve(url, target_path)


def _verify_file_hash(
    file_path: pathlib.Path, expected_hash: str, truncate: bool = False
) -> bool:
    """Verify file hash matches expected hash."""
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
        if truncate:
            file_hash = file_hash[:8]
        return file_hash == expected_hash


def _find_matching_file(
    directory: pathlib.Path, pattern: str, uuid: str, truncate: bool = False
) -> pathlib.Path | None:
    """Find file with matching hash in directory."""
    for file in directory.glob(pattern):
        if _verify_file_hash(file, uuid, truncate):
            return file
    return None


def _download_and_verify(
    url: str, temp_path: pathlib.Path, uuid: str, truncate: bool = False
) -> pathlib.Path | None:
    """Download file and verify its hash."""
    try:
        if not _validate_url(url):
            raise ValueError("Invalid URL format")

        _download_file(url, temp_path)

        if _verify_file_hash(temp_path, uuid, truncate):
            return temp_path

        temp_path.unlink()
        return None

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise ValueError(f"Failed to download from URL: {str(e)}")


def fetch_safetensors_by_uuid(uuid: str, url: str | None = None) -> ActualTensors:
    """Import weights from a safetensor file identified by UUID.

    Args:
        uuid: The UUID/hash of the safetensor file to load
        url: Optional URL to download the file from if not found locally

    Returns:
        Dict mapping tensor names to tensors

    Raises:
        ValueError if no matching file found or hash mismatch
    """
    app_dir = APP_DIR / "safetensors"
    app_dir.mkdir(parents=True, exist_ok=True)

    matching_file = _find_matching_file(app_dir, "*.safetensors", uuid)

    if matching_file is None and url is not None:
        matching_file = _download_and_verify(
            url, app_dir / f"temp_{uuid}.safetensors", uuid
        )

    if matching_file is None:
        raise ValueError(f"No file found with UUID {uuid}")

    return safe_load(matching_file)


###
# Get Exported Task
###


def fetch_exported_task_by_uuid(
    uuid: str, url: str | None = None
) -> GraphProgram | ValueError:
    """Get an exported task from a UUID.

    Args:
        uuid: The UUID/hash of the task file to load
        url: Optional URL to download the file from if not found locally

    Returns:
        The loaded task or ValueError if not found/invalid
    """
    tasks_dir = APP_DIR / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    matching_file = _find_matching_file(tasks_dir, "*.pkl", uuid, truncate=True)

    if matching_file is None and url is not None:
        matching_file = _download_and_verify(
            url, tasks_dir / f"temp_{uuid}.pkl", uuid, truncate=True
        )

    if not matching_file:
        return ValueError(f"No file found matching UUID: {uuid}")

    with open(matching_file, "rb") as f:
        return GraphProgram.from_bytes(f.read())
