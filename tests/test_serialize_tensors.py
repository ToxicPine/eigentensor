import pytest
import numpy as np
from typing import List, Tuple
from tinygrad import Tensor, dtypes

from anytensor.src.anytensor.serialize_tensors import TensorSerializer


class TestTensorSerializer:
    @pytest.fixture
    def tensor_serializer(self) -> TensorSerializer:
        return TensorSerializer()

    @pytest.fixture
    def sample_tensors(self) -> List[Tensor]:
        """Create a variety of tensors for testing serialization."""
        return [
            Tensor.ones((2, 3)),  # Basic float32 tensor
            Tensor.zeros((5, 5), dtype=dtypes.float16),  # Different dtype
            Tensor.arange(10),  # 1D tensor with values
            Tensor([1, 2, 3, 4], dtype=dtypes.int32),  # Integer tensor
            Tensor([[1.5, 2.5], [3.5, 4.5]]),  # Small 2D tensor with specific values
            Tensor.ones((0,)),  # Empty tensor
            Tensor.ones((1, 1, 1, 1)),  # 4D tensor with single element
        ]

    def test_tensor_to_bytes_basic(self, tensor_serializer: TensorSerializer) -> None:
        """Test basic serialization of a tensor to bytes."""
        tensor = Tensor.ones((2, 3))
        serialized = TensorSerializer.tensor_to_bytes(tensor)

        # Verify it's bytes
        assert isinstance(serialized, bytes)

        # Check that the metadata is included
        first_line = serialized.split(b"\n")[0]
        assert first_line.decode() == "2,3"  # Shape

        second_line = serialized.split(b"\n")[1]
        assert second_line.decode() == "float32"  # Dtype

    def test_tensor_from_bytes_basic(self, tensor_serializer: TensorSerializer) -> None:
        """Test basic deserialization of bytes to a tensor."""
        # Create manually serialized tensor data
        shape_str = "2,3"
        dtype_str = "float32"
        raw_data = np.ones((2, 3), dtype=np.float32).tobytes()

        serialized = f"{shape_str}\n{dtype_str}\n".encode() + raw_data

        # Deserialize
        tensor = TensorSerializer.tensor_from_bytes(serialized)

        # Verify properties
        assert tensor.shape == (2, 3)
        assert tensor.dtype == dtypes.float32
        assert np.array_equal(tensor.numpy(), np.ones((2, 3), dtype=np.float32))

    @pytest.mark.parametrize(
        "dtype",
        [
            dtypes.float32,
            dtypes.float16,
            dtypes.int32,
            dtypes.int64,
            dtypes.bool,
        ],
    )
    def test_different_dtypes(
        self, tensor_serializer: TensorSerializer, dtype: dtypes
    ) -> None:
        """Test serialization/deserialization with different dtypes."""
        original = Tensor.ones((3, 4), dtype=dtype)

        # Serialize and deserialize
        serialized = TensorSerializer.tensor_to_bytes(original)
        reconstructed = TensorSerializer.tensor_from_bytes(serialized)

        # Verify
        assert reconstructed.shape == original.shape
        assert reconstructed.dtype == original.dtype
        # We have to compare the numpy arrays because Tensor equality isn't implemented
        assert np.array_equal(reconstructed.numpy(), original.numpy())

    @pytest.mark.parametrize(
        "shape",
        [
            (10,),  # 1D
            (5, 5),  # 2D square
            (2, 3, 4),  # 3D
            (2, 3, 4, 5),  # 4D
            (1, 1, 1, 1),  # 4D singleton
            (0,),  # Empty 1D
            (0, 3),  # Empty first dimension
        ],
    )
    def test_different_shapes(
        self, tensor_serializer: TensorSerializer, shape: Tuple[int, ...]
    ) -> None:
        """Test serialization/deserialization with different tensor shapes."""
        # Create tensor with the specified shape
        original = Tensor.ones(shape)

        # Serialize and deserialize
        serialized = TensorSerializer.tensor_to_bytes(original)
        reconstructed = TensorSerializer.tensor_from_bytes(serialized)

        # Verify
        assert reconstructed.shape == original.shape
        assert reconstructed.dtype == original.dtype
        assert np.array_equal(reconstructed.numpy(), original.numpy())

    def test_roundtrip_all_samples(
        self, tensor_serializer: TensorSerializer, sample_tensors: List[Tensor]
    ) -> None:
        """Test roundtrip serialization and deserialization for all sample tensors."""
        for original in sample_tensors:
            # Serialize and deserialize
            serialized = TensorSerializer.tensor_to_bytes(original)
            reconstructed = TensorSerializer.tensor_from_bytes(serialized)

            # Verify
            assert reconstructed.shape == original.shape
            assert reconstructed.dtype == original.dtype
            assert np.array_equal(reconstructed.numpy(), original.numpy())

    def test_non_contiguous_tensor(self, tensor_serializer: TensorSerializer) -> None:
        """Test serialization of a non-contiguous tensor."""
        # Create a large tensor and then slice it to get a non-contiguous view
        large = Tensor.ones((10, 10))
        non_contiguous = large[:5, ::2]  # Every other column in first 5 rows

        # Serialize and deserialize
        serialized = TensorSerializer.tensor_to_bytes(non_contiguous)
        reconstructed = TensorSerializer.tensor_from_bytes(serialized)

        # Verify
        assert reconstructed.shape == non_contiguous.shape
        assert reconstructed.dtype == non_contiguous.dtype
        assert np.array_equal(reconstructed.numpy(), non_contiguous.numpy())

    def test_tensor_with_specific_values(
        self, tensor_serializer: TensorSerializer
    ) -> None:
        """Test serialization of a tensor with specific values."""
        # Create a tensor with specific values
        values = [[1.5, -2.5, 3.0], [4.0, -5.5, 6.0]]
        original = Tensor(values)

        # Serialize and deserialize
        serialized = TensorSerializer.tensor_to_bytes(original)
        reconstructed = TensorSerializer.tensor_from_bytes(serialized)

        # Verify
        assert reconstructed.shape == original.shape
        assert reconstructed.dtype == original.dtype
        assert np.allclose(reconstructed.numpy(), np.array(values, dtype=np.float32))

    def test_serialized_size(self, tensor_serializer: TensorSerializer) -> None:
        """Test that the serialized size is reasonable."""
        # Create tensors of different sizes
        small = Tensor.ones((10, 10))  # 400 bytes of float32 data
        large = Tensor.ones((100, 100))  # 40,000 bytes of float32 data

        # Serialize
        small_bytes = TensorSerializer.tensor_to_bytes(small)
        large_bytes = TensorSerializer.tensor_to_bytes(large)

        # Verify sizes are reasonable
        # We expect the serialized size to be close to the raw data size plus a small overhead for metadata
        small_expected = 10 * 10 * 4 + 20  # 4 bytes per float32 + ~20 bytes overhead
        large_expected = 100 * 100 * 4 + 20  # 4 bytes per float32 + ~20 bytes overhead

        assert len(small_bytes) >= small_expected
        assert len(large_bytes) >= large_expected

        # Also verify that the large tensor produces significantly more bytes than the small one
        assert (
            len(large_bytes) > len(small_bytes) * 5
        )  # Should be ~100x bigger, but allow some margin
