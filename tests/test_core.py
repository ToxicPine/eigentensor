import pytest
import pickle
from typing import Dict, Tuple, Union, Callable, Any
from pathlib import Path

from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, Ops
from tinygrad.shape.shapetracker import ShapeTracker

from anytensor.src.anytensor.core import (
    PlaceholderInfo,
    TensorTemplateManager,
    TensorContext,
    GraphProgram,
    find_all_placeholders,
    execute_graph_on_gpu,
    infer_tensor_context_from_weights,
    ActualTensors,
)

from anytensor.src.anytensor.graph_rewriting import (
    buffer_uop_contains_placeholder,
    get_placeholder_name,
)


@pytest.fixture
def placeholder_info() -> PlaceholderInfo:
    return PlaceholderInfo(True, "test_tensor", (1, 2, 3), dtypes.float32)


@pytest.fixture
def tensor_template_manager() -> TensorTemplateManager:
    return TensorTemplateManager()


@pytest.fixture
def tensor_context() -> TensorContext:
    return TensorContext()


class TestPlaceholderInfo:
    def test_creation(self, placeholder_info: PlaceholderInfo) -> None:
        assert placeholder_info.placeholder is True
        assert placeholder_info.name == "test_tensor"
        assert placeholder_info.shape == (1, 2, 3)
        assert placeholder_info.dtype == dtypes.float32

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                {
                    "placeholder": True,
                    "name": "test_tensor",
                    "shape": (1, 2, 3),
                    "dtype": dtypes.float32,
                },
                True,
            ),
            (
                {"name": "test_tensor", "shape": (1, 2, 3), "dtype": dtypes.float32},
                False,
            ),
            (
                {
                    "placeholder": True,
                    "name": 123,
                    "shape": (1, 2, 3),
                    "dtype": dtypes.float32,
                },
                False,
            ),
        ],
    )
    def test_from_dict(self, test_input: Dict[str, Any], expected: bool) -> None:
        info = PlaceholderInfo.from_dict(test_input)
        if expected:
            assert info is not None
            assert info.name == "test_tensor"
            assert info.shape == (1, 2, 3)
        else:
            assert info is None

    def test_to_dict(self, placeholder_info: PlaceholderInfo) -> None:
        d = placeholder_info.to_dict()
        assert d["placeholder"] is True
        assert d["name"] == "test_tensor"
        assert d["shape"] == (1, 2, 3)
        assert d["dtype"] == "float32"  # Note: converts to name string

    def test_to_string(self, placeholder_info: PlaceholderInfo) -> None:
        s = placeholder_info.to_string()
        assert s == "placeholder:test_tensor:1,2,3:float32"

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            ("placeholder:test_tensor:1,2,3:float32", True),
            ("not_placeholder:test_tensor:1,2,3:float32", False),
            ("placeholder:test_tensor", False),
        ],
    )
    def test_from_string(self, test_input: str, expected: bool) -> None:
        info = PlaceholderInfo.from_string(test_input)
        if expected:
            assert info is not None
            assert info.name == "test_tensor"
            assert info.shape == (1, 2, 3)
            assert info.dtype == dtypes.float32
        else:
            assert info is None


class TestBufferOperations:
    @pytest.fixture
    def buffer_uop_with_placeholder(self) -> UOp:
        return UOp.new_buffer(
            device="DEFAULT",
            size=6,
            dtype=dtypes.float32,
        ).replace(arg=("DEFAULT", 6, "placeholder:test_tensor:1,2,3:float32"))

    def test_buffer_contains_placeholder(
        self, buffer_uop_with_placeholder: UOp
    ) -> None:
        assert buffer_uop_contains_placeholder(buffer_uop_with_placeholder) is True

        normal_buffer: UOp = UOp.new_buffer(
            device="DEFAULT",
            size=6,
            dtype=dtypes.float32,
        )
        assert buffer_uop_contains_placeholder(normal_buffer) is False

        non_buffer: UOp = UOp(
            Ops.ADD, dtypes.float32, (), ShapeTracker.from_shape((2, 3))
        )
        assert buffer_uop_contains_placeholder(non_buffer) is False

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (("DEFAULT", 6, "placeholder:test_tensor:1,2,3:float32"), "test_tensor"),
            (("DEFAULT", 6), None),
            (("DEFAULT", 6, "not_a_placeholder"), None),
        ],
    )
    def test_get_placeholder_name(
        self, test_input: Tuple[Any, ...], expected: str | None
    ) -> None:
        assert get_placeholder_name(test_input) == expected


class TestTensorTemplateManager:
    def test_create_placeholder_tensor(
        self, tensor_template_manager: TensorTemplateManager
    ) -> None:
        tensor = tensor_template_manager.create("test_tensor", (2, 3), dtypes.float32)

        assert tensor.shape == (2, 3)
        assert tensor.dtype == dtypes.float32

        buffer_uop = tensor.lazydata.src[1]  # Get the buffer UOp
        assert buffer_uop_contains_placeholder(buffer_uop)
        assert get_placeholder_name(buffer_uop.arg) == "test_tensor"

    def test_find_all_placeholders(
        self, tensor_template_manager: TensorTemplateManager
    ) -> None:
        t1 = tensor_template_manager.create("input1", (2, 3))
        t2 = tensor_template_manager.create("input2", (3, 4))

        result = t1 @ t2  # Matrix multiplication

        placeholders = find_all_placeholders(result.lazydata)
        assert "input1" in placeholders
        assert "input2" in placeholders
        assert len(placeholders) == 2

    def test_substitute_placeholder_uop(
        self, tensor_template_manager: TensorTemplateManager
    ) -> None:
        template1 = tensor_template_manager.create("input1", (2, 3))
        template2 = tensor_template_manager.create("input2", (3, 4))

        template_result = template1 @ template2

        real_input1 = Tensor.ones((2, 3))
        real_input2 = Tensor.ones((3, 4))

        inputs: ActualTensors = {"input1": real_input1, "input2": real_input2}
        final_uop = tensor_template_manager.substitute_placeholder_uop(
            template_result.lazydata, inputs
        )

        found_after = find_all_placeholders(final_uop)
        assert len(found_after) == 0  # No more placeholders


class TestTensorContext:
    def test_add_graph_input(self, tensor_context: TensorContext) -> None:
        tensor = tensor_context.add_graph_input("input", (2, 3), dtypes.float32)

        assert tensor.shape == (2, 3)
        assert tensor.dtype == dtypes.float32

        assert len(tensor_context.placeholders) == 1
        assert tensor_context.placeholders[0].name == "input"
        assert tensor_context.placeholders[0].shape == (2, 3)
        assert tensor_context.placeholders[0].dtype == dtypes.float32

    def test_compile_to_graph(
        self, tensor_context: TensorContext, tmp_path: Path
    ) -> None:
        input1 = tensor_context.add_graph_input("input1", (2, 3))
        input2 = tensor_context.add_graph_input("input2", (3, 4))

        result = input1 @ input2

        task = tensor_context.compile_to_graph(result)

        assert isinstance(task, GraphProgram)
        assert len(task.placeholders) == 2
        placeholder_names = [p.name for p in task.placeholders]
        assert "input1" in placeholder_names
        assert "input2" in placeholder_names

    def test_lazy_tensor_with_unknown_placeholders(
        self, tensor_context: TensorContext, tmp_path: Path
    ) -> None:
        input1 = tensor_context.add_graph_input("input1", (2, 3))

        outside_tensor = TensorTemplateManager().create("outside", (3, 4))

        result = input1 @ outside_tensor

        task = tensor_context.compile_to_graph(result)

        assert isinstance(task, ValueError)


class TestGraphProgram:
    @pytest.fixture
    def sample_task(self, tensor_context: TensorContext) -> GraphProgram:
        input1 = tensor_context.add_graph_input("input1", (2, 3))
        input2 = tensor_context.add_graph_input("input2", (3, 4))
        result = input1 @ input2
        return GraphProgram(result, tensor_context.placeholders)

    def test_to_bytes_and_from_bytes(self, sample_task: GraphProgram) -> None:
        data = sample_task.to_bytes()

        loaded_task = GraphProgram.from_bytes(data)

        assert not isinstance(loaded_task, ValueError)
        assert len(loaded_task.placeholders) == 2
        placeholder_names = [p.name for p in loaded_task.placeholders]
        assert "input1" in placeholder_names
        assert "input2" in placeholder_names

    def test_to_json_and_from_json(self, sample_task: GraphProgram) -> None:
        json_data = sample_task.to_json()

        loaded_task = GraphProgram.from_json(json_data)

        assert not isinstance(loaded_task, ValueError)
        assert len(loaded_task.placeholders) == 2
        placeholder_names = [p.name for p in loaded_task.placeholders]
        assert "input1" in placeholder_names
        assert "input2" in placeholder_names

    @pytest.mark.parametrize(
        "test_input,test_func",
        [
            (pickle.dumps({"not_a_task": True}), GraphProgram.from_bytes),
            ('{"not_a_task": true}', GraphProgram.from_json),
        ],
    )
    def test_invalid_data(
        self, test_input: Union[bytes, str], test_func: Callable
    ) -> None:
        result = test_func(test_input)
        assert isinstance(result, ValueError)


class TestHelperFunctions:
    def test_infer_tensor_context_from_weights(self) -> None:
        weights: ActualTensors = {
            "weight1": Tensor.ones((10, 5)),
            "weight2": Tensor.zeros((5, 3), dtype=dtypes.float16),
        }

        ctx = infer_tensor_context_from_weights(weights)

        assert len(ctx.placeholders) == 2

        for placeholder in ctx.placeholders:
            assert placeholder.name in weights
            tensor = weights[placeholder.name]
            assert placeholder.shape == tuple(tensor.shape)
            assert placeholder.dtype == tensor.dtype

    def test_execute_graph_on_gpu(self, tensor_context: TensorContext) -> None:
        input1 = tensor_context.add_graph_input("input1", (2, 3))
        input2 = tensor_context.add_graph_input("input2", (3, 4))

        result = input1 @ input2

        task = GraphProgram(result, tensor_context.placeholders)

        real_input1 = Tensor.ones((2, 3))
        real_input2 = Tensor.ones((3, 4))

        inputs: ActualTensors = {"input1": real_input1}
        weights: ActualTensors = {"input2": real_input2}

        complete_result = execute_graph_on_gpu(task, inputs, weights)

        assert not isinstance(complete_result, ValueError)
        assert complete_result.shape == (2, 4)

    def test_execute_graph_on_gpu_missing_inputs(
        self, tensor_context: TensorContext
    ) -> None:
        input1 = tensor_context.add_graph_input("input1", (2, 3))
        input2 = tensor_context.add_graph_input("input2", (3, 4))

        result = input1 @ input2

        task = GraphProgram(result, tensor_context.placeholders)

        inputs: ActualTensors = {"input1": Tensor.ones((2, 3))}
        weights: ActualTensors = {}

        complete_result = execute_graph_on_gpu(task, inputs, weights)

        assert isinstance(complete_result, ValueError)
