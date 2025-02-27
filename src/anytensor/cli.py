import click
import pathlib
import random
from typing import List
from tinygrad import Tensor, dtypes
from anytensor.core import TensorContext, GraphProgram, execute_graph_on_gpu


@click.group()
def cli():
    """CLI tool for working with safetensors and inspecting AST in tinygrad."""
    pass


@cli.command()
@click.option(
    "--output", "-o", type=click.Path(), required=True, help="Output pickle file"
)
@click.option(
    "--shape",
    "-s",
    type=str,
    default="10,10",
    help="Shape for tensors (comma-separated values)",
)
@click.option(
    "--dtype",
    "-d",
    type=click.Choice(["float32", "float16", "int32", "uint8"]),
    default="float32",
    help="Data type for tensors",
)
def placeholder_export(output, shape, dtype):
    """Generate tensors, use placeholder, export/import schedule and substitute."""
    output_path = pathlib.Path(output)
    if not output_path.name.endswith(".pkl"):
        output_path = output_path.with_suffix(".pkl")

    # Handle dtype
    dtype_map = {
        "float32": dtypes.float32,
        "float16": dtypes.float16,
        "int32": dtypes.int32,
        "uint8": dtypes.uint8,
    }
    tensor_dtype = dtype_map[dtype]

    # Parse shape
    tensor_shape = tuple(int(dim) for dim in shape.split(","))

    # Generate 5 random tensors
    tensors: List[Tensor] = []
    click.echo(f"\nGenerating 5 Random Tensors...")
    for i in range(5):
        tensor = Tensor.uniform(*tensor_shape, low=0, high=1, dtype=tensor_dtype)
        tensors.append(tensor)
        click.echo(f"  Created tensor_{i}: shape={tensor_shape}, dtype={dtype}")

    # Add all 5 tensors normally first
    click.echo("\n### Sum of all 5 tensors ###")
    full_sum = sum(tensors)
    print(full_sum.realize().tolist())

    # Create placeholder and add with first 4 tensors
    click.echo("\n### Creating placeholder and adding with first 4 tensors ###")
    tensor_context = TensorContext()
    placeholder_name = f"placeholder_{random.randint(10, 99)}"
    placeholder = tensor_context.add_graph_input(placeholder_name, tensor_shape, tensor_dtype)
    partial_sum = sum(tensors[:4]) + placeholder

    # Export the schedule
    click.echo("\n### Exporting task ###")
    task_bytes = tensor_context.compile_to_graph(partial_sum)
    if isinstance(task_bytes, ValueError):
        click.echo(f"Error exporting task: {task_bytes}")
        click.Abort()
    with open(output_path, "wb") as f:
        f.write(task_bytes.to_bytes())
    click.echo(f"Exported task to {output_path}")

    # Import and substitute
    click.echo("\n### Importing schedule and substituting ###")
    with open(output_path, "rb") as f:
        imported_bytes = f.read()

    exported_task = GraphProgram.from_bytes(imported_bytes)
    if isinstance(exported_task, ValueError):
        click.echo(f"Error importing task: {exported_task}")
        return

    result = execute_graph_on_gpu(exported_task, {placeholder_name: tensors[4]})
    if isinstance(result, ValueError):
        click.echo(f"Error substituting placeholders: {result}")
        return

    print("\nResult after substitution:")
    print(result.realize().tolist())


if __name__ == "__main__":
    cli()
