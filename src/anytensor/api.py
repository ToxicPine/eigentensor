###
# FastAPI Server
###

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rich.logging import RichHandler
from anytensor.core import execute_graph_on_gpu, GraphProgram, ActualTensors
from anytensor.serialize_tensors import TensorSerializer
from anytensor.storage_manager import (
    fetch_exported_task_by_uuid,
    fetch_safetensors_by_uuid,
)
from typing import Dict, Any, Optional
import logging

# Configure rich logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("anytensor")

# Store loaded tasks and weights
_tasks: Dict[str, GraphProgram] = {}
_weights: Dict[str, ActualTensors] = {}

# Get UUIDs from environment
PRIMARY_TASK_UUID = os.getenv("PRIMARY_TASK_UUID")
PRIMARY_WEIGHTS_UUID = os.getenv("PRIMARY_WEIGHTS_UUID")

if not PRIMARY_TASK_UUID or not PRIMARY_WEIGHTS_UUID:
    logger.error(
        "Required environment variables PRIMARY_TASK_UUID and PRIMARY_WEIGHTS_UUID not set"
    )
    exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load tasks and weights on startup
    try:
        logger.info("Loading primary task and weights...")
        # Try to load both task and weights
        if not PRIMARY_TASK_UUID:
            logger.error("PRIMARY_TASK_UUID is not set")
            raise ValueError("PRIMARY_TASK_UUID is not set")
        task = fetch_exported_task_by_uuid(PRIMARY_TASK_UUID)

        if not PRIMARY_WEIGHTS_UUID:
            logger.error("PRIMARY_WEIGHTS_UUID is not set")
            raise ValueError("PRIMARY_WEIGHTS_UUID is not set")

        if isinstance(task, ValueError):
            logger.error(f"Failed to load primary task: {task}")
            raise ValueError("Failed to load primary task")
        weights = fetch_safetensors_by_uuid(PRIMARY_WEIGHTS_UUID)
        if isinstance(weights, ValueError):
            logger.error(f"Failed to load primary weights: {weights}")
            raise ValueError("Failed to load primary weights")

        # Store them if successful
        _tasks[PRIMARY_TASK_UUID] = task
        _weights[PRIMARY_WEIGHTS_UUID] = weights
        logger.info("Successfully loaded primary task and weights")
        yield
    finally:
        logger.info("Clearing task and weight caches...")
        _tasks.clear()
        _weights.clear()


app = FastAPI(lifespan=lifespan)


class TaskInput(BaseModel):
    task_uuid: str
    weights_uuid: str
    input_tensors: Dict[str, bytes]


@app.post("/execute_graph_on_gpu")
async def handle_task(task_input: TaskInput):
    """Complete a task with provided inputs and weights."""
    # Check if task and weights exist in cache
    if task_input.task_uuid not in _tasks or task_input.weights_uuid not in _weights:
        logger.error(
            f"Task {task_input.task_uuid} or weights {task_input.weights_uuid} not found in cache"
        )
        raise HTTPException(
            status_code=404, detail="Task or weights not found in cache"
        )

    # Get task and weights from cache
    task = _tasks[task_input.task_uuid]
    weights = _weights[task_input.weights_uuid]
    input_tensors = {
        k: TensorSerializer.tensor_from_bytes(v)
        for k, v in task_input.input_tensors.items()
    }

    logger.info(f"Executing task {task_input.task_uuid}")
    # Complete the task
    result = execute_graph_on_gpu(task, input_tensors, weights)
    if isinstance(result, ValueError):
        logger.error(f"Failed to complete task: {result}")
        raise HTTPException(
            status_code=400, detail=f"Failed to complete task: {str(result)}"
        )

    # Pickle and return the result tensor
    try:
        logger.info("Task completed successfully, serializing result")
        return {"tensor": TensorSerializer.tensor_to_bytes(result)}
    except Exception as e:
        logger.error(f"Failed to serialize result: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to pickle result: {str(e)}"
        )
