# EigenTensor

> the easiest way to deploy code onto GPUs ever. oh, and the result of each computation is verifiable.

## Purpose

Building GPU applications is hard. Whether you're deploying LLMs, doing scientific computing, or any other GPU-intensive task, you quickly run into limitations:

1. **Security Nightmares**: distributed GPU computing usually means running arbitrary code on other people's machines - this can lead to data theft, DDoS attacks, and even physical damage to the machine
2. **Leaky Abstractions**: high-level frameworks break down when you need custom functionality, leaving you to either write complex low-level code or dumb-down your application for the sake of shipping on time
3. **Cross-Platform Complexity**: getting different GPUs to behave the same way is a nightmare, especially when deterministic consensus is required   

EigenTensor solves this by taking the popular tinygrad library - known for its simple yet powerful syntax for writing memory-safe GPU code - and extending it to make it verifiable, distributed, and easy to deploy. With just ~3 lines of code added to any existing tinygrad codebase, you can:

1. Compile your GPU instructions into a universal, optimized format
2. Deploy them to any GPU node, without any risk of malicious code execution
3. Get a simple REST API for your GPU application, featuring one-click deployment

The cross-verification of results comes as standard, with a simple economic security model that makes it irrational for nodes to cheat.

No more wrestling with GPU complexities. No more framework limitations. Just write your computation in tinygrad, and EigenTensor handles the rest.

## The TinyGrad Advantage

EigenTensor leverages tinygrad as its computational backbone for several key reasons:

1. **Lazy Evaluation**: rather than immediately executing instructions, tinygrad builds out a graph representing all of the instructions which need to be executed on the GPU. This graph is not GPU-code, so it's memory-safe, cross-platform and can be optimized on a per-node basis. Yet, any two devices executing the same graph will get the same result, so it becomes possible to build a system of consensus around the result of each computation.
2. **Model Compatibility**: many existing ML models are already specified in tinygrad, making them immediately usable within EigenTensor
3. **Cross-Platform**: tinygrad runs efficiently across CPU, GPU, and other accelerators without code changes
4. **Optimization Features**: automatic kernel fusion, graph rewrites, and other optimizations boost performance
5. **Small Footprint**: lightweight codebase (<5000 lines) is easier to audit and less prone to vulnerabilities

## How It Works

EigenTensor consists of three main components:

### 1. Task Definition & Export

```python
# Create a tensor context
context = TensorContext()

# Define placeholder tensors
input_a = context.add_graph_input("matrix_a", (1000, 1000))
input_b = context.add_graph_input("matrix_b", (1000, 1000))

# Define computation (matmul in this example)
# This can get as complex as you want, you can write entire models if you want
result = input_a @ input_b

# Export the task for later use
task = context.compile_to_graph(result)
```

Remember, computations are executed lazily, so we're not actually computing anything yet.

### 2. Local Execution

```python
# Load an exported task
with open("matmul_task.pkl", "rb") as f:
    computational_graph = GraphProgram.from_bytes(f.read())

# Prepare actual input tensors
inputs = {
    "matrix_a": Tensor.ones((1000, 1000)),
    "matrix_b": Tensor.ones((1000, 1000))
}

# Execute remotely via API
result = execute_graph_on_gpu(computational_graph, inputs)
```

We also provide tools for fetching model weights over the internet or local cache, specified by a UUID (hash of the file).

```python
# Fetch weights from a URL
# URL is optional, if not provided the function will try to find the file in the local cache
weights = fetch_safetensors_by_uuid("1111-2222-3333-4444")

# Fetch a task from a URL
# URL is optional, if not provided the function will try to find the file in the local cache
task = fetch_exported_task_by_uuid("1111-2222-3333-4444")

# Execute the task with the weights
result = execute_graph_on_gpu(task, inputs, weights)
```

### 3. Distributed Execution

TODO!!

## Economic Security Model

The security of EigenTensor relies on game-theoretic principles of EigenLayer's restaking model. From the perspective of an individual operator node:

### Node Decision Payoff Matrix

| Strategy | Scenario | Expected Payoff |
|----------|----------|-----------------|
| **Honest Computation** | No check triggered | $\text{Payment} - \text{Computation Cost}$ |
| **Honest Computation** | Check triggered, others honest | $\text{Payment} - \text{Computation Cost}$ |
| **Honest Computation** | Check triggered, others collude | $-\text{Slashed Stake Value}$ |
| **Dishonest Computation** | No check triggered | $\text{Payment} - \text{Negligible Computation Cost}$ |
| **Dishonest Computation** | Check triggered, not caught | $\text{Payment} - \text{Negligible Computation Cost}$ |
| **Dishonest Computation** | Check triggered, caught | $-\text{Slashed Stake Value}$ |

### Expected Value Analysis

For a node with:
- Stake value $S$
- Payment per task $P$
- Computation cost $C$ 
- Check probability $\alpha$
- Collusion probability $\beta$
- Negligible computation cost $\epsilon C$ where $\epsilon \ll 1$
- Probability of being caught when checked $\gamma$

**Expected value of honesty**:
$(1-\alpha)(P-C) + \alpha(1-\beta)(P-C) + \alpha\beta(-S)$

**Expected value of dishonesty**:
$(1-\alpha)(P-\epsilon C) + \alpha(1-\gamma)(P-\epsilon C) + \alpha\gamma(-S)$

The system needs careful parameter tuning to ensure honesty remains profitable even under collusion attacks, while maintaining sufficient slashing penalties to deter widespread dishonesty.

Notably, by purchasing multiple GPUs, the operator could increase the probability of collusion.

## Getting Started

```bash
# Install the package
pip install pipenv

# Install the dependencies
pipenv install

# Build the package
pipenv run tox -e build

# Enter development shell
pipenv shell

# Install the package, if it still isn't installed
pip install -e .

# Run the API server, if you have set Environment Variables
python -m anytensor.api
```
---

EigenTensor is licensed under the GPL V3 License.
