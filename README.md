# EigenTensor

> the easiest way to deploy work onto a remote GPU ever. oh, and the result of each computation is verifiable.

TLDR: We implemented a universal, memory-safe, cross-platform format and runtime for GPU computations, which can run efficiently on any AVS node. We achieved this by reverse-engineering the undocumented internal representation of tinygrad, a popular open-source machine learning library. This gives developers access to the full extent of each GPU's capabilities without permitting malicious code execution, offering more functionality than Kite, Autonome, and Hyperbolic. It also supports tinygrad's familiar high-level API, often used by ML engineers. The key breakthrough came from exploiting a bug in tinygrad's BUFFER UOp implementation, which lets us substitute input values into the computational graph for execution on any GPU node. We also implemented basic consensus mechanisms to ensure the result of each computation is correct.

## Why?

Building GPU applications is hard. Whether you're deploying LLMs, doing scientific computing, or any other GPU-intensive task, you quickly run into limitations:

1. **Security Nightmares**: distributed GPU computing usually means running arbitrary code on other people's machines - this can lead to data theft, DDoS attacks, and even physical damage to the machine
2. **Leaky Abstractions**: high-level frameworks break down when you need custom functionality, leaving you to either write complex low-level code or dumb-down your application for the sake of shipping on time
3. **Cross-Platform Complexity**: getting different GPUs to behave the same way is a nightmare, especially when deterministic consensus is required   

EigenTensor solves this by taking the popular tinygrad library - known for its simple yet powerful syntax for writing memory-safe GPU code - and extending it to make it verifiable, distributed, and easy to deploy. With just ~3 lines of code added to any existing tinygrad codebase, you can:

1. Compile your GPU instructions into a universal, optimized format
2. Deploy them to any GPU node, without any risk of malicious code execution
3. Get a simple REST API for your GPU application, featuring one-click deployment

Results are automatically verified through consensus between multiple nodes. Our economic incentive model ensures nodes are rewarded for honest computation and penalized for dishonest behavior, making cheating unprofitable.

No more wrestling with GPU complexities. No more framework limitations. Just write your computation in tinygrad, and EigenTensor handles the rest.

## The TinyGrad Advantage

EigenTensor leverages tinygrad as its computational backbone for several key reasons:

1. **Lazy Evaluation**: rather than immediately executing instructions, tinygrad builds out a graph representing all operations to be executed on the GPU
2. **Memory Safety**: the computational graph is not low-level GPU code, making it memory-safe and cross-platform
3. **Consensus Support**: any two devices executing the same graph will get identical results, enabling consensus verification
4. **Model Compatibility**: many existing ML models are already specified in tinygrad, making them immediately usable
5. **Cross-Platform**: runs efficiently across CPU, GPU, and other accelerators without code changes
6. **Graph Optimization**: automatic kernel fusion and graph rewrites boost performance
7. **Per-Node Tuning**: the graph compiles into code which is heavily optimized for each specific node
8. **Small Footprint**: lightweight codebase (<5000 lines) is easier to audit and less prone to vulnerabilities

Alternative ways of accomplishing the same goals would either limit the applicability of EigenTensor to only certain types of computations, or be significantly more complex to use.

## How To Use EigenTensor

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

Since computations are executed lazily, defining `result` does not actually compute anything yet. Here, `result` is represented by a computational graph, which can be safely shared across multiple nodes.

This API may be slightly clunky for now, but it's entirely compatible with tinygrad's API and enables existing models to be used with minimal changes.

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

# Execute locally
result = execute_graph_on_gpu(computational_graph, inputs)
```

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

You can use our REST API to access the GPU.

```bash
PRIMARY_TASK_UUID="1111-2222-3333-4444"
PRIMARY_WEIGHTS_UUID="1111-2222-3333-4444"

# The server will automatically download the task and weights from the URLs, verifying their hashes against the provided UUID values.
anytensor-server 
```

For EigenLayer support, you also need to run the AVS client:

```bash
# TODO!!
# anytensor-avs-client
```

We've provided a dockerfile for convenience, which orchestrates the above steps.

```bash
docker build -t anytensor .
docker run -e PRIMARY_TASK_UUID="1111-2222-3333-4444" -e PRIMARY_WEIGHTS_UUID="1111-2222-3333-4444" anytensor
```

## The Hackery

The core innovation of EigenTensor came from discovering and exploiting a subtle implementation detail in TinyGrad's tensor representation system. 

TinyGrad operates by building computational graphs that represent operations on tensors, rather than executing operations immediately. When you add two tensors, TinyGrad creates a node in this graph connecting them with an ADD operation. The actual GPU computation only happens when you call `.realize()` on a tensor.

Our challenge was creating a system where computational graphs could be defined without specifying all input data upfront. We needed "placeholder" tensors that could be substituted with real data at execution time. This would allow us to share executable graphs that nodes could run with their own inputs.

The solution involved modifying how TinyGrad's BUFFER operation works. Normally, a BUFFER operation takes a two-element tuple that encodes tensor information like shape and size. We discovered we could extend this tuple with a third element containing a special placeholder label string, without breaking TinyGrad's internal operations. This allowed us to substitute input values into the computational graph by specifying the placeholder label.

This hack was necessary because TinyGrad's official metadata systems (like VOID patterns) were incompatible with graph composition operations - using them would have made it impossible to combine graphs involving placeholder tensors. Our approach was the only viable method we found after extensive testing.

With this technique, we could create computational graphs that referenced input tensors. These graphs could represent an entire ML model's execution process. When a node wants to run the computation, our system uses the undocumented graph-rewriting API of tinygrad to substitute the placeholders with actual tensors containing their data. We then serialize the graph using safe techniques, allowing it to be safely shared, stored, and executed later on any compatible GPU.

The consensus verification system built on top of this is inspired by my previous academic work on the EvML paper for Exo Labs, which modeled consensus for verifiable computation. This theoretical foundation informed our implementation of the economic security model that makes EigenTensor's execution trustworthy across distributed nodes.

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

Notably, by purchasing multiple GPUs, the operator could increase the probability of collusion. When there are enough possible nodes available for selection, it is difficult to influence the overall proportion of possible checking nodes that may jointly collude, so the probability of being caught when cheating remains high.

## Limitations

1. there's no way of delegating non-tensor computations to the AVS just yet, although this can be done fairly easily. support for non-tensor computations is important for popular LLMs, since the process of tokenizing and looping over tokens until the end of the sequence is a non-tensor operation. you can still perform this client-side, it's just not efficient.
2. the consensus features are not yet formally proven, although we have tested them they appear to work as expected.
3. some aspects of the code, such as our hacking around tinygrad and the system of deployment for this codebase, are prone to future breakage in future versions of tinygrad and Tangle.

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
