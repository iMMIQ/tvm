# Typhoon Timing-Simulation Backend Design

## Overview

This document defines the first-stage design for a new TVM backend named `typhoon`.

The goal of this stage is not to map existing TVM operators automatically. The goal is to model
and simulate a new accelerator device with explicit SRAM management and task-graph execution,
using CPU-executable code as the simulation vehicle.

The design is driven by these hardware assumptions:

- Device codename: `typhoon`
- Current scope: single core only
- One core contains:
  - `1x matrix` compute unit
  - `1x vector` compute unit
  - `1x reshape` unit
  - `1x DMA` engine
  - `1MB` SRAM shared by all units
- The device does not execute a sequential instruction stream as the primary abstraction
- The device accepts a task DAG with explicit dependencies and schedules it internally
- All compute tasks must access data in SRAM
- SRAM is manually managed and should not be modeled as dynamic `alloc`-style memory
- SRAM contention exists in reality, but its exact arbitration behavior is not yet known

This stage prioritizes timing simulation. A temporary empirical cost model will be implemented for
development and debugging, and the design must leave a clean replacement point for a future
confidential binary cost model.

## Non-Goals

The following are explicitly out of scope for the first stage:

- Automatic lowering from Relax or Relay operators to Typhoon tasks
- Automatic mapping of standard TOPI/TE operators
- Multi-core execution and inter-core contention
- Detailed SRAM arbitration or bandwidth-sharing behavior between concurrent tasks
- Automatic SRAM address planning
- Real hardware ISA generation, firmware integration, or driver integration
- Production-quality performance modeling

## Design Choice

Three approaches were considered:

1. Encode dependencies using sequential `async/wait` style TIR
2. Represent the task DAG explicitly in TIR and submit it as a graph
3. Build a separate task-graph DSL outside TIR and convert it later

Approach 2 is selected.

The rationale is:

- It matches the hardware execution model more closely
- It makes timing simulation naturally task-centric
- It keeps the design inside TVM's existing `target + TIR + runtime` structure
- It avoids reconstructing a DAG from sequential code during simulation
- It leaves room for future multi-core scheduling and stronger memory analysis

## Execution Model

Each `PrimFunc` targeting `typhoon` corresponds to one submitted task graph.

The host side constructs a task graph through TIR intrinsics. These intrinsics do not mean
"execute now". They mean:

- declare SRAM regions
- define task nodes
- define dependencies
- submit the graph
- wait for completion when needed

The runtime simulator receives the graph description, performs dependency-driven scheduling, and
simulates task timing and resource occupancy.

## IR Shape

The first-stage TIR representation will treat the following concepts as first-class:

### 1. SRAM Regions

SRAM is represented as explicit static regions:

- `region_id`
- `offset`
- `size`
- `alignment`
- `tag`

These regions represent physical address intervals in the fixed 1MB SRAM space.

Tasks must reference SRAM through `region_id`, not through raw `offset/size` pairs and not through
generic TIR `Buffer` objects alone. The region table is therefore the single source of truth for
physical SRAM layout in the graph. Lowering may still carry underlying buffer handles for global
memory arguments, but compute-side SRAM operands are expressed in terms of declared regions.

### 2. Task Nodes

The first stage supports these task kinds only:

- `dma`
- `matmul`
- `vector`
- `reshape`

Each task is a static descriptor, not a mutable runtime object.

### 3. Dependency Edges

Dependencies are explicit graph edges between tasks. They are not inferred from a sequence of
blocking waits.

### 4. Graph Submission

The graph is submitted once using a submit-style intrinsic. The host may then wait for the whole
graph, but the graph itself is the unit of device execution.

## Task Descriptor Requirements

Every task descriptor must contain at least:

- `task_id`
- `kind`
- input region references or global-memory endpoints
- output region references or global-memory endpoints
- data type
- shape/workload parameters
- dependency list
- fields required by the cost model

Minimum fields per task kind:

### DMA

- direction: `global_to_sram` or `sram_to_global`
- byte count
- source global endpoint or source `region_id`
- destination global endpoint or destination `region_id`

### Matmul

- input `region_id` for `A` and `B`
- output `region_id` for `C`
- `M`, `N`, `K`
- layout metadata
- data type

### Vector

- vector op kind
- input/output `region_id`
- element count
- data type

### Reshape

- input/output `region_id`
- total element count
- layout or stride transform metadata

## Timing Model

Timing simulation is task-granular.

There is currently one instance of each functional resource per core:

- one DMA engine
- one matrix unit
- one vector unit
- one reshape unit

Scheduling rules:

- A task becomes ready when all dependencies are complete
- A ready task may start only when its required resource is free
- On start, the simulator records `start_time`
- On completion, the simulator records `end_time`
- Resource occupancy lasts for the full simulated latency of the task

The scheduler must be deterministic. If multiple tasks become ready for the same resource at the
same simulated timestamp, the tie-break rule is:

1. smaller `task_id` first
2. if needed, stable creation order as the secondary key

The same deterministic ordering rule should be used anywhere the simulator must choose between
multiple equally-ready tasks. Determinism is required for repeatable traces and tests.

This allows legal overlap such as:

- DMA with matrix
- DMA with vector
- vector with reshape

provided dependencies and resource availability permit it.

## SRAM Access Model

SRAM is a shared resource and must be modeled explicitly, but the exact contention behavior is not
yet known.

The first-stage design therefore uses a two-layer approach:

1. All tasks report SRAM read/write volume through a common SRAM access interface
2. The default timing implementation assumes a single task can saturate SRAM bandwidth and does not
   yet penalize concurrent tasks with extra slowdown

This keeps the simulator simple for initial development while preserving a clean insertion point
for future SRAM arbitration models.

Each task should report:

- bytes read from SRAM
- bytes written to SRAM
- access pattern metadata when useful

The simulator should record this information in trace output even if the default model does not
yet use it to slow down concurrent tasks.

## Cost Model Interface

The simulator must call a task-level latency estimator through a stable interface:

`EstimateLatency(task_desc, hw_config, sram_access_stats) -> latency`

The default estimator in this phase is an empirical development-only model.

Expected behavior:

- `dma`: fixed launch cost plus bytes/bandwidth term
- `matmul`: estimated from `M*N*K`, data type, and assumed throughput
- `vector`: estimated from element count, op kind, and assumed vector throughput
- `reshape`: light compute cost plus SRAM movement cost

This estimator is not expected to be accurate enough for product decisions. It exists to support
backend bring-up, trace inspection, and timing-debug workflows until a confidential binary cost
model is integrated later.

## SRAM Management Model

SRAM is not dynamically allocated during execution in the first stage.

Instead, each graph declares all SRAM regions up front. This creates a static region table for the
submitted graph. Each region is an explicitly named physical interval inside the fixed 1MB SRAM.

This design is chosen because it:

- matches the requirement to avoid `alloc`-style memory management
- makes timing traces easier to inspect
- simplifies correctness checks
- leaves room for a later address-planning pass without changing runtime semantics

The runtime simulator still owns a concrete 1MB SRAM byte array plus metadata for validation and
debugging.

## Validation Strategy

### Compile-Time Checks

The compiler side should perform at least the following checks:

- every SRAM region fits inside 1MB
- alignment requirements are satisfied
- declared SRAM regions do not overlap
- compute tasks only access SRAM regions
- only DMA tasks may cross between global memory and SRAM
- task dependencies are structurally valid
- task inputs are logically initialized before use

The first-stage implementation may use conservative checks. It does not need full symbolic memory
provenance.

For initialization tracking, the minimum rules are:

- `global_to_sram DMA` initializes the destination SRAM region
- compute tasks initialize their declared output SRAM regions
- `sram_to_global DMA` does not initialize SRAM; it only consumes SRAM input
- a task may read an SRAM region only if that region is an input graph region explicitly marked as
  preinitialized, or if it is initialized by an earlier dependency chain

### Runtime Checks

The runtime simulator should still perform defensive validation:

- out-of-bounds SRAM access
- read-before-initialize on SRAM regions
- output size mismatch
- illegal overlapping writes when they occur in execution
- graph/task metadata inconsistencies

Compile-time checks prevent common structural mistakes. Runtime checks remain essential for
debugging during bring-up.

## TVM Integration Scope

The first-stage implementation is intentionally limited to these layers:

- target registration
- TIR intrinsics for graph construction
- lowering from TIR intrinsics to runtime API calls
- runtime simulator
- Python helper APIs for authoring task graphs

The implementation should not depend on Relay or Relax integration in this phase.

The intended integration shape is:

### Target Layer

Add a new target kind named `typhoon`.

Initial target attributes should include at least:

- `sram_size`
- optional future-facing fields such as `num_cores`
- debug or trace toggles as needed

### TIR Layer

Define Typhoon-specific graph-construction intrinsics for:

- SRAM region declaration
- task creation
- dependency declaration
- graph submission
- whole-graph completion wait

The first-stage API should use a minimal, explicit graph-builder shape. Strawman intrinsic surface:

- `typhoon.region_decl(region_id, offset, size, alignment, tag)`
- `typhoon.task_dma(task_id, direction, src, dst, bytes, dep_task_ids...)`
- `typhoon.task_matmul(task_id, a_region_id, b_region_id, c_region_id, m, n, k, dtype, layout, dep_task_ids...)`
- `typhoon.task_vector(task_id, op_kind, in0_region_id, in1_region_id, out_region_id, elem_count, dtype, dep_task_ids...)`
- `typhoon.task_reshape(task_id, in_region_id, out_region_id, elem_count, transform_desc, dep_task_ids...)`
- `typhoon.submit_graph(graph_id)`
- `typhoon.wait_graph(graph_id)`

The exact spelling may change during implementation, but the model should stay explicit:

- regions are declared before use
- tasks refer to SRAM by `region_id`
- dependencies are part of task construction
- one graph is submitted once

### Lowering/Codegen Layer

Lower the Typhoon intrinsics into runtime API calls in host-executable generated code.

The first stage should reuse host-side code generation, rather than introducing a dedicated binary
device module format.

The runtime boundary should also be explicit. A strawman runtime API shape is:

- `TyphoonGraphBegin(ctx, graph_id)`
- `TyphoonDeclareRegion(ctx, graph_id, region_id, offset, size, alignment, tag)`
- `TyphoonAddDMATask(ctx, graph_id, task_desc)`
- `TyphoonAddMatmulTask(ctx, graph_id, task_desc)`
- `TyphoonAddVectorTask(ctx, graph_id, task_desc)`
- `TyphoonAddReshapeTask(ctx, graph_id, task_desc)`
- `TyphoonSubmitGraph(ctx, graph_id)`
- `TyphoonWaitGraph(ctx, graph_id)`

`task_desc` may be represented either as an explicit runtime struct or as flattened arguments in
the first implementation. The important constraint for planning is that the compiler/runtime
boundary be defined around graph construction and submission, not around immediate task execution.

### Runtime Layer

Implement the Typhoon simulator:

- graph ingestion
- dependency-driven scheduling
- resource tracking
- time progression
- SRAM state tracking
- trace generation
- cost model calls

## Implementation Staging

### Stage 1 Deliverable

The first stage is considered successful when the project has:

- a `typhoon` target
- a minimal set of Typhoon TIR intrinsics
- Python helpers to author task graphs
- lowering from TIR graph intrinsics to runtime simulator API calls
- a runtime simulator for one core
- a temporary empirical task-level cost model
- trace output for timing/debugging
- end-to-end examples and tests

Minimum end-to-end examples should include:

- `DMA -> matmul -> DMA`
- `DMA` overlapping with `vector`
- mixed `reshape/vector/matmul`
- invalid SRAM usage that fails validation

### Deferred Work

The following work should be deferred until after the first stage is stable:

- automatic lowering from high-level operators
- multi-core execution
- detailed SRAM contention
- automatic address planning
- integration with the confidential binary cost model

## Workload Estimate

This is a medium-to-large first-stage backend effort, but still bounded if the project stays
strictly within the selected scope.

Major work packages are:

- Typhoon target registration
- TIR intrinsic design and helper APIs
- lowering to runtime API
- simulator scheduler and trace engine
- SRAM validation and region tracking
- empirical task-level cost model
- tests and examples

Expected effort:

- rough proof-of-concept: about 2 to 4 weeks
- better engineering quality with cleaner interfaces, diagnostics, and tests: closer to 4+ weeks

The estimate assumes no automatic high-level operator mapping is attempted in the same phase.

## Open Assumptions Recorded Explicitly

These assumptions are intentional and should remain explicit in planning:

- only one core is modeled in phase 1
- each core has exactly one DMA, matrix, vector, and reshape unit
- SRAM contention exists, but the initial model does not yet impose slowdown on concurrent tasks
- one `PrimFunc` maps to one submitted graph
- timing simulation is the primary goal of this phase
- the default cost model is a temporary development substitute

## Planning Readiness

This spec is intended to be ready for implementation planning of a first-stage Typhoon timing
simulation backend. The plan should decompose work by subsystem, not by speculative future
features.
