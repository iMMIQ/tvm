# Typhoon Full-Graph ResNet18 Design

## Goal

Compile fixed-shape `resnet18.onnx` (`1x3x224x224`, `float32`, `NCHW`) into Typhoon C ABI for the entire graph, execute the entire model in the Typhoon runtime simulator with no host fallback, and verify:

- runtime detects illegal SRAM overlap, out-of-bounds accesses, uninitialized reads, dependency mistakes, and concurrent write hazards
- final output is elementwise close to ONNXRuntime

## Non-Negotiable Requirements

- All ResNet18 operators must be supported by Typhoon lowering. Unsupported operators are a compile error.
- The compiled result must use Typhoon C ABI for the full graph. Host fallback is not allowed for ResNet18 execution.
- Numerical validation is against ONNXRuntime with elementwise comparison.
- Default acceptance tolerance is `rtol=1e-4, atol=1e-4`.
- Runtime safety checks must remain active during full-graph simulation.

## Current Baseline

- The current compiler/runtime can lower and simulate the ResNet18 stem through Typhoon.
- Current full-model execution is mixed-mode: only a local Typhoon subgraph is emitted, while the rest runs on the generic host path.
- Current recognition metadata reports `recognized_scope = "stem"`.
- This design upgrades the scope from local subgraph lowering to full-graph Typhoon lowering.

## Canonical Input Model Contract

This design is intentionally pinned to the local fixed-shape model artifact used by the current tests:

- model path convention: `~/model/resnet18.onnx` or `TYPHOON_RESNET18_ONNX_PATH`
- input shape: `1x3x224x224`
- dtype: `float32`
- layout: `NCHW`

The supported graph contract is the canonical graph observed from the current artifact after import through:

1. `from_onnx(..., keep_params_in_input=True)`
2. `relax.transform.DecomposeOpsForInference()`
3. `relax.transform.LegalizeOps()`
4. `relax.get_pipeline("default")`

The local ONNX artifact does not contain `BatchNormalization` nodes. Batch norm is therefore treated as already folded away upstream of Typhoon recognition and lowering. Supporting ONNX variants with explicit batch norm is out of scope for this design and should fail early instead of silently broadening recognition.

The canonical high-level operator family for the supported artifact is:

- `Conv`
- `Relu`
- `Identity`
- `Add`
- `MaxPool`
- `GlobalAveragePool`
- `Flatten`
- `Gemm`

The Typhoon full-graph recognizer may operate on the canonicalized Relax/TIR form of this graph, but it must remain contractually tied to this artifact family rather than claim support for arbitrary ResNet18 exporter variants.

## Recommended Approach

Use a staged full-graph approach:

1. Expand the compiler from stem-only recognition to fixed-topology full-graph recognition.
2. Expand SRAM planning from local-region assignment to whole-model lifetime-aware region reuse.
3. Emit Typhoon task DAGs for the entire model using the existing four task/resource families.
4. Expand runtime execution and validation so the resulting full graph can run under the existing simulator model.
5. Keep end-to-end validation anchored on full-model ONNXRuntime comparison throughout implementation.

This is preferred over a one-shot refactor because it preserves a clear debugging path while still converging on a no-fallback full-graph result.

## Architecture

### 1. Full-Graph Recognition and Planning

The Typhoon compiler front-end must stop recognizing only isolated lowered kernels and instead recover the complete fixed-shape ResNet18 topology. The output remains a serialized `typhoon_resnet18_plan`, but its scope changes from stem-only to full-graph.

The plan must include:

- `model`, `input_shape`, `dtype`
- `recognized_scope = "full_graph"`
- ordered stages, blocks, and layers
- layer kind per node
- layer input/output shapes
- residual/skip connectivity
- required layout transforms between layers
- whether a layer requires intermediate materialization such as `im2col`

The graph builder must consume this plan as the authoritative source of model structure. It must not rediscover topology by re-parsing TIR after planning.

### 2. Whole-Graph SRAM and Layout Planning

The SRAM planner must become full-graph and lifetime-aware. It should statically assign regions and reuse them only when lifetimes permit.

The plan must model at least these region roles:

- `ACT`
- `WGT`
- `COL`
- `AUX`
- `OUT`
- `RESIDUAL`

For each region assignment, the planner must record enough metadata to validate reuse decisions:

- region id, offset, size, alignment
- producing layer/task
- consuming layer/task window
- last use
- reuse point

`WGT` regions represent staged SRAM residency for the current layer or tile, not full-model persistent SRAM residency. Weight data may be reloaded into SRAM across layers and tiles through explicit DMA tasks, and the planner should treat weight residency as streamable unless a specific optimization chooses to keep a weight tile resident longer.

Special handling is required for residual blocks:

- skip-branch activations must stay live until the residual `add`
- planner must reject reuse that would clobber a live skip activation

The planner must reject full-graph schedules that cannot fit within the 1MB SRAM budget under the selected tiling and reuse policy.

### 3. Full-Graph Typhoon DAG Emission

The compiler should continue to target the existing Typhoon task families instead of introducing many new task kinds:

- `task_dma`
- `task_matmul`
- `task_vector`
- `task_reshape`

The full graph is then represented as a Typhoon task DAG assembled from these low-level operations.

Required operator lowering coverage for ResNet18:

- `conv2d`
- `add`
- `relu`
- `maxpool`
- `global average pool`
- final `dense`
- `reshape`
- `flatten`
- `transpose`
- logical/fractal layout conversion

Required low-level semantic coverage:

- `task_vector`: `add`, `relu`, `maxpool`, `global_avg_pool`
- `task_reshape`: identity/copy, `im2col`, flatten, transpose, logical/fractal layout conversion
- `task_matmul`: conv-backed GEMM tiles and final dense

The emitted result should be either a single full-graph Typhoon helper or a very small number of large Typhoon helpers, but the execution path for ResNet18 must remain entirely inside Typhoon C ABI.

### 4. Runtime and Simulator Execution

The runtime remains a functional plus timing simulator built around the current four resource classes:

- DMA
- matrix
- vector
- reshape

It must be extended to execute all task forms needed by the full ResNet18 graph.

Runtime responsibilities:

- functional execution of the full task DAG
- scheduling under the existing four-resource model
- safety validation for region use and task dependencies
- trace emission for debugging and validation

The design does not require cycle-accurate hardware modeling. It does require that the current simulator meaningfully models:

- explicit task dependencies
- resource occupancy
- SRAM reads/writes
- write hazards
- region lifetime misuse

## Safety Model

Correctness must be enforced in two layers.

### Compile-Time Guarantees

The compiler must guarantee:

- region offsets, sizes, and alignments are valid
- layer/task lifetime data is internally consistent
- peak live SRAM usage never exceeds 1MB
- no planned reuse violates residual or downstream live ranges

### Runtime Checks

The runtime must dynamically reject:

- overlapping region declarations
- out-of-bounds task accesses
- reads from undeclared or uninitialized regions
- illegal dependencies
- duplicate task ids
- concurrent write hazards
- region reads/writes that contradict the compiled lifetime plan

Compile-time legality is necessary but not sufficient. Runtime checks remain mandatory.

## Testing Strategy

### 1. Plan-Level Tests

Verify full-graph recognition:

- `recognized_scope == "full_graph"`
- all ResNet18 stages and basic blocks are present
- expected layer kinds and shapes appear in order
- residual edges and final head layers are represented

### 2. SRAM-Planning Tests

Verify whole-graph memory legality:

- all regions are in-bounds and non-overlapping
- peak live usage is within 1MB
- residual values stay live until merge
- reuse begins only after the final consumer

### 3. Codegen and C-ABI Tests

Verify actual emitted C source, not only intermediate TIR:

- full-graph Typhoon helper is present
- `TVMTyphoonDeclareRegion`
- `TVMTyphoonAddDMATask`
- `TVMTyphoonAddMatmulTask`
- `TVMTyphoonAddVectorTask`
- `TVMTyphoonAddReshapeTask`
- `TVMTyphoonSubmitGraph`

The emitted ResNet18 execution path must show no host fallback for model operators.

### 4. End-to-End Simulation Tests

Run the full compiled graph through the Typhoon simulator and compare against ONNXRuntime:

- fixed random seed
- elementwise output comparison with `rtol=1e-4, atol=1e-4`
- runtime trace must complete without safety violations

If ONNXRuntime is unavailable in a given environment, the test should fail with a clear missing-dependency message rather than silently weakening validation.

## Diagnostics and Observability

Implementation should expose enough artifacts to debug full-graph failures quickly:

- serialized full-graph plan
- serialized SRAM plan
- runtime trace JSON
- first-mismatch reporting for elementwise output failures
- task/layer context near the mismatch when available

The runtime trace should at minimum record:

- task id
- task kind
- resource class
- start time
- end time
- SRAM bytes read
- SRAM bytes written

It is desirable to extend this with region read/write ids for hazard debugging.

## Out of Scope

- Dynamic shapes
- Non-ResNet18 topologies
- Quantized or mixed-precision numerical modes
- Multi-core or multi-cluster Typhoon simulation
- Cycle-accurate hardware timing

## Acceptance Criteria

This design is complete when all of the following are true:

1. `resnet18.onnx` compiles into full-graph Typhoon C ABI with no host fallback.
2. The generated C source clearly contains full-graph Typhoon task emission.
3. The Typhoon runtime simulator executes the entire model without memory or dependency violations.
4. Final outputs are elementwise close to ONNXRuntime under `rtol=1e-4, atol=1e-4`.
5. The trace and planning artifacts are sufficient to diagnose full-graph scheduling or memory bugs.
