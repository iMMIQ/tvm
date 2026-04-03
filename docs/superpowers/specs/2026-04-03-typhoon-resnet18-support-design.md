# Typhoon ResNet18 Fixed-Shape Support Design

## Overview

This document defines the next-stage design for `typhoon`: supporting automatic compilation of a
fixed-shape `resnet18.onnx` model into the existing Typhoon task-DAG timing simulator.

This stage builds on the already implemented first-stage backend:

- `typhoon` target registration and host-C codegen
- `tirx.typhoon.*` graph-building ops
- compile-time and runtime graph validation
- single-core scheduler and trace generation
- task kinds: `dma`, `matmul`, `vector`, `reshape`

The key gap today is that generic Relax/VM compilation to `target="typhoon"` does not produce
Typhoon task graphs. It produces a VM executable tagged with the Typhoon target, but the model is
not lowered into `region_decl + task_* + submit_graph`.

This design closes that gap for one tightly scoped case:

- model: `resnet18.onnx`
- input shape: `1 x 3 x 224 x 224`
- datatype: `float32`
- layout: `NCHW`

The hardware SRAM capacity remains the same fixed assumption used by the stage-1 backend:

- `1MB` shared SRAM per core

## Goal

Compile the fixed-shape ResNet18 model into Typhoon task-DAG IR that:

- uses explicit SRAM regions
- models `conv2d` through `im2col + matmul`
- emits explicit `dma`, `matmul`, `vector`, and `reshape` tasks
- runs through the existing Typhoon host/runtime simulator
- produces timing traces based on the existing single-core multi-resource scheduler
- preserves numerically correct execution for this fixed-shape workload

## Non-Goals

The following remain out of scope for this stage:

- arbitrary ONNX model support
- dynamic shapes
- batch sizes other than `1`
- general CNN lowering beyond the ResNet18 patterns required here
- new Typhoon task kinds such as native `conv` or `pool`
- multi-core execution
- detailed SRAM contention modeling
- quantized datatypes such as `int8` or `fp16`

## Supported Model Scope

Only the fixed-shape ResNet18 inference graph is supported.

First-stage operator coverage is constrained to the patterns required by this graph:

- `conv2d`
- `dense`
- `bias/add`
- `relu`
- residual `add`
- average pool or fixed pool patterns that appear in ResNet18
- `reshape` / `flatten`

Support is defined by successful lowering of the canonical ResNet18 graph structure, not by a
promise of general support for every instance of these operators.

## Chosen Approach

Three approaches were considered:

1. Rewrite Relax operators directly into Typhoon task-DAG form
2. Lower through the normal model pipeline, then add a Typhoon-specific post-lowering pass that
   recognizes canonical TIR patterns and emits Typhoon tasks
3. Add a special-case importer for ResNet18 that bypasses existing lowering

Approach 2 is selected.

Rationale:

- It reuses the current ONNX -> Relax -> lower pipeline
- It keeps Typhoon-specific automatic mapping isolated from frontend import logic
- It matches the existing Typhoon TIR/TIRx runtime ABI design
- It allows future expansion by adding more pattern recognizers instead of replacing the pipeline
- It avoids hard-coding a one-off ONNX importer path

The implementation may initially behave like “Approach 2 plus narrow canonical-pattern support”,
but the architectural boundary remains a Typhoon-specific lowering pass rather than a model-specific
frontend fork.

## Matrix Layout Contract

The hardware matrix unit has fixed layout requirements:

- left matrix input must be `zZ`
- right matrix input must be `nN`
- output may be either `nN` or `zZ`

This design treats these as physical SRAM layouts for matrix-unit consumption, not as generic
logical tensor layouts visible to the model frontend.

### Dtype-Dependent Tile Shape

The matrix-unit fractal tile shape depends on datatype.

For this stage:

- only `float32` is supported
- `float32` uses `m0 = n0 = k0 = 8`

Future datatypes may use different fractal sizes. This stage does not define layout rules for
other dtypes beyond reserving that extension point.

### Fractal Layout Definitions

For a logical left-hand matrix `A[M, K]`:

- `M1 = ceil_div(M, m0)`
- `K1 = ceil_div(K, k0)`
- `zZ` is represented as `[M1, K1, m0, k0]`
- logical access shape is `src[m1, k1, mi, ki]`

For a logical right-hand matrix `B[K, N]`:

- `K1 = ceil_div(K, k0)`
- `N1 = ceil_div(N, n0)`
- `nN` is represented as `[K1, N1, k0, n0]`
- logical access shape is `src[k1, n1, ki, ni]`

The output layout may be either:

- `zZ`: `[M1, N1, m0, n0]`
- `nN`: `[N1, M1, n0, m0]`

The planner must choose the output layout explicitly and emit a real `task_reshape` whenever a
physical reorder is required.

### Fractal Conversion Rule

For left-hand matrix preparation, converting `nZ -> zZ` is not a block-internal transpose. It is a
reorder of the outer fractal grid:

- `src[k1, m1, mi, ki] -> dst[m1, k1, mi, ki]`

This is considered a real data reorder. If it changes linear memory order, it must be represented
as a Typhoon `task_reshape`. Pure view changes do not emit a task.

## Convolution Mapping

`conv2d` is mapped through `im2col + matmul`.

The first stage does not introduce a new Typhoon `conv` task kind.

For each supported convolution block:

1. input activation is brought into SRAM via `task_dma`
2. convolution input is transformed into an `im2col` matrix
3. `im2col` is emitted as `task_reshape`
4. in this stage, `task_reshape` is explicitly extended to cover shape-unit-native transforms that
   may duplicate data, including `im2col`
5. this extension is encoded by expanding reshape transform metadata, not by adding a new task kind
4. the left matrix is arranged as `zZ`
5. weights are brought into SRAM via `task_dma`
6. weights are arranged as `nN`
7. `task_matmul` performs the main convolution compute
8. bias and activation are handled by `task_vector`
9. the output stays in whichever matrix layout is most useful for the next consumer, unless a real
   reorder is required

## Other Operator Mappings

### Dense

Lower as:

- `dma(input)`
- `dma(weight)`
- optional `reshape`
- `task_matmul`
- optional `task_vector` for bias
- optional `task_vector` for activation

### Residual Add

Lower as:

- optional `task_reshape` to align physical layout
- `task_vector(op=add)`

### ReLU

Lower as `task_vector(op=relu)`

### Pool

No new pool unit is introduced in this stage.

Only the specific pooling patterns required by fixed-shape ResNet18 are supported:

- initial `maxpool2d(kernel=3x3, stride=2, padding=1)`
- final global average pool over the final spatial extent

These are lowered as `task_vector` operations because the vector unit is assumed to support pooling
natively in hardware.

This stage extends the vector op-code set with the exact pooling operations needed for fixed-shape
ResNet18. It does not define a generic pooling framework beyond these exact cases.

For planning purposes, this means:

- `task_vector` payload is extended with the exact parameters needed for the supported pooling ops
- `task_reshape` payload is extended with the exact parameters needed for `im2col`
- the task ABI shape remains the same at a high level: existing task kinds are reused

### Flatten / Reshape

- pure logical/view-only changes do not emit Typhoon tasks
- any physical reorder emits `task_reshape`

## SRAM Management Strategy

SRAM management remains explicit and compile-time planned.

This stage does not add a general-purpose allocator.

Instead, SRAM is divided into a fixed set of statically planned region classes:

- `ACT`: activation regions
- `WGT`: weight regions
- `COL`: `im2col` temporary regions
- `AUX`: layout-conversion, bias, and other temporary regions

Suggested region pool:

- `act0`
- `act1`
- `residual`
- `wgt0`
- `wgt1`
- `col0`
- `aux0`

Compilation computes concrete `offset/size/alignment` for each region and emits them as
`region_decl` statements. Tasks reference only `region_id`.

### Region Reuse

Region reuse is static and template-driven:

- no runtime allocation
- no generic linear-scan planner in this stage
- fixed double-buffering only where it clearly helps, especially for overlapping weight DMA with
  current-layer matmul

The primary success criterion is correct explicit SRAM planning for the fixed ResNet18 workload, not
 global optimality.

### Compile-Time Tiling Contract

This stage requires compile-time tiling.

Neither `conv2d` nor `dense` is executed as a whole-layer task. Both are lowered into tile-granular
task sequences.

For `float32`, the stage-1 fixed matmul tile is:

- `Mt = 64`
- `Nt = 64`
- `Kt = 64`

This is a fixed planning contract for the first implementation, not a search space.

Implications:

- each `task_matmul` handles one `64 x 64 x 64` tile, with edge tiles cropped at boundaries
- `im2col` is also performed tile-by-tile
- SRAM region planning, dependency structure, and task counts are defined around these compile-time
  tiles
- the first stage does not introduce an autotuned or per-layer tile policy

## Task-DAG Construction

Each major model layer expands into a sequence of Typhoon tasks.

The graph builder must emit explicit dependencies only where data ordering requires them. Resource
parallelism is then exposed naturally to the existing scheduler.

Expected overlap opportunities include:

- next-layer weight DMA with current-layer matmul
- reshape on one path with DMA on another path
- branch-local work before residual merge

The graph unit of execution remains:

- one `PrimFunc`
- one submitted Typhoon graph

## Compiler Pipeline Integration

The implementation is split into four logical layers.

### 1. Model Scope Gate

Reject anything outside the supported contract:

- model structure not matching supported ResNet18 patterns
- input shape not equal to `1 x 3 x 224 x 224`
- unsupported dtype/layout

This gate should fail early with a clear message.

### 2. ResNet18 Pattern Extractor

Recognize the canonical lowered patterns corresponding to the supported ResNet18 subgraphs and build
an internal layer plan.

This step performs recognition only. It does not emit Typhoon tasks directly.

Its input boundary is:

- post-generic model lowering, at the TIR representation that is stable enough to expose canonical
  computation patterns for ResNet18

Canonicalization into that matchable TIR form is in scope for this stage if the existing lowering
does not already provide it.

### 3. Typhoon SRAM Planner

Translate the layer plan into:

- explicit region assignments
- region reuse schedule
- task operand region references

### 4. Typhoon Graph Builder Pass

Emit the existing Typhoon task-DAG IR:

- `tirx.typhoon.region_decl`
- `tirx.typhoon.task_dma`
- `tirx.typhoon.task_matmul`
- `tirx.typhoon.task_vector`
- `tirx.typhoon.task_reshape`
- `tirx.typhoon.submit_graph`
- `tirx.typhoon.wait_graph`

This pass must run before the existing Typhoon lowering chain:

- after generic model lowering has produced the canonical TIR patterns to be recognized
- before `VerifyTyphoonGraph`
- `VerifyTyphoonGraph`
- `LowerTyphoonTaskDeps`
- `LowerTyphoonSubmitGraph`
- `LowerIntrin`

## File and Responsibility Boundaries

Recommended high-level split:

- model-scope gating near the Relax/TIR boundary
- a Typhoon-specific pattern extraction pass
- a Typhoon-specific SRAM planning module
- a Typhoon-specific graph-emission pass

The design intentionally keeps:

- the overall runtime ABI shape unchanged
- the scheduler resource model unchanged
- the existing task kind set unchanged

This stage explicitly does allow targeted Typhoon runtime-stack changes where the reused task kinds
gain richer payload semantics:

- Typhoon op/IR metadata for `task_reshape(im2col)` and `task_vector(pool)`
- `VerifyTyphoonGraph` validation for those payloads
- runtime execution semantics for those payloads
- cost-model accounting and trace reporting for those payloads

Codegen remains structurally the same because it still lowers to the existing `TVMTyphoon*`
entrypoints. What changes is the metadata carried by the reused task kinds.

This means the bulk of new work is still in automatic mapping logic, but the Typhoon validation and
runtime layers are also in scope where required to support the richer reshape/vector semantics.

## Error Handling

Compilation should fail clearly when:

- the model is not recognized as the supported ResNet18 structure
- a required operator pattern is unsupported
- SRAM planning exceeds 1MB capacity for the required `float32` block/tile choice
- a required layout conversion cannot be represented within the existing task kinds

Errors should identify:

- the offending layer or pattern
- the unsupported condition
- whether the failure is due to scope restriction or a true planning error

## Testing Strategy

Testing should be layered.

### Unit Tests

- fractal layout conversion planning
- `nZ -> zZ` reorder planning
- ResNet18 pattern recognition for supported canonical subgraphs
- SRAM region planning and reuse
- task-DAG emission for key layer types
- `im2col` lowering through `task_reshape`
- fixed ResNet18 pool lowering through `task_vector`

### Integration Tests

- fixed-shape ResNet18 compiles to Typhoon task-DAG
- emitted graph passes `VerifyTyphoonGraph`
- generated C code contains Typhoon runtime ABI calls
- VM executable exports successfully

The development entry path is:

- default model path: `~/model/resnet18.onnx`
- automated tests may override this through an environment variable so the path is reproducible

### Runtime Tests

- ResNet18 task graph runs through the existing simulator
- scheduler produces deterministic traces
- traces show overlap where dependencies and resources allow it
- numerical output is correct for the fixed-shape model within `float32` tolerance of
  `rtol=1e-4, atol=1e-4`

## Task Kind Layout Consumption Rules

The stage-1 plan relies on explicit layout-consumption rules so the compiler knows when a reshape is
mandatory.

- `task_dma`
  - GM side may be in ordinary logical tensor layout
  - SRAM side may produce or consume whatever physical layout the destination region declares
- `task_matmul`
  - left input must consume `zZ`
  - right input must consume `nN`
  - output may produce `zZ` or `nN`
- `task_vector`
  - first-stage vector tasks may consume and produce ordinary contiguous logical layout
  - any use on data currently held in matmul-specific fractal layout requires an explicit
    `task_reshape` before or after unless the particular vector op is later defined to support that
    fractal form
- `task_reshape`
  - may consume one physical layout and produce another physical layout
  - includes ordinary reorder transforms and `im2col`

## Success Criteria

This stage is complete only when all of the following are true:

- the configured ResNet18 ONNX path, defaulting to `~/model/resnet18.onnx`, with input
  `1 x 3 x 224 x 224` compiles successfully to Typhoon
- the emitted Typhoon-side C source contains `TVMTyphoon*` graph/runtime ABI calls
- the compiled model is represented as Typhoon task-DAG IR, not only as generic Relax/VM code
- `conv2d` is lowered through `im2col + matmul`
- left matrix layout is enforced as `zZ`
- right matrix layout is enforced as `nN`
- real layout reorders are emitted as `task_reshape`
- explicit SRAM regions are emitted and validated
- the model runs through the Typhoon timing simulator
- output is numerically correct for the fixed-shape workload
- scheduler trace reflects the existing single-core multi-resource timing model

## Deferred Work

The following are explicitly deferred to later stages:

- support for models other than ResNet18
- batch sizes other than `1`
- dynamic shapes
- generic CNN pattern support
- broader pool lowering
- datatype expansion beyond `float32`
- multi-core scheduling
- SRAM contention penalties
- a generalized SRAM planner

## Cost Model Update

The temporary empirical cost model must be updated for this stage.

For the first implementation, all supported task kinds share one common fixed launch-noise term:

- `latency = common_fixed_noise + workload_term`

Where:

- `common_fixed_noise` is the same constant for DMA, matmul, vector, and reshape
- `workload_term` remains task-kind specific

This is required so tile-size effects are visible:

- smaller tiles incur the same fixed noise more often
- therefore total latency grows as tiles become smaller, even if total arithmetic work is
  unchanged

The goal is not production accuracy. The goal is to preserve the qualitative behavior that
over-fragmenting a graph into many tiny tiles is slower because launch overhead accumulates.
