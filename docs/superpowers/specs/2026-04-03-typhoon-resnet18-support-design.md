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

### Fractal Conversion Rule

For a logical matrix `A[M, K]`, with matrix-unit tile shape `m0 x k0`:

- `M1 = ceil_div(M, m0)`
- `K1 = ceil_div(K, k0)`

The physical fractal forms are interpreted as:

- `zZ`: `[M1, K1, m0, k0]`
- `nZ`: `[K1, M1, m0, k0]`

Converting `nZ -> zZ` is therefore not a block-internal transpose. It is a reorder of the outer
fractal grid:

- `src[k1, m1, mi, ki] -> dst[m1, k1, mi, ki]`

This is considered a real data reorder. If it changes linear memory order, it must be represented
as a Typhoon `task_reshape`. Pure view changes do not emit a task.

## Convolution Mapping

`conv2d` is mapped through `im2col + matmul`.

The first stage does not introduce a new Typhoon `conv` task kind.

For each supported convolution block:

1. input activation is brought into SRAM via `task_dma`
2. convolution input is transformed into an `im2col` matrix
3. if `im2col` requires physical data reorder, emit `task_reshape`
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

Only the specific pooling patterns required by ResNet18 are supported. They may be lowered using a
restricted combination of `task_reshape` and `task_vector`, or a similarly constrained expansion,
as long as the resulting graph stays within the existing Typhoon task kinds.

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

- runtime ABI and scheduler unchanged
- codegen shape unchanged except for consuming the already-existing Typhoon IR

This isolates new work to automatic mapping logic rather than disturbing the existing simulator.

## Error Handling

Compilation should fail clearly when:

- the model is not recognized as the supported ResNet18 structure
- a required operator pattern is unsupported
- SRAM planning exceeds 1MB capacity for a required block/tile choice
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

### Integration Tests

- fixed-shape ResNet18 compiles to Typhoon task-DAG
- emitted graph passes `VerifyTyphoonGraph`
- generated C code contains Typhoon runtime ABI calls
- VM executable exports successfully

### Runtime Tests

- ResNet18 task graph runs through the existing simulator
- scheduler produces deterministic traces
- traces show overlap where dependencies and resources allow it
- numerical output is correct for the fixed-shape model within expected float32 tolerance

## Success Criteria

This stage is complete only when all of the following are true:

- `~/model/resnet18.onnx` with input `1 x 3 x 224 x 224` compiles successfully to Typhoon
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
