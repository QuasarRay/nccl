# Rust bindings for NCCL

This directory contains first-party Rust bindings for the public NCCL C API.

- `nccl-sys` mirrors the complete ABI in `src/nccl.h.in`, including communicator management, collectives, RMA/window APIs, group semantics, and runtime parameter access. It uses the C ABI directly and does not require CUDA headers to compile.
- `nccl` provides safe ownership for communicators, configs, NCCL allocations, registrations, windows, dynamic reduction operators, unique IDs, and parameter handles. Collective submission is intentionally `unsafe` because NCCL enqueues asynchronous CUDA work and Rust cannot prove raw device-buffer lifetime.

## Build

By default `nccl-sys` links dynamically against `libnccl`.

If NCCL is not in the system linker path, set either:

```sh
export NCCL_HOME=/path/to/nccl
# or
export NCCL_LIB_DIR=/path/to/nccl/lib
```

Then build with:

```sh
cargo build --manifest-path bindings/rust/Cargo.toml
```

For source-only CI, documentation, or static analysis on a machine without NCCL, the workspace provides a `no-link` feature:

```sh
cargo check --manifest-path bindings/rust/Cargo.toml --workspace --all-targets --features no-link
```

`no-link` is not suitable for running a program that calls NCCL.

## Example

```rust,no_run
use nccl::{Communicator, CudaStream, Reduction, UniqueId};

fn enqueue(
    comm: &Communicator,
    send: *const f32,
    recv: *mut f32,
    count: usize,
) -> nccl::Result<()> {
    // SAFETY:
    // - send and recv are CUDA-accessible buffers containing count f32 values;
    // - they remain allocated until the default CUDA stream completes;
    // - the buffers satisfy NCCL's aliasing/in-place requirements.
    unsafe {
        comm.all_reduce(
            send,
            recv,
            count,
            Reduction::Sum.op(),
            CudaStream::DEFAULT,
        )
    }
}

fn make_rank(id: UniqueId, nranks: i32, rank: i32) -> nccl::Result<Communicator> {
    Communicator::init_rank(nranks, id, rank)
}
```

## Safety model

The high-level crate does not turn CUDA pointers into Rust references or slices. That would imply host dereferenceability and lifetimes that NCCL cannot guarantee. Instead, typed collective methods validate the NCCL datatype at compile time but remain `unsafe` at the enqueue boundary.

RAII wrappers ensure local NCCL handles are destroyed or deregistered once. Group scope handling also attempts `ncclGroupEnd` if the Rust closure returns an error or panics, preventing a successful `ncclGroupStart` from being silently left open.

The complete raw API is available as `nccl::sys` for advanced APIs and threading patterns not represented by the conservative safe layer.
