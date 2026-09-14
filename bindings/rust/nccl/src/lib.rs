//! Safe Rust bindings for NCCL.
//!
//! The safe layer owns communicator/configuration/parameter resources and
//! deliberately keeps CUDA collective submission `unsafe`: NCCL enqueues work
//! asynchronously, so Rust cannot prove that caller-supplied device pointers
//! remain alive until their CUDA stream completes.
//!
//! The complete C ABI remains available through [`sys`].

use std::ffi::{c_char, c_int, c_void, CStr, CString, NulError};
use std::fmt;
use std::marker::PhantomData;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::slice;

pub use nccl_sys as sys;

include!("part1.rs");
include!("part2.rs");
include!("part3.rs");
include!("part4.rs");
