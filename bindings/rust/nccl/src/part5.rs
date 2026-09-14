/// Description of one NCCL signal wait condition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WaitSignal {
    pub operation_count: i32,
    pub peer: i32,
    pub signal_index: i32,
    pub context: i32,
}

impl WaitSignal {
    fn as_raw(self) -> sys::ncclWaitSignalDesc_t {
        sys::ncclWaitSignalDesc_t {
            opCnt: self.operation_count,
            peer: self.peer,
            sigIdx: self.signal_index,
            ctx: self.context,
        }
    }
}

impl Communicator {
    /// Enqueue broadcast with a per-collective config.
    ///
    /// # Safety
    ///
    /// The device pointers, root rank, and asynchronous lifetime requirements
    /// must satisfy NCCL's `ncclBroadcastConfig` contract. Every pointer
    /// reachable through `config` must remain valid for the duration of the call.
    pub unsafe fn broadcast_config<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        root: i32,
        stream: CudaStream,
        config: &CollConfig,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclBroadcastConfig(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                root,
                self.raw(),
                stream.as_raw(),
                config.raw(),
            )
        })
    }

    /// Enqueue reduce with a per-collective config.
    ///
    /// # Safety
    ///
    /// The device pointers, root rank, reduction op, and asynchronous lifetime
    /// requirements must satisfy NCCL's `ncclReduceConfig` contract. Every
    /// pointer reachable through `config` must remain valid for the call.
    pub unsafe fn reduce_config<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        op: ReductionOp<'_>,
        root: i32,
        stream: CudaStream,
        config: &CollConfig,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclReduceConfig(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                op.raw,
                root,
                self.raw(),
                stream.as_raw(),
                config.raw(),
            )
        })
    }

    /// Enqueue all-gather with a per-collective config.
    ///
    /// # Safety
    ///
    /// Device pointers must have the sizes NCCL derives from `send_count` and
    /// communicator size and remain alive until stream completion. Every pointer
    /// reachable through `config` must remain valid for the call.
    pub unsafe fn all_gather_config<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        send_count: usize,
        stream: CudaStream,
        config: &CollConfig,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclAllGatherConfig(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                send_count,
                T::DATA_TYPE.as_raw(),
                self.raw(),
                stream.as_raw(),
                config.raw(),
            )
        })
    }

    /// Enqueue reduce-scatter with a per-collective config.
    ///
    /// # Safety
    ///
    /// Device pointers, reduction op, and asynchronous lifetime requirements
    /// must satisfy NCCL's `ncclReduceScatterConfig` contract. Every pointer
    /// reachable through `config` must remain valid for the call.
    pub unsafe fn reduce_scatter_config<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        recv_count: usize,
        op: ReductionOp<'_>,
        stream: CudaStream,
        config: &CollConfig,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclReduceScatterConfig(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                recv_count,
                T::DATA_TYPE.as_raw(),
                op.raw,
                self.raw(),
                stream.as_raw(),
                config.raw(),
            )
        })
    }

    /// Enqueue all-to-all with a per-collective config.
    ///
    /// # Safety
    ///
    /// Device pointers and asynchronous lifetime requirements must satisfy
    /// NCCL's `ncclAlltoAllConfig` contract. Every pointer reachable through
    /// `config` must remain valid for the call.
    pub unsafe fn all_to_all_config<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        stream: CudaStream,
        config: &CollConfig,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclAlltoAllConfig(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                self.raw(),
                stream.as_raw(),
                config.raw(),
            )
        })
    }

    /// Enqueue gather with a per-collective config.
    ///
    /// # Safety
    ///
    /// Device pointers, root rank, buffer sizes, and asynchronous lifetime
    /// requirements must satisfy NCCL's `ncclGatherConfig` contract. Every
    /// pointer reachable through `config` must remain valid for the call.
    pub unsafe fn gather_config<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        root: i32,
        stream: CudaStream,
        config: &CollConfig,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclGatherConfig(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                root,
                self.raw(),
                stream.as_raw(),
                config.raw(),
            )
        })
    }

    /// Enqueue scatter with a per-collective config.
    ///
    /// # Safety
    ///
    /// Device pointers, root rank, buffer sizes, and asynchronous lifetime
    /// requirements must satisfy NCCL's `ncclScatterConfig` contract. Every
    /// pointer reachable through `config` must remain valid for the call.
    pub unsafe fn scatter_config<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        root: i32,
        stream: CudaStream,
        config: &CollConfig,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclScatterConfig(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                root,
                self.raw(),
                stream.as_raw(),
                config.raw(),
            )
        })
    }

    /// Enqueue one-sided data transfer followed by a signal.
    ///
    /// # Safety
    ///
    /// `local` must point to at least `count` CUDA-accessible values of `T` and
    /// remain alive until `stream` completes. `peer_window` and
    /// `peer_window_offset` must identify a valid remote window region large
    /// enough for the transfer, and the signal/context indices must satisfy the
    /// communicator configuration.
    pub unsafe fn put_signal<T: NcclType>(
        &self,
        local: *const T,
        count: usize,
        peer: i32,
        peer_window: &Window<'_>,
        peer_window_offset: usize,
        signal_index: i32,
        context: i32,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclPutSignal(
                local.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                peer,
                peer_window.raw.as_ptr(),
                peer_window_offset,
                signal_index,
                context,
                0,
                self.raw(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue a signal to `peer` without transferring data.
    pub fn signal(
        &self,
        peer: i32,
        signal_index: i32,
        context: i32,
        stream: CudaStream,
    ) -> Result<()> {
        // SAFETY: there are no caller-owned buffers. CudaStream construction
        // establishes the validity requirement for the stream handle.
        check(unsafe {
            sys::ncclSignal(
                peer,
                signal_index,
                context,
                0,
                self.raw(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue a wait for all supplied signal conditions.
    pub fn wait_signals(&self, waits: &[WaitSignal], stream: CudaStream) -> Result<()> {
        let count = i32::try_from(waits.len())
            .map_err(|_| Error::InvalidArgument("too many signal descriptors"))?;
        if waits.iter().any(|wait| wait.operation_count < 0) {
            return Err(Error::InvalidArgument(
                "signal operation counts must be non-negative",
            ));
        }

        let mut raw: Vec<_> = waits.iter().copied().map(WaitSignal::as_raw).collect();
        // NCCL consumes the host descriptor array while enqueueing the operation;
        // the array itself is not a device-side asynchronous input.
        check(unsafe {
            sys::ncclWaitSignal(count, raw.as_mut_ptr(), self.raw(), stream.as_raw())
        })
    }
}
