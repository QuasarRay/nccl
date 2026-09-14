/// An owned NCCL communicator.
///
/// The wrapper is intentionally neither `Send` nor `Sync`. Applications with
/// specialized cross-thread communicator ownership can use [`sys`] directly.
pub struct Communicator {
    raw: NonNull<sys::ncclComm>,
    closed: bool,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl Communicator {
    pub fn init_rank(nranks: i32, id: UniqueId, rank: i32) -> Result<Self> {
        validate_rank(nranks, rank)?;
        let mut raw = ptr::null_mut();
        // SAFETY: output pointer and by-value unique id are valid.
        check(unsafe { sys::ncclCommInitRank(&mut raw, nranks, id.raw(), rank) })?;
        Self::from_raw(raw, "ncclCommInitRank")
    }

    pub fn init_rank_with_config(
        nranks: i32,
        id: UniqueId,
        rank: i32,
        config: &mut Config,
    ) -> Result<Self> {
        validate_rank(nranks, rank)?;
        let mut raw = ptr::null_mut();
        // SAFETY: config owns all pointer fields for the duration of the call.
        check(unsafe {
            sys::ncclCommInitRankConfig(&mut raw, nranks, id.raw(), rank, config.raw_mut())
        })?;
        Self::from_raw(raw, "ncclCommInitRankConfig")
    }

    pub fn init_rank_scalable(
        nranks: i32,
        rank: i32,
        ids: &mut [UniqueId],
        config: &mut Config,
    ) -> Result<Self> {
        validate_rank(nranks, rank)?;
        if ids.is_empty() {
            return Err(Error::InvalidArgument("at least one unique ID is required"));
        }
        let n_id = i32::try_from(ids.len())
            .map_err(|_| Error::InvalidArgument("too many unique IDs"))?;
        let mut raw = ptr::null_mut();
        // SAFETY: UniqueId is repr(transparent) over ncclUniqueId.
        check(unsafe {
            sys::ncclCommInitRankScalable(
                &mut raw,
                nranks,
                rank,
                n_id,
                ids.as_mut_ptr().cast::<sys::ncclUniqueId>(),
                config.raw_mut(),
            )
        })?;
        Self::from_raw(raw, "ncclCommInitRankScalable")
    }

    fn from_raw(raw: sys::ncclComm_t, api: &'static str) -> Result<Self> {
        let raw = NonNull::new(raw).ok_or(Error::NullOnSuccess(api))?;
        Ok(Self {
            raw,
            closed: false,
            _not_send_or_sync: PhantomData,
        })
    }

    pub fn count(&self) -> Result<i32> {
        let mut value = 0;
        // SAFETY: live communicator and valid output pointer.
        check(unsafe { sys::ncclCommCount(self.raw.as_ptr(), &mut value) })?;
        Ok(value)
    }

    pub fn cuda_device(&self) -> Result<i32> {
        let mut value = 0;
        // SAFETY: live communicator and valid output pointer.
        check(unsafe { sys::ncclCommCuDevice(self.raw.as_ptr(), &mut value) })?;
        Ok(value)
    }

    pub fn rank(&self) -> Result<i32> {
        let mut value = 0;
        // SAFETY: live communicator and valid output pointer.
        check(unsafe { sys::ncclCommUserRank(self.raw.as_ptr(), &mut value) })?;
        Ok(value)
    }

    pub fn unique_id(&self) -> Result<UniqueId> {
        let mut raw = sys::ncclUniqueId {
            internal: [0 as c_char; sys::NCCL_UNIQUE_ID_BYTES],
        };
        // SAFETY: live communicator and valid output pointer.
        check(unsafe { sys::ncclCommGetUniqueId(self.raw.as_ptr(), &mut raw) })?;
        Ok(UniqueId(raw))
    }

    pub fn async_error(&self) -> Result<ResultCode> {
        let mut value = sys::ncclSuccess;
        // SAFETY: live communicator and valid output pointer.
        check(unsafe { sys::ncclCommGetAsyncError(self.raw.as_ptr(), &mut value) })?;
        Ok(ResultCode(value))
    }

    pub fn last_error(&self) -> String {
        // SAFETY: live communicator.
        let ptr = unsafe { sys::ncclGetLastError(self.raw.as_ptr()) };
        if ptr.is_null() {
            return String::new();
        }
        // SAFETY: NCCL returns a NUL-terminated string.
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }

    pub fn finalize(&mut self) -> Result<()> {
        // SAFETY: live communicator.
        check(unsafe { sys::ncclCommFinalize(self.raw.as_ptr()) })
    }

    pub fn revoke(&mut self) -> Result<()> {
        // SAFETY: live communicator and only supported flag value is used.
        check(unsafe { sys::ncclCommRevoke(self.raw.as_ptr(), sys::NCCL_REVOKE_DEFAULT) })
    }

    pub fn suspend_memory(&mut self) -> Result<()> {
        // SAFETY: live communicator.
        check(unsafe { sys::ncclCommSuspend(self.raw.as_ptr(), sys::NCCL_SUSPEND_MEM) })
    }

    pub fn resume(&mut self) -> Result<()> {
        // SAFETY: live communicator.
        check(unsafe { sys::ncclCommResume(self.raw.as_ptr()) })
    }

    pub fn memory_stat(&self, stat: MemoryStat) -> Result<u64> {
        let mut value = 0;
        // SAFETY: live communicator and valid output pointer.
        check(unsafe { sys::ncclCommMemStats(self.raw.as_ptr(), stat.as_raw(), &mut value) })?;
        Ok(value)
    }

    pub fn split(
        &self,
        color: Option<i32>,
        key: i32,
        mut config: Option<&mut Config>,
    ) -> Result<Option<Self>> {
        let mut raw = ptr::null_mut();
        let color = color.unwrap_or(sys::NCCL_SPLIT_NOCOLOR);
        let config_ptr = config
            .as_mut()
            .map_or(ptr::null_mut(), |cfg| cfg.raw_mut());
        // SAFETY: pointers are valid for the duration of this collective call.
        check(unsafe {
            sys::ncclCommSplit(self.raw.as_ptr(), color, key, &mut raw, config_ptr)
        })?;
        if raw.is_null() {
            return Ok(None);
        }
        Self::from_raw(raw, "ncclCommSplit").map(Some)
    }

    pub fn shrink(
        &self,
        exclude_ranks: &mut [i32],
        mut config: Option<&mut Config>,
        abort_parent_operations: bool,
    ) -> Result<Self> {
        let count = i32::try_from(exclude_ranks.len())
            .map_err(|_| Error::InvalidArgument("too many excluded ranks"))?;
        let mut raw = ptr::null_mut();
        let config_ptr = config
            .as_mut()
            .map_or(ptr::null_mut(), |cfg| cfg.raw_mut());
        let flags = if abort_parent_operations {
            sys::NCCL_SHRINK_ABORT
        } else {
            sys::NCCL_SHRINK_DEFAULT
        };
        // SAFETY: slice/config pointers remain valid during the call.
        check(unsafe {
            sys::ncclCommShrink(
                self.raw.as_ptr(),
                exclude_ranks.as_mut_ptr(),
                count,
                &mut raw,
                config_ptr,
                flags,
            )
        })?;
        Self::from_raw(raw, "ncclCommShrink")
    }

    pub fn destroy(mut self) -> Result<()> {
        // Transfer ownership to NCCL before the call. Even an error must not
        // cause Drop to retry destruction against a potentially consumed handle.
        self.closed = true;
        // SAFETY: live communicator owned by self and consumed by this method.
        check(unsafe { sys::ncclCommDestroy(self.raw.as_ptr()) })
    }

    pub fn abort(mut self) -> Result<()> {
        // ncclCommAbort frees communicator resources, so consuming self prevents
        // any subsequent safe method from using the invalidated C handle.
        self.closed = true;
        // SAFETY: live communicator owned by self and consumed by this method.
        check(unsafe { sys::ncclCommAbort(self.raw.as_ptr()) })
    }

    pub fn create_pre_mul_sum<T: NcclType + Copy>(
        &self,
        scalar: T,
    ) -> Result<OwnedReductionOp<'_>> {
        let mut scalar = scalar;
        let mut raw = sys::ncclSum;
        // SAFETY: host-immediate tells NCCL to copy/dereference scalar before
        // returning, so the stack pointer is not retained.
        check(unsafe {
            sys::ncclRedOpCreatePreMulSum(
                &mut raw,
                (&mut scalar as *mut T).cast::<c_void>(),
                T::DATA_TYPE.as_raw(),
                sys::ncclScalarHostImmediate,
                self.raw.as_ptr(),
            )
        })?;
        Ok(OwnedReductionOp { raw, comm: self })
    }

    /// Register memory for zero-copy use.
    ///
    /// # Safety
    ///
    /// `ptr..ptr+size` must denote a CUDA buffer acceptable to NCCL and remain
    /// allocated until the returned registration is dropped, after all
    /// asynchronous operations using it have completed.
    pub unsafe fn register<'a>(
        &'a self,
        ptr: *mut c_void,
        size: usize,
    ) -> Result<Registration<'a>> {
        if ptr.is_null() || size == 0 {
            return Err(Error::InvalidArgument("registered buffer must be non-null and non-empty"));
        }
        let mut handle = ptr::null_mut();
        // SAFETY: upheld by caller.
        check(unsafe {
            sys::ncclCommRegister(self.raw.as_ptr(), ptr, size, &mut handle)
        })?;
        let handle = NonNull::new(handle).ok_or(Error::NullOnSuccess("ncclCommRegister"))?;
        Ok(Registration { handle, comm: self })
    }

    /// Register a NCCL memory window.
    ///
    /// # Safety
    ///
    /// `ptr..ptr+size` must meet NCCL's window alignment/storage requirements
    /// and remain alive until the returned window is deregistered and all
    /// operations using it complete.
    pub unsafe fn register_window<'a>(
        &'a self,
        ptr: *mut c_void,
        size: usize,
        flags: i32,
    ) -> Result<Window<'a>> {
        if ptr.is_null() || size == 0 {
            return Err(Error::InvalidArgument("window buffer must be non-null and non-empty"));
        }
        let mut raw = ptr::null_mut();
        // SAFETY: upheld by caller.
        check(unsafe {
            sys::ncclCommWindowRegister(self.raw.as_ptr(), ptr, size, &mut raw, flags)
        })?;
        let raw = NonNull::new(raw).ok_or(Error::NullOnSuccess("ncclCommWindowRegister"))?;
        Ok(Window { raw, comm: self })
    }

    /// Enqueue all-reduce.
    ///
    /// # Safety
    ///
    /// `send` and `recv` must be valid CUDA-accessible buffers for `count`
    /// elements of `T`, satisfy NCCL's aliasing rules, and remain valid until
    /// `stream` has completed this operation. `op` must be valid for this
    /// communicator.
    pub unsafe fn all_reduce<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        op: ReductionOp<'_>,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclAllReduce(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                op.raw,
                self.raw.as_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue broadcast.
    ///
    /// # Safety
    ///
    /// The device pointers, root rank, and asynchronous lifetime requirements
    /// must satisfy NCCL's `ncclBroadcast` contract.
    pub unsafe fn broadcast<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        root: i32,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclBroadcast(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                root,
                self.raw.as_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue reduce.
    ///
    /// # Safety
    ///
    /// The device pointers, root rank, reduction op, and asynchronous lifetime
    /// requirements must satisfy NCCL's `ncclReduce` contract.
    pub unsafe fn reduce<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        op: ReductionOp<'_>,
        root: i32,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclReduce(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                op.raw,
                root,
                self.raw.as_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue all-gather.
    ///
    /// # Safety
    ///
    /// Device pointers must have the sizes NCCL derives from `send_count` and
    /// communicator size and remain alive until stream completion.
    pub unsafe fn all_gather<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        send_count: usize,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclAllGather(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                send_count,
                T::DATA_TYPE.as_raw(),
                self.raw.as_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue reduce-scatter.
    ///
    /// # Safety
    ///
    /// Device pointers and their asynchronous lifetimes must satisfy NCCL's
    /// `ncclReduceScatter` contract.
    pub unsafe fn reduce_scatter<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        recv_count: usize,
        op: ReductionOp<'_>,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclReduceScatter(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                recv_count,
                T::DATA_TYPE.as_raw(),
                op.raw,
                self.raw.as_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue all-to-all.
    ///
    /// # Safety
    ///
    /// Device pointers and their asynchronous lifetimes must satisfy NCCL's
    /// `ncclAlltoAll` contract.
    pub unsafe fn all_to_all<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclAlltoAll(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                self.raw.as_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue gather.
    ///
    /// # Safety
    ///
    /// Device pointers, root rank, and buffer sizes must satisfy NCCL's
    /// `ncclGather` contract, and all participating buffers must remain alive
    /// until stream completion.
    pub unsafe fn gather<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        root: i32,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclGather(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                root,
                self.raw.as_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue scatter.
    ///
    /// # Safety
    ///
    /// Device pointers, root rank, and buffer sizes must satisfy NCCL's
    /// `ncclScatter` contract, and all participating buffers must remain alive
    /// until stream completion.
    pub unsafe fn scatter<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        root: i32,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclScatter(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                root,
                self.raw.as_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue point-to-point send.
    ///
    /// # Safety
    ///
    /// `send` must remain valid until stream completion and the peer must issue
    /// a matching receive.
    pub unsafe fn send<T: NcclType>(
        &self,
        send: *const T,
        count: usize,
        peer: i32,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclSend(
                send.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                peer,
                self.raw.as_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue point-to-point receive.
    ///
    /// # Safety
    ///
    /// `recv` must remain valid and writable until stream completion and the
    /// peer must issue a matching send.
    pub unsafe fn recv<T: NcclType>(
        &self,
        recv: *mut T,
        count: usize,
        peer: i32,
        stream: CudaStream,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclRecv(
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                peer,
                self.raw.as_ptr(),
                stream.as_raw(),
            )
        })
    }

    /// Enqueue all-reduce with a per-collective config.
    ///
    /// # Safety
    ///
    /// Same requirements as [`Communicator::all_reduce`], plus every pointer
    /// reachable through `config` must remain valid for this call.
    pub unsafe fn all_reduce_config<T: NcclType>(
        &self,
        send: *const T,
        recv: *mut T,
        count: usize,
        op: ReductionOp<'_>,
        stream: CudaStream,
        config: &CollConfig,
    ) -> Result<()> {
        check(unsafe {
            sys::ncclAllReduceConfig(
                send.cast::<c_void>(),
                recv.cast::<c_void>(),
                count,
                T::DATA_TYPE.as_raw(),
                op.raw,
                self.raw.as_ptr(),
                stream.as_raw(),
                config.raw(),
            )
        })
    }

    fn raw(&self) -> sys::ncclComm_t {
        self.raw.as_ptr()
    }
}

impl Drop for Communicator {
    fn drop(&mut self) {
        if !self.closed {
            // SAFETY: self owns this communicator. Drop cannot report errors.
            let _ = unsafe { sys::ncclCommDestroy(self.raw.as_ptr()) };
            self.closed = true;
        }
    }
}

fn validate_rank(nranks: i32, rank: i32) -> Result<()> {
    if nranks <= 0 {
        return Err(Error::InvalidArgument("nranks must be positive"));
    }
    if rank < 0 || rank >= nranks {
        return Err(Error::InvalidArgument("rank must be in 0..nranks"));
    }
    Ok(())
}
