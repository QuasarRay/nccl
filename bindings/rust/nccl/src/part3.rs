/// NCCL communicator memory-statistic selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryStat {
    SuspendableBytes,
    Suspended,
    PersistentBytes,
    TotalBytes,
}

impl MemoryStat {
    const fn as_raw(self) -> sys::ncclCommMemStat_t {
        match self {
            Self::SuspendableBytes => sys::ncclStatGpuMemSuspend,
            Self::Suspended => sys::ncclStatGpuMemSuspended,
            Self::PersistentBytes => sys::ncclStatGpuMemPersist,
            Self::TotalBytes => sys::ncclStatGpuMemTotal,
        }
    }
}

/// Owned dynamic reduction operator.
pub struct OwnedReductionOp<'c> {
    raw: sys::ncclRedOp_t,
    comm: &'c Communicator,
}

impl OwnedReductionOp<'_> {
    pub fn op(&self) -> ReductionOp<'_> {
        ReductionOp {
            raw: self.raw,
            _lifetime: PhantomData,
        }
    }
}

impl Drop for OwnedReductionOp<'_> {
    fn drop(&mut self) {
        // SAFETY: op was created for this communicator and is destroyed once.
        let _ = unsafe { sys::ncclRedOpDestroy(self.raw, self.comm.raw()) };
    }
}

/// RAII registration returned by `ncclCommRegister`.
pub struct Registration<'c> {
    handle: NonNull<c_void>,
    comm: &'c Communicator,
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        // SAFETY: handle belongs to this live communicator.
        let _ = unsafe { sys::ncclCommDeregister(self.comm.raw(), self.handle.as_ptr()) };
    }
}

/// RAII NCCL memory window.
pub struct Window<'c> {
    raw: NonNull<sys::ncclWindow_vidmem>,
    comm: &'c Communicator,
}

impl Window<'_> {
    pub fn user_ptr(&self) -> Result<*mut c_void> {
        let mut ptr = ptr::null_mut();
        // SAFETY: window and communicator remain live through self.
        check(unsafe { sys::ncclWinGetUserPtr(self.comm.raw(), self.raw.as_ptr(), &mut ptr) })?;
        Ok(ptr)
    }

    pub fn as_raw(&self) -> sys::ncclWindow_t {
        self.raw.as_ptr()
    }
}

impl Drop for Window<'_> {
    fn drop(&mut self) {
        // SAFETY: window belongs to this communicator and is deregistered once.
        let _ = unsafe { sys::ncclCommWindowDeregister(self.comm.raw(), self.raw.as_ptr()) };
    }
}
