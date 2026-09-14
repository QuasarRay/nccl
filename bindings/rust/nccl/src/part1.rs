/// Result type used by the safe wrapper.
pub type Result<T> = std::result::Result<T, Error>;

/// A NCCL result code, including future codes unknown to this crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct ResultCode(pub sys::ncclResult_t);

impl ResultCode {
    pub const SUCCESS: Self = Self(sys::ncclSuccess);
    pub const UNHANDLED_CUDA_ERROR: Self = Self(sys::ncclUnhandledCudaError);
    pub const SYSTEM_ERROR: Self = Self(sys::ncclSystemError);
    pub const INTERNAL_ERROR: Self = Self(sys::ncclInternalError);
    pub const INVALID_ARGUMENT: Self = Self(sys::ncclInvalidArgument);
    pub const INVALID_USAGE: Self = Self(sys::ncclInvalidUsage);
    pub const REMOTE_ERROR: Self = Self(sys::ncclRemoteError);
    pub const IN_PROGRESS: Self = Self(sys::ncclInProgress);
    pub const TIMEOUT: Self = Self(sys::ncclTimeout);

    pub const fn is_success(self) -> bool {
        self.0 == sys::ncclSuccess
    }

    pub const fn as_raw(self) -> sys::ncclResult_t {
        self.0
    }

    pub fn description(self) -> String {
        // SAFETY: ncclGetErrorString accepts any ncclResult_t value.
        let ptr = unsafe { sys::ncclGetErrorString(self.0) };
        if ptr.is_null() {
            return format!("NCCL error {}", self.0);
        }
        // SAFETY: NCCL returns a stable NUL-terminated string.
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }
}

/// Errors from either NCCL or the Rust binding's argument/lifetime guards.
#[derive(Debug)]
pub enum Error {
    Nccl(ResultCode),
    InvalidArgument(&'static str),
    InteriorNul(NulError),
    NullOnSuccess(&'static str),
}

impl Error {
    pub fn nccl_code(&self) -> Option<ResultCode> {
        match self {
            Self::Nccl(code) => Some(*code),
            _ => None,
        }
    }
}

impl From<NulError> for Error {
    fn from(value: NulError) -> Self {
        Self::InteriorNul(value)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nccl(code) => write!(f, "{}", code.description()),
            Self::InvalidArgument(message) => f.write_str(message),
            Self::InteriorNul(_) => f.write_str("string contains an interior NUL byte"),
            Self::NullOnSuccess(api) => write!(f, "{api} returned success with a null handle"),
        }
    }
}

impl std::error::Error for Error {}

fn check(code: sys::ncclResult_t) -> Result<()> {
    if code == sys::ncclSuccess {
        Ok(())
    } else {
        Err(Error::Nccl(ResultCode(code)))
    }
}

/// Runtime NCCL version code.
pub fn version() -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a valid output pointer.
    check(unsafe { sys::ncclGetVersion(&mut value) })?;
    Ok(value)
}

/// Opaque NCCL communicator bootstrap identifier.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct UniqueId(sys::ncclUniqueId);

impl UniqueId {
    pub fn generate() -> Result<Self> {
        let mut raw = sys::ncclUniqueId {
            internal: [0 as c_char; sys::NCCL_UNIQUE_ID_BYTES],
        };
        // SAFETY: `raw` is a correctly sized output object.
        check(unsafe { sys::ncclGetUniqueId(&mut raw) })?;
        Ok(Self(raw))
    }

    pub fn from_bytes(bytes: [u8; sys::NCCL_UNIQUE_ID_BYTES]) -> Self {
        let mut internal = [0 as c_char; sys::NCCL_UNIQUE_ID_BYTES];
        for (dst, src) in internal.iter_mut().zip(bytes) {
            *dst = src as c_char;
        }
        Self(sys::ncclUniqueId { internal })
    }

    pub fn to_bytes(self) -> [u8; sys::NCCL_UNIQUE_ID_BYTES] {
        let mut bytes = [0_u8; sys::NCCL_UNIQUE_ID_BYTES];
        for (dst, src) in bytes.iter_mut().zip(self.0.internal) {
            *dst = src as u8;
        }
        bytes
    }

    fn raw(self) -> sys::ncclUniqueId {
        self.0
    }
}

/// Opaque CUDA stream handle.
///
/// `DEFAULT` represents CUDA's null/default stream.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaStream(sys::cudaStream_t);

impl CudaStream {
    pub const DEFAULT: Self = Self(ptr::null_mut());

    /// Construct from a CUDA runtime stream handle.
    ///
    /// # Safety
    ///
    /// `raw` must be null or a valid `cudaStream_t` for every operation where
    /// this value is used.
    pub const unsafe fn from_raw(raw: sys::cudaStream_t) -> Self {
        Self(raw)
    }

    pub const fn as_raw(self) -> sys::cudaStream_t {
        self.0
    }
}

/// NCCL element type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataType {
    Int8,
    Uint8,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Float16,
    Float32,
    Float64,
    Bfloat16,
    Float8E4M3,
    Float8E5M2,
}

impl DataType {
    pub const fn as_raw(self) -> sys::ncclDataType_t {
        match self {
            Self::Int8 => sys::ncclInt8,
            Self::Uint8 => sys::ncclUint8,
            Self::Int32 => sys::ncclInt32,
            Self::Uint32 => sys::ncclUint32,
            Self::Int64 => sys::ncclInt64,
            Self::Uint64 => sys::ncclUint64,
            Self::Float16 => sys::ncclFloat16,
            Self::Float32 => sys::ncclFloat32,
            Self::Float64 => sys::ncclFloat64,
            Self::Bfloat16 => sys::ncclBfloat16,
            Self::Float8E4M3 => sys::ncclFloat8e4m3,
            Self::Float8E5M2 => sys::ncclFloat8e5m2,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Rust scalar types with a native NCCL datatype mapping.
pub trait NcclType: sealed::Sealed {
    const DATA_TYPE: DataType;
}

macro_rules! impl_nccl_type {
    ($ty:ty, $kind:expr) => {
        impl sealed::Sealed for $ty {}
        impl NcclType for $ty {
            const DATA_TYPE: DataType = $kind;
        }
    };
}

impl_nccl_type!(i8, DataType::Int8);
impl_nccl_type!(u8, DataType::Uint8);
impl_nccl_type!(i32, DataType::Int32);
impl_nccl_type!(u32, DataType::Uint32);
impl_nccl_type!(i64, DataType::Int64);
impl_nccl_type!(u64, DataType::Uint64);
impl_nccl_type!(f32, DataType::Float32);
impl_nccl_type!(f64, DataType::Float64);

/// Bit-preserving storage type for CUDA half values.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct F16(pub u16);
impl_nccl_type!(F16, DataType::Float16);

/// Bit-preserving storage type for CUDA bfloat16 values.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct Bf16(pub u16);
impl_nccl_type!(Bf16, DataType::Bfloat16);

/// Bit-preserving storage type for CUDA e4m3 fp8 values.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct Fp8E4M3(pub u8);
impl_nccl_type!(Fp8E4M3, DataType::Float8E4M3);

/// Bit-preserving storage type for CUDA e5m2 fp8 values.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct Fp8E5M2(pub u8);
impl_nccl_type!(Fp8E5M2, DataType::Float8E5M2);

/// Built-in NCCL reduction operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reduction {
    Sum,
    Product,
    Max,
    Min,
    Average,
}

impl Reduction {
    pub const fn op(self) -> ReductionOp<'static> {
        let raw = match self {
            Self::Sum => sys::ncclSum,
            Self::Product => sys::ncclProd,
            Self::Max => sys::ncclMax,
            Self::Min => sys::ncclMin,
            Self::Average => sys::ncclAvg,
        };
        ReductionOp {
            raw,
            _lifetime: PhantomData,
        }
    }
}

/// Reduction-operation handle whose lifetime prevents use after destruction.
#[derive(Clone, Copy)]
pub struct ReductionOp<'a> {
    raw: sys::ncclRedOp_t,
    _lifetime: PhantomData<&'a ()>,
}

impl ReductionOp<'_> {
    pub const fn as_raw(self) -> sys::ncclRedOp_t {
        self.raw
    }
}

/// Owned communicator configuration.
///
/// Any strings referenced by the raw C struct are stored inside this object, so
/// moving `Config` does not invalidate NCCL's input pointers.
pub struct Config {
    raw: sys::ncclConfig_t,
    net_name: Option<CString>,
    comm_name: Option<CString>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            raw: sys::ncclConfig_t::initialized(),
            net_name: None,
            comm_name: None,
        }
    }
}

impl Config {
    pub fn blocking(mut self, value: bool) -> Self {
        self.raw.blocking = if value { 1 } else { 0 };
        self
    }

    pub fn cga_cluster_size(mut self, value: i32) -> Self {
        self.raw.cgaClusterSize = value;
        self
    }

    pub fn min_ctas(mut self, value: i32) -> Self {
        self.raw.minCTAs = value;
        self
    }

    pub fn max_ctas(mut self, value: i32) -> Self {
        self.raw.maxCTAs = value;
        self
    }

    pub fn split_share(mut self, value: bool) -> Self {
        self.raw.splitShare = if value { 1 } else { 0 };
        self
    }

    pub fn traffic_class(mut self, value: i32) -> Self {
        self.raw.trafficClass = value;
        self
    }

    pub fn collnet_enable(mut self, value: bool) -> Self {
        self.raw.collnetEnable = if value { 1 } else { 0 };
        self
    }

    pub fn cta_policy(mut self, value: i32) -> Self {
        self.raw.CTAPolicy = value;
        self
    }

    pub fn shrink_share(mut self, value: bool) -> Self {
        self.raw.shrinkShare = if value { 1 } else { 0 };
        self
    }

    pub fn nvls_ctas(mut self, value: i32) -> Self {
        self.raw.nvlsCTAs = value;
        self
    }

    pub fn channels_per_net_peer(mut self, value: i32) -> Self {
        self.raw.nChannelsPerNetPeer = value;
        self
    }

    pub fn nvlink_centric_sched(mut self, value: bool) -> Self {
        self.raw.nvlinkCentricSched = if value { 1 } else { 0 };
        self
    }

    pub fn graph_usage_mode(mut self, value: i32) -> Self {
        self.raw.graphUsageMode = value;
        self
    }

    pub fn num_rma_contexts(mut self, value: i32) -> Self {
        self.raw.numRmaCtx = value;
        self
    }

    pub fn max_p2p_peers(mut self, value: i32) -> Self {
        self.raw.maxP2pPeers = value;
        self
    }

    pub fn graph_stream_ordering(mut self, value: bool) -> Self {
        self.raw.graphStreamOrdering = if value { 1 } else { 0 };
        self
    }

    pub fn launch_order_implicit(mut self, value: bool) -> Self {
        self.raw.launchOrderImplicit = if value { 1 } else { 0 };
        self
    }

    pub fn num_rma_signals(mut self, value: i32) -> Self {
        self.raw.numRmaSig = value;
        self
    }

    pub fn rma_eager_init(mut self, value: bool) -> Self {
        self.raw.rmaEagerInit = if value { 1 } else { 0 };
        self
    }

    pub fn host_cft_mode(mut self, value: sys::ncclHostCftMode_t) -> Self {
        self.raw.hostCftMode = value;
        self
    }

    pub fn net_name(mut self, value: impl AsRef<str>) -> Result<Self> {
        let value = CString::new(value.as_ref())?;
        self.raw.netName = value.as_ptr();
        self.net_name = Some(value);
        Ok(self)
    }

    pub fn comm_name(mut self, value: impl AsRef<str>) -> Result<Self> {
        let value = CString::new(value.as_ref())?;
        self.raw.commName = value.as_ptr();
        self.comm_name = Some(value);
        Ok(self)
    }

    fn raw_mut(&mut self) -> *mut sys::ncclConfig_t {
        &mut self.raw
    }
}

/// Per-collective NCCL configuration.
pub struct CollConfig {
    raw: sys::ncclCollConfig_t,
    algorithm: Option<CString>,
}

impl Default for CollConfig {
    fn default() -> Self {
        Self {
            raw: sys::ncclCollConfig_t::initialized(),
            algorithm: None,
        }
    }
}

impl CollConfig {
    pub fn min_ctas(mut self, value: i32) -> Self {
        self.raw.minCTAs = value;
        self
    }

    pub fn max_ctas(mut self, value: i32) -> Self {
        self.raw.maxCTAs = value;
        self
    }

    pub fn nvls_ctas(mut self, value: i32) -> Self {
        self.raw.nvlsCTAs = value;
        self
    }

    pub fn cga_cluster_size(mut self, value: i32) -> Self {
        self.raw.cgaClusterSize = value;
        self
    }

    pub fn force_algorithm(mut self, value: bool) -> Self {
        self.raw.forceAlgSelection = if value { 1 } else { 0 };
        self
    }

    pub fn cta_policy(mut self, value: i32) -> Self {
        self.raw.CTAPolicy = value;
        self
    }

    pub fn profiler_tag(mut self, value: u64) -> Self {
        self.raw.userProfilerTag = value;
        self
    }

    pub fn algorithm(mut self, value: impl AsRef<str>) -> Result<Self> {
        let value = CString::new(value.as_ref())?;
        self.raw.algSelection = value.as_ptr();
        self.algorithm = Some(value);
        Ok(self)
    }

    /// Attach a vendor extension linked list.
    ///
    /// # Safety
    ///
    /// `extension` and every node/string/pointer reachable from it must remain
    /// valid for the full duration of every collective call using this config,
    /// and the list must satisfy NCCL's uniqueness/non-circular requirements.
    pub unsafe fn extension(mut self, extension: *mut sys::ncclConfigExt_t) -> Self {
        self.raw.ext = extension;
        self
    }

    fn raw(&self) -> *const sys::ncclCollConfig_t {
        &self.raw
    }
}

/// Memory allocated through `ncclMemAlloc`.
pub struct NcclMemory<T> {
    ptr: NonNull<T>,
    len: usize,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl<T> NcclMemory<T> {
    pub fn allocate(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::InvalidArgument("NCCL allocation length must be non-zero"));
        }
        let bytes = std::mem::size_of::<T>()
            .checked_mul(len)
            .ok_or(Error::InvalidArgument("NCCL allocation size overflow"))?;
        if bytes == 0 {
            return Err(Error::InvalidArgument("zero-sized Rust types cannot be NCCL buffers"));
        }

        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid output pointer and `bytes` is non-zero.
        check(unsafe { sys::ncclMemAlloc(&mut raw, bytes) })?;
        let ptr = NonNull::new(raw.cast::<T>())
            .ok_or(Error::NullOnSuccess("ncclMemAlloc"))?;
        Ok(Self {
            ptr,
            len,
            _not_send_or_sync: PhantomData,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        false
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for NcclMemory<T> {
    fn drop(&mut self) {
        // SAFETY: this allocation came from ncclMemAlloc and is freed once.
        let _ = unsafe { sys::ncclMemFree(self.ptr.as_ptr().cast::<c_void>()) };
    }
}
