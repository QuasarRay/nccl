//! Raw ABI bindings for NCCL 2.31.x.
//!
//! These declarations mirror `src/nccl.h.in` without depending on CUDA headers.
//! CUDA's `cudaStream_t` is an opaque pointer in the runtime ABI, represented
//! here as `*mut c_void`. Set `NCCL_LIB_DIR` or `NCCL_HOME` when libnccl is not
//! on the platform's default linker search path.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::ffi::{c_char, c_int, c_uint, c_void};

pub const NCCL_MAJOR: c_int = 2;
pub const NCCL_MINOR: c_int = 31;
pub const NCCL_PATCH: c_int = 2;
pub const NCCL_VERSION_CODE: c_int = 23102;
pub const NCCL_SUFFIX: &str = "";

pub const fn NCCL_VERSION(major: c_int, minor: c_int, patch: c_int) -> c_int {
    if major <= 2 && minor <= 8 {
        major * 1000 + minor * 100 + patch
    } else {
        major * 10000 + minor * 100 + patch
    }
}

pub const NCCL_UNIQUE_ID_BYTES: usize = 128;
pub const NCCL_API_MAGIC: c_uint = 0xcafebeef;
pub const NCCL_CONFIG_UNDEF_INT: c_int = c_int::MIN;
pub const NCCL_SPLIT_NOCOLOR: c_int = -1;
pub const NCCL_UNDEF_FLOAT: f32 = -1.0;

pub const NCCL_WIN_DEFAULT: c_int = 0x00;
pub const NCCL_WIN_COLL_SYMMETRIC: c_int = 0x01;
pub const NCCL_WIN_STRICT_ORDERING: c_int = 0x02;
pub const NCCL_WIN_REQUIRED_ALIGNMENT: usize = 4096;

pub const NCCL_CTA_POLICY_DEFAULT: c_int = 0x00;
pub const NCCL_CTA_POLICY_EFFICIENCY: c_int = 0x01;
pub const NCCL_CTA_POLICY_ZERO: c_int = 0x02;

pub const NCCL_SHRINK_DEFAULT: c_int = 0x00;
pub const NCCL_SHRINK_ABORT: c_int = 0x01;
pub const NCCL_REVOKE_DEFAULT: c_int = 0x00;
pub const NCCL_SUSPEND_MEM: c_int = 0x01;

pub type cudaStream_t = *mut c_void;

#[repr(C)]
pub struct ncclComm {
    _private: [u8; 0],
}
pub type ncclComm_t = *mut ncclComm;
pub const NCCL_COMM_NULL: ncclComm_t = std::ptr::null_mut();

#[repr(C)]
pub struct ncclWindow_vidmem {
    _private: [u8; 0],
}
pub type ncclWindow_t = *mut ncclWindow_vidmem;

#[repr(C)]
pub struct ncclParamHandle {
    _private: [u8; 0],
}
pub type ncclParamHandle_t = *mut ncclParamHandle;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ncclUniqueId {
    pub internal: [c_char; NCCL_UNIQUE_ID_BYTES],
}

pub type ncclResult_t = c_int;
pub const ncclSuccess: ncclResult_t = 0;
pub const ncclUnhandledCudaError: ncclResult_t = 1;
pub const ncclSystemError: ncclResult_t = 2;
pub const ncclInternalError: ncclResult_t = 3;
pub const ncclInvalidArgument: ncclResult_t = 4;
pub const ncclInvalidUsage: ncclResult_t = 5;
pub const ncclRemoteError: ncclResult_t = 6;
pub const ncclInProgress: ncclResult_t = 7;
pub const ncclTimeout: ncclResult_t = 8;
pub const ncclNumResults: ncclResult_t = 9;

pub type ncclHostCftMode_t = c_int;
pub const ncclHostCftDefault: ncclHostCftMode_t = NCCL_CONFIG_UNDEF_INT;
pub const ncclHostCftEnable: ncclHostCftMode_t = 1;
pub const ncclHostCftDisable: ncclHostCftMode_t = 2;
pub const ncclHostCftFallback: ncclHostCftMode_t = 3;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ncclConfig_t {
    pub size: usize,
    pub magic: c_uint,
    pub version: c_uint,
    pub blocking: c_int,
    pub cgaClusterSize: c_int,
    pub minCTAs: c_int,
    pub maxCTAs: c_int,
    pub netName: *const c_char,
    pub splitShare: c_int,
    pub trafficClass: c_int,
    pub commName: *const c_char,
    pub collnetEnable: c_int,
    pub CTAPolicy: c_int,
    pub shrinkShare: c_int,
    pub nvlsCTAs: c_int,
    pub nChannelsPerNetPeer: c_int,
    pub nvlinkCentricSched: c_int,
    pub graphUsageMode: c_int,
    pub numRmaCtx: c_int,
    pub maxP2pPeers: c_int,
    pub graphStreamOrdering: c_int,
    pub launchOrderImplicit: c_int,
    pub numRmaSig: c_int,
    pub rmaEagerInit: c_int,
    pub hostCftMode: c_int,
}

pub type ncclConfig_v23100 = ncclConfig_t;

impl ncclConfig_t {
    pub const fn initialized() -> Self {
        Self {
            size: std::mem::size_of::<Self>(),
            magic: NCCL_API_MAGIC,
            version: NCCL_VERSION_CODE as c_uint,
            blocking: NCCL_CONFIG_UNDEF_INT,
            cgaClusterSize: NCCL_CONFIG_UNDEF_INT,
            minCTAs: NCCL_CONFIG_UNDEF_INT,
            maxCTAs: NCCL_CONFIG_UNDEF_INT,
            netName: std::ptr::null(),
            splitShare: NCCL_CONFIG_UNDEF_INT,
            trafficClass: NCCL_CONFIG_UNDEF_INT,
            commName: std::ptr::null(),
            collnetEnable: NCCL_CONFIG_UNDEF_INT,
            CTAPolicy: NCCL_CONFIG_UNDEF_INT,
            shrinkShare: NCCL_CONFIG_UNDEF_INT,
            nvlsCTAs: NCCL_CONFIG_UNDEF_INT,
            nChannelsPerNetPeer: NCCL_CONFIG_UNDEF_INT,
            nvlinkCentricSched: NCCL_CONFIG_UNDEF_INT,
            graphUsageMode: NCCL_CONFIG_UNDEF_INT,
            numRmaCtx: NCCL_CONFIG_UNDEF_INT,
            maxP2pPeers: NCCL_CONFIG_UNDEF_INT,
            graphStreamOrdering: NCCL_CONFIG_UNDEF_INT,
            launchOrderImplicit: NCCL_CONFIG_UNDEF_INT,
            numRmaSig: NCCL_CONFIG_UNDEF_INT,
            rmaEagerInit: NCCL_CONFIG_UNDEF_INT,
            hostCftMode: NCCL_CONFIG_UNDEF_INT,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ncclConfigExtKey_t {
    pub vendorId: c_int,
    pub optionId: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union ncclConfigExtVal_t {
    pub i: c_int,
    pub s: *const c_char,
    pub raw: *mut c_void,
}

#[repr(C)]
pub struct ncclConfigExt_t {
    pub next: *mut ncclConfigExt_t,
    pub key: ncclConfigExtKey_t,
    pub val: ncclConfigExtVal_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ncclCollConfig_t {
    pub size: usize,
    pub magic: c_uint,
    pub version: c_uint,
    pub ext: *mut ncclConfigExt_t,
    pub minCTAs: c_int,
    pub maxCTAs: c_int,
    pub nvlsCTAs: c_int,
    pub cgaClusterSize: c_int,
    pub algSelection: *const c_char,
    pub forceAlgSelection: c_int,
    pub CTAPolicy: c_int,
    pub userProfilerTag: u64,
}

pub type ncclCollConfig_v23100 = ncclCollConfig_t;

impl ncclCollConfig_t {
    pub const fn initialized() -> Self {
        Self {
            size: std::mem::size_of::<Self>(),
            magic: NCCL_API_MAGIC,
            version: NCCL_VERSION_CODE as c_uint,
            ext: std::ptr::null_mut(),
            minCTAs: NCCL_CONFIG_UNDEF_INT,
            maxCTAs: NCCL_CONFIG_UNDEF_INT,
            nvlsCTAs: NCCL_CONFIG_UNDEF_INT,
            cgaClusterSize: NCCL_CONFIG_UNDEF_INT,
            algSelection: std::ptr::null(),
            forceAlgSelection: 1,
            CTAPolicy: NCCL_CONFIG_UNDEF_INT,
            userProfilerTag: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ncclSimInfo_t {
    pub size: usize,
    pub magic: c_uint,
    pub version: c_uint,
    pub estimatedTime: f32,
}

pub type ncclSimInfo_v22200 = ncclSimInfo_t;

impl ncclSimInfo_t {
    pub const fn initialized() -> Self {
        Self {
            size: std::mem::size_of::<Self>(),
            magic: 0x74685283,
            version: NCCL_VERSION_CODE as c_uint,
            estimatedTime: NCCL_UNDEF_FLOAT,
        }
    }
}

pub type ncclCommMemStat_t = c_int;
pub const ncclStatGpuMemSuspend: ncclCommMemStat_t = 0;
pub const ncclStatGpuMemSuspended: ncclCommMemStat_t = 1;
pub const ncclStatGpuMemPersist: ncclCommMemStat_t = 2;
pub const ncclStatGpuMemTotal: ncclCommMemStat_t = 3;

pub type ncclRedOp_t = c_int;
pub const ncclSum: ncclRedOp_t = 0;
pub const ncclProd: ncclRedOp_t = 1;
pub const ncclMax: ncclRedOp_t = 2;
pub const ncclMin: ncclRedOp_t = 3;
pub const ncclAvg: ncclRedOp_t = 4;
pub const ncclNumOps: ncclRedOp_t = 5;
pub const ncclMaxRedOp: ncclRedOp_t = 0x7fffffff;

pub type ncclDataType_t = c_int;
pub const ncclInt8: ncclDataType_t = 0;
pub const ncclChar: ncclDataType_t = 0;
pub const ncclUint8: ncclDataType_t = 1;
pub const ncclInt32: ncclDataType_t = 2;
pub const ncclInt: ncclDataType_t = 2;
pub const ncclUint32: ncclDataType_t = 3;
pub const ncclInt64: ncclDataType_t = 4;
pub const ncclUint64: ncclDataType_t = 5;
pub const ncclFloat16: ncclDataType_t = 6;
pub const ncclHalf: ncclDataType_t = 6;
pub const ncclFloat32: ncclDataType_t = 7;
pub const ncclFloat: ncclDataType_t = 7;
pub const ncclFloat64: ncclDataType_t = 8;
pub const ncclDouble: ncclDataType_t = 8;
pub const ncclBfloat16: ncclDataType_t = 9;
pub const ncclFloat8e4m3: ncclDataType_t = 10;
pub const ncclFloat8e5m2: ncclDataType_t = 11;
pub const ncclNumTypes: ncclDataType_t = 12;

pub type ncclScalarResidence_t = c_int;
pub const ncclScalarDevice: ncclScalarResidence_t = 0;
pub const ncclScalarHostImmediate: ncclScalarResidence_t = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ncclWaitSignalDesc_t {
    pub opCnt: c_int,
    pub peer: c_int,
    pub sigIdx: c_int,
    pub ctx: c_int,
}

extern "C" {
    pub fn ncclMemAlloc(ptr: *mut *mut c_void, size: usize) -> ncclResult_t;
    pub fn pncclMemAlloc(ptr: *mut *mut c_void, size: usize) -> ncclResult_t;
    pub fn ncclMemFree(ptr: *mut c_void) -> ncclResult_t;
    pub fn pncclMemFree(ptr: *mut c_void) -> ncclResult_t;
    pub fn ncclGetVersion(version: *mut c_int) -> ncclResult_t;
    pub fn pncclGetVersion(version: *mut c_int) -> ncclResult_t;
    pub fn ncclGetUniqueId(unique_id: *mut ncclUniqueId) -> ncclResult_t;
    pub fn pncclGetUniqueId(unique_id: *mut ncclUniqueId) -> ncclResult_t;

    pub fn ncclCommInitRankConfig(comm: *mut ncclComm_t, nranks: c_int, comm_id: ncclUniqueId, rank: c_int, config: *mut ncclConfig_t) -> ncclResult_t;
    pub fn pncclCommInitRankConfig(comm: *mut ncclComm_t, nranks: c_int, comm_id: ncclUniqueId, rank: c_int, config: *mut ncclConfig_t) -> ncclResult_t;
    pub fn ncclCommInitRank(comm: *mut ncclComm_t, nranks: c_int, comm_id: ncclUniqueId, rank: c_int) -> ncclResult_t;
    pub fn pncclCommInitRank(comm: *mut ncclComm_t, nranks: c_int, comm_id: ncclUniqueId, rank: c_int) -> ncclResult_t;
    pub fn ncclCommInitAll(comm: *mut ncclComm_t, ndev: c_int, devlist: *const c_int) -> ncclResult_t;
    pub fn pncclCommInitAll(comm: *mut ncclComm_t, ndev: c_int, devlist: *const c_int) -> ncclResult_t;
    pub fn ncclCommFinalize(comm: ncclComm_t) -> ncclResult_t;
    pub fn pncclCommFinalize(comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;
    pub fn pncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t;
    pub fn pncclCommAbort(comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclCommRevoke(comm: ncclComm_t, revoke_flags: c_int) -> ncclResult_t;
    pub fn pncclCommRevoke(comm: ncclComm_t, revoke_flags: c_int) -> ncclResult_t;
    pub fn ncclCommSplit(comm: ncclComm_t, color: c_int, key: c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t) -> ncclResult_t;
    pub fn pncclCommSplit(comm: ncclComm_t, color: c_int, key: c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t) -> ncclResult_t;
    pub fn ncclCommShrink(comm: ncclComm_t, exclude_ranks: *mut c_int, exclude_count: c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t, shrink_flags: c_int) -> ncclResult_t;
    pub fn pncclCommShrink(comm: ncclComm_t, exclude_ranks: *mut c_int, exclude_count: c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t, shrink_flags: c_int) -> ncclResult_t;
    pub fn ncclCommGetUniqueId(comm: ncclComm_t, unique_id: *mut ncclUniqueId) -> ncclResult_t;
    pub fn pncclCommGetUniqueId(comm: ncclComm_t, unique_id: *mut ncclUniqueId) -> ncclResult_t;
    pub fn ncclCommGrow(comm: ncclComm_t, nranks: c_int, unique_id: *const ncclUniqueId, rank: c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t) -> ncclResult_t;
    pub fn pncclCommGrow(comm: ncclComm_t, nranks: c_int, unique_id: *const ncclUniqueId, rank: c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t) -> ncclResult_t;
    pub fn ncclCommInitRankScalable(newcomm: *mut ncclComm_t, nranks: c_int, myrank: c_int, n_id: c_int, comm_ids: *mut ncclUniqueId, config: *mut ncclConfig_t) -> ncclResult_t;
    pub fn pncclCommInitRankScalable(newcomm: *mut ncclComm_t, nranks: c_int, myrank: c_int, n_id: c_int, comm_ids: *mut ncclUniqueId, config: *mut ncclConfig_t) -> ncclResult_t;

    pub fn ncclGetErrorString(result: ncclResult_t) -> *const c_char;
    pub fn pncclGetErrorString(result: ncclResult_t) -> *const c_char;
    pub fn ncclGetLastError(comm: ncclComm_t) -> *const c_char;
    pub fn pncclGetLastError(comm: ncclComm_t) -> *const c_char;
    #[cfg(target_os = "linux")]
    pub fn ncclResetDebugInit();
    #[cfg(target_os = "linux")]
    pub fn pncclResetDebugInit();
    pub fn ncclCommGetAsyncError(comm: ncclComm_t, async_error: *mut ncclResult_t) -> ncclResult_t;
    pub fn pncclCommGetAsyncError(comm: ncclComm_t, async_error: *mut ncclResult_t) -> ncclResult_t;
    pub fn ncclCommCount(comm: ncclComm_t, count: *mut c_int) -> ncclResult_t;
    pub fn pncclCommCount(comm: ncclComm_t, count: *mut c_int) -> ncclResult_t;
    pub fn ncclCommCuDevice(comm: ncclComm_t, device: *mut c_int) -> ncclResult_t;
    pub fn pncclCommCuDevice(comm: ncclComm_t, device: *mut c_int) -> ncclResult_t;
    pub fn ncclCommUserRank(comm: ncclComm_t, rank: *mut c_int) -> ncclResult_t;
    pub fn pncclCommUserRank(comm: ncclComm_t, rank: *mut c_int) -> ncclResult_t;
    pub fn ncclCommRegister(comm: ncclComm_t, buff: *mut c_void, size: usize, handle: *mut *mut c_void) -> ncclResult_t;
    pub fn pncclCommRegister(comm: ncclComm_t, buff: *mut c_void, size: usize, handle: *mut *mut c_void) -> ncclResult_t;
    pub fn ncclCommDeregister(comm: ncclComm_t, handle: *mut c_void) -> ncclResult_t;
    pub fn pncclCommDeregister(comm: ncclComm_t, handle: *mut c_void) -> ncclResult_t;
    pub fn ncclCommSuspend(comm: ncclComm_t, flags: c_int) -> ncclResult_t;
    pub fn pncclCommSuspend(comm: ncclComm_t, flags: c_int) -> ncclResult_t;
    pub fn ncclCommResume(comm: ncclComm_t) -> ncclResult_t;
    pub fn pncclCommResume(comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclCommMemStats(comm: ncclComm_t, stat: ncclCommMemStat_t, value: *mut u64) -> ncclResult_t;
    pub fn pncclCommMemStats(comm: ncclComm_t, stat: ncclCommMemStat_t, value: *mut u64) -> ncclResult_t;
    pub fn ncclCommWindowRegister(comm: ncclComm_t, buff: *mut c_void, size: usize, win: *mut ncclWindow_t, win_flags: c_int) -> ncclResult_t;
    pub fn pncclCommWindowRegister(comm: ncclComm_t, buff: *mut c_void, size: usize, win: *mut ncclWindow_t, win_flags: c_int) -> ncclResult_t;
    pub fn ncclCommWindowDeregister(comm: ncclComm_t, win: ncclWindow_t) -> ncclResult_t;
    pub fn pncclCommWindowDeregister(comm: ncclComm_t, win: ncclWindow_t) -> ncclResult_t;
    pub fn ncclWinGetUserPtr(comm: ncclComm_t, win: ncclWindow_t, out_user_ptr: *mut *mut c_void) -> ncclResult_t;
    pub fn pncclWinGetUserPtr(comm: ncclComm_t, win: ncclWindow_t, out_user_ptr: *mut *mut c_void) -> ncclResult_t;

    pub fn ncclRedOpCreatePreMulSum(op: *mut ncclRedOp_t, scalar: *mut c_void, datatype: ncclDataType_t, residence: ncclScalarResidence_t, comm: ncclComm_t) -> ncclResult_t;
    pub fn pncclRedOpCreatePreMulSum(op: *mut ncclRedOp_t, scalar: *mut c_void, datatype: ncclDataType_t, residence: ncclScalarResidence_t, comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclRedOpDestroy(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t;
    pub fn pncclRedOpDestroy(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t;

    pub fn ncclReduce(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclReduce(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclBcast(buff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclBcast(buff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclBroadcast(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclBroadcast(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclAllReduce(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclAllReduce(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclReduceScatter(sendbuff: *const c_void, recvbuff: *mut c_void, recvcount: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclReduceScatter(sendbuff: *const c_void, recvbuff: *mut c_void, recvcount: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclAllGather(sendbuff: *const c_void, recvbuff: *mut c_void, sendcount: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclAllGather(sendbuff: *const c_void, recvbuff: *mut c_void, sendcount: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclAlltoAll(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclAlltoAll(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclGather(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclGather(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclScatter(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclScatter(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;

    pub fn ncclAllReduceConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn pncclAllReduceConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn ncclBroadcastConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn pncclBroadcastConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn ncclReduceConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn pncclReduceConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn ncclAllGatherConfig(sendbuff: *const c_void, recvbuff: *mut c_void, sendcount: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn pncclAllGatherConfig(sendbuff: *const c_void, recvbuff: *mut c_void, sendcount: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn ncclReduceScatterConfig(sendbuff: *const c_void, recvbuff: *mut c_void, recvcount: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn pncclReduceScatterConfig(sendbuff: *const c_void, recvbuff: *mut c_void, recvcount: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn ncclAlltoAllConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn pncclAlltoAllConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn ncclGatherConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn pncclGatherConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn ncclScatterConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;
    pub fn pncclScatterConfig(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, root: c_int, comm: ncclComm_t, stream: cudaStream_t, config: *const ncclCollConfig_t) -> ncclResult_t;

    pub fn ncclSend(sendbuff: *const c_void, count: usize, datatype: ncclDataType_t, peer: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclSend(sendbuff: *const c_void, count: usize, datatype: ncclDataType_t, peer: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclRecv(recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, peer: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclRecv(recvbuff: *mut c_void, count: usize, datatype: ncclDataType_t, peer: c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclPutSignal(localbuff: *const c_void, count: usize, datatype: ncclDataType_t, peer: c_int, peer_win: ncclWindow_t, peer_win_offset: usize, sig_idx: c_int, ctx: c_int, flags: c_uint, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclPutSignal(localbuff: *const c_void, count: usize, datatype: ncclDataType_t, peer: c_int, peer_win: ncclWindow_t, peer_win_offset: usize, sig_idx: c_int, ctx: c_int, flags: c_uint, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclSignal(peer: c_int, sig_idx: c_int, ctx: c_int, flags: c_uint, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclSignal(peer: c_int, sig_idx: c_int, ctx: c_int, flags: c_uint, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn ncclWaitSignal(n_desc: c_int, signal_descs: *mut ncclWaitSignalDesc_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
    pub fn pncclWaitSignal(n_desc: c_int, signal_descs: *mut ncclWaitSignalDesc_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;

    pub fn ncclGroupStart() -> ncclResult_t;
    pub fn pncclGroupStart() -> ncclResult_t;
    pub fn ncclGroupEnd() -> ncclResult_t;
    pub fn pncclGroupEnd() -> ncclResult_t;
    pub fn ncclGroupSimulateEnd(sim_info: *mut ncclSimInfo_t) -> ncclResult_t;
    pub fn pncclGroupSimulateEnd(sim_info: *mut ncclSimInfo_t) -> ncclResult_t;

    pub fn ncclParamBind(out: *mut ncclParamHandle_t, key: *const c_char) -> ncclResult_t;
    pub fn pncclParamBind(out: *mut ncclParamHandle_t, key: *const c_char) -> ncclResult_t;
    pub fn ncclParamGetI8(h: ncclParamHandle_t, out: *mut i8) -> ncclResult_t;
    pub fn pncclParamGetI8(h: ncclParamHandle_t, out: *mut i8) -> ncclResult_t;
    pub fn ncclParamGetI16(h: ncclParamHandle_t, out: *mut i16) -> ncclResult_t;
    pub fn pncclParamGetI16(h: ncclParamHandle_t, out: *mut i16) -> ncclResult_t;
    pub fn ncclParamGetI32(h: ncclParamHandle_t, out: *mut i32) -> ncclResult_t;
    pub fn pncclParamGetI32(h: ncclParamHandle_t, out: *mut i32) -> ncclResult_t;
    pub fn ncclParamGetI64(h: ncclParamHandle_t, out: *mut i64) -> ncclResult_t;
    pub fn pncclParamGetI64(h: ncclParamHandle_t, out: *mut i64) -> ncclResult_t;
    pub fn ncclParamGetU8(h: ncclParamHandle_t, out: *mut u8) -> ncclResult_t;
    pub fn pncclParamGetU8(h: ncclParamHandle_t, out: *mut u8) -> ncclResult_t;
    pub fn ncclParamGetU16(h: ncclParamHandle_t, out: *mut u16) -> ncclResult_t;
    pub fn pncclParamGetU16(h: ncclParamHandle_t, out: *mut u16) -> ncclResult_t;
    pub fn ncclParamGetU32(h: ncclParamHandle_t, out: *mut u32) -> ncclResult_t;
    pub fn pncclParamGetU32(h: ncclParamHandle_t, out: *mut u32) -> ncclResult_t;
    pub fn ncclParamGetU64(h: ncclParamHandle_t, out: *mut u64) -> ncclResult_t;
    pub fn pncclParamGetU64(h: ncclParamHandle_t, out: *mut u64) -> ncclResult_t;
    pub fn ncclParamGetStr(h: ncclParamHandle_t, out: *mut *const c_char) -> ncclResult_t;
    pub fn pncclParamGetStr(h: ncclParamHandle_t, out: *mut *const c_char) -> ncclResult_t;
    pub fn ncclParamGet(h: ncclParamHandle_t, out: *mut c_void, max_len: c_int, len: *mut c_int) -> ncclResult_t;
    pub fn pncclParamGet(h: ncclParamHandle_t, out: *mut c_void, max_len: c_int, len: *mut c_int) -> ncclResult_t;
    pub fn ncclParamGetParameter(key: *const c_char, value: *mut *const c_char, value_len: *mut c_int) -> ncclResult_t;
    pub fn pncclParamGetParameter(key: *const c_char, value: *mut *const c_char, value_len: *mut c_int) -> ncclResult_t;
    pub fn ncclParamGetAllParameterKeys(table: *mut *const *const c_char, table_len: *mut c_int) -> ncclResult_t;
    pub fn pncclParamGetAllParameterKeys(table: *mut *const *const c_char, table_len: *mut c_int) -> ncclResult_t;
    pub fn ncclParamDumpAll();
    pub fn pncclParamDumpAll();
}
