/// Execute NCCL group semantics while guaranteeing `ncclGroupEnd` is attempted
/// on normal errors and panics.
pub fn group<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    // SAFETY: no pointers involved; NCCL tracks group state per calling thread.
    check(unsafe { sys::ncclGroupStart() })?;

    let body = catch_unwind(AssertUnwindSafe(f));
    // SAFETY: balances the successful GroupStart above.
    let end = check(unsafe { sys::ncclGroupEnd() });

    match body {
        Ok(Ok(value)) => {
            end?;
            Ok(value)
        }
        Ok(Err(error)) => {
            let _ = end;
            Err(error)
        }
        Err(payload) => {
            let _ = end;
            resume_unwind(payload)
        }
    }
}

/// Simulate the current group and return NCCL's estimated duration.
///
/// This terminates the active group just like `ncclGroupSimulateEnd`.
pub fn group_simulate_end() -> Result<f32> {
    let mut info = sys::ncclSimInfo_t::initialized();
    // SAFETY: info is initialized according to NCCL's required ABI.
    check(unsafe { sys::ncclGroupSimulateEnd(&mut info) })?;
    Ok(info.estimatedTime)
}

/// Bound NCCL runtime parameter. NCCL owns the underlying handle.
pub struct Parameter {
    raw: NonNull<sys::ncclParamHandle>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl Parameter {
    pub fn bind(key: impl AsRef<str>) -> Result<Self> {
        let key = CString::new(key.as_ref())?;
        let mut raw = ptr::null_mut();
        // SAFETY: key is NUL-terminated and output pointer is valid.
        check(unsafe { sys::ncclParamBind(&mut raw, key.as_ptr()) })?;
        let raw = NonNull::new(raw).ok_or(Error::NullOnSuccess("ncclParamBind"))?;
        Ok(Self {
            raw,
            _not_send_or_sync: PhantomData,
        })
    }

    pub fn get_i8(&self) -> Result<i8> {
        self.get_with(sys::ncclParamGetI8)
    }

    pub fn get_i16(&self) -> Result<i16> {
        self.get_with(sys::ncclParamGetI16)
    }

    pub fn get_i32(&self) -> Result<i32> {
        self.get_with(sys::ncclParamGetI32)
    }

    pub fn get_i64(&self) -> Result<i64> {
        self.get_with(sys::ncclParamGetI64)
    }

    pub fn get_u8(&self) -> Result<u8> {
        self.get_with(sys::ncclParamGetU8)
    }

    pub fn get_u16(&self) -> Result<u16> {
        self.get_with(sys::ncclParamGetU16)
    }

    pub fn get_u32(&self) -> Result<u32> {
        self.get_with(sys::ncclParamGetU32)
    }

    pub fn get_u64(&self) -> Result<u64> {
        self.get_with(sys::ncclParamGetU64)
    }

    fn get_with<T>(
        &self,
        f: unsafe extern "C" fn(sys::ncclParamHandle_t, *mut T) -> sys::ncclResult_t,
    ) -> Result<T>
    where
        T: Default,
    {
        let mut value = T::default();
        // SAFETY: output points to initialized storage of the requested type.
        check(unsafe { f(self.raw.as_ptr(), &mut value) })?;
        Ok(value)
    }

    pub fn get_string(&self) -> Result<String> {
        let mut value = ptr::null();
        // SAFETY: valid handle and output pointer.
        check(unsafe { sys::ncclParamGetStr(self.raw.as_ptr(), &mut value) })?;
        if value.is_null() {
            return Ok(String::new());
        }
        // Copy immediately because NCCL documents thread-local temporary lifetime.
        Ok(unsafe { CStr::from_ptr(value) }.to_string_lossy().into_owned())
    }

    pub fn get_bytes(&self, max_len: usize) -> Result<Vec<u8>> {
        let max_len_i32 = i32::try_from(max_len)
            .map_err(|_| Error::InvalidArgument("parameter buffer exceeds i32::MAX"))?;
        let mut bytes = vec![0_u8; max_len];
        let mut actual = 0;
        // SAFETY: vector has `max_len` writable bytes.
        check(unsafe {
            sys::ncclParamGet(
                self.raw.as_ptr(),
                bytes.as_mut_ptr().cast::<c_void>(),
                max_len_i32,
                &mut actual,
            )
        })?;
        if actual < 0 || actual as usize > bytes.len() {
            return Err(Error::InvalidArgument("NCCL returned an invalid parameter length"));
        }
        bytes.truncate(actual as usize);
        Ok(bytes)
    }
}

/// Get a runtime parameter by key, copying NCCL's temporary string result.
pub fn parameter(key: impl AsRef<str>) -> Result<Vec<u8>> {
    let key = CString::new(key.as_ref())?;
    let mut value = ptr::null();
    let mut len = 0;
    // SAFETY: key/output pointers are valid.
    check(unsafe { sys::ncclParamGetParameter(key.as_ptr(), &mut value, &mut len) })?;
    if len < 0 || (value.is_null() && len != 0) {
        return Err(Error::InvalidArgument("NCCL returned an invalid parameter value"));
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: NCCL reports `len` readable bytes for this temporary value.
    Ok(unsafe { slice::from_raw_parts(value.cast::<u8>(), len as usize) }.to_vec())
}

/// Return all currently published NCCL parameter keys as owned Rust strings.
pub fn parameter_keys() -> Result<Vec<String>> {
    let mut table: *const *const c_char = ptr::null();
    let mut len = 0;
    // SAFETY: output pointers are valid.
    check(unsafe { sys::ncclParamGetAllParameterKeys(&mut table, &mut len) })?;
    if len < 0 || (table.is_null() && len != 0) {
        return Err(Error::InvalidArgument("NCCL returned an invalid parameter table"));
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: NCCL reports an array of `len` C string pointers.
    let entries = unsafe { slice::from_raw_parts(table, len as usize) };
    let mut result = Vec::with_capacity(entries.len());
    for &entry in entries {
        if entry.is_null() {
            return Err(Error::InvalidArgument("NCCL parameter table contains a null key"));
        }
        result.push(unsafe { CStr::from_ptr(entry) }.to_string_lossy().into_owned());
    }
    Ok(result)
}

/// Ask NCCL to dump all runtime parameters to its log.
pub fn dump_parameters() {
    // SAFETY: no arguments or caller-managed memory.
    unsafe { sys::ncclParamDumpAll() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_config_initializer_matches_public_abi() {
        let raw = sys::ncclConfig_t::initialized();
        assert_eq!(raw.magic, sys::NCCL_API_MAGIC);
        assert_eq!(raw.version, sys::NCCL_VERSION_CODE as u32);
        assert_eq!(raw.size, std::mem::size_of::<sys::ncclConfig_t>());
    }

    #[test]
    fn unique_id_round_trip_is_lossless() {
        let mut bytes = [0_u8; sys::NCCL_UNIQUE_ID_BYTES];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = index as u8;
        }
        assert_eq!(UniqueId::from_bytes(bytes).to_bytes(), bytes);
    }

    #[test]
    fn native_datatype_mapping_is_stable() {
        assert_eq!(<f32 as NcclType>::DATA_TYPE, DataType::Float32);
        assert_eq!(<i64 as NcclType>::DATA_TYPE, DataType::Int64);
        assert_eq!(Reduction::Sum.op().as_raw(), sys::ncclSum);
    }
}
