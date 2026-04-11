//! C-ABI exports for calling CTE from non-Rust languages (e.g., TypeScript via Bun FFI).
//!
//! All functions use C-compatible types and return 0 on success, -1 on failure.
//! Opaque `*mut c_void` pointers represent `Box<Tag>` handles.
//!
//! All functions that call into CXX (which may panic on C++ exceptions) are wrapped
//! in `catch_unwind` to prevent UB at the `extern "C"` boundary.

use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::catch_unwind;
use std::ptr;
use std::slice;

use crate::sync::Tag;

/// Helper: convert a `*const c_char` to `&str`, returning `Err` on null or invalid UTF-8.
unsafe fn cstr_to_str<'a>(p: *const c_char) -> Result<&'a str, ()> {
    if p.is_null() {
        return Err(());
    }
    unsafe { CStr::from_ptr(p) }.to_str().map_err(|_| ())
}

/// Initialize CTE runtime. `config` may be null or empty for defaults.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn cte_c_init(config: *const c_char) -> i32 {
    let path = if config.is_null() {
        ""
    } else {
        match unsafe { cstr_to_str(config) } {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };
    let path = path.to_owned();
    match catch_unwind(move || crate::sync::init(&path)) {
        Ok(Ok(_)) => 0,
        _ => -1,
    }
}

/// Create or open a tag by name. Returns an opaque pointer (owned `Box<Tag>`).
/// Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn cte_c_tag_new(name: *const c_char) -> *mut c_void {
    let name = match unsafe { cstr_to_str(name) } {
        Ok(s) => s.to_owned(),
        Err(_) => return ptr::null_mut(),
    };
    match catch_unwind(move || Box::new(Tag::new(&name))) {
        Ok(tag) => Box::into_raw(tag) as *mut c_void,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            eprintln!("cte_c_tag_new panic: {}", msg);
            ptr::null_mut()
        }
    }
}

/// Free a tag handle previously returned by `cte_c_tag_new`.
#[no_mangle]
pub unsafe extern "C" fn cte_c_tag_free(tag: *mut c_void) {
    if !tag.is_null() {
        drop(unsafe { Box::from_raw(tag as *mut Tag) });
    }
}

/// Write data into a blob.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn cte_c_tag_put_blob(
    tag: *mut c_void,
    name: *const c_char,
    data: *const u8,
    len: u64,
    offset: u64,
    score: f32,
) -> i32 {
    if tag.is_null() || data.is_null() {
        return -1;
    }
    let tag_ref = unsafe { &*(tag as *const Tag) };
    let name = match unsafe { cstr_to_str(name) } {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let data = unsafe { slice::from_raw_parts(data, len as usize) };
    // Validate score
    if score < 0.0 || score > 1.0 {
        return -1;
    }
    // Tag is not UnwindSafe, so use AssertUnwindSafe
    let tag_ptr = std::panic::AssertUnwindSafe(tag_ref as *const Tag);
    let data_ptr = data.as_ptr();
    let data_len = data.len();
    let name = name.to_owned();
    match catch_unwind(move || {
        let tag = unsafe { &*tag_ptr.0 };
        let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };
        tag.put_blob_with_options(&name, data, offset, score)
    }) {
        Ok(Ok(())) => 0,
        Ok(Err(_)) => -1, // Validation error
        Err(_) => -1,
    }
}

/// Get the size of a blob in bytes.
/// Returns 0 if the tag or name is invalid or if validation fails.
#[no_mangle]
pub unsafe extern "C" fn cte_c_tag_get_blob_size(tag: *mut c_void, name: *const c_char) -> u64 {
    if tag.is_null() {
        return 0;
    }
    let tag_ref = unsafe { &*(tag as *const Tag) };
    let name = match unsafe { cstr_to_str(name) } {
        Ok(s) => s.to_owned(),
        Err(_) => return 0,
    };
    let tag_ptr = std::panic::AssertUnwindSafe(tag_ref as *const Tag);
    match catch_unwind(move || {
        let tag = unsafe { &*tag_ptr.0 };
        tag.get_blob_size(&name)
    }) {
        Ok(Ok(size)) => size,
        Ok(Err(_)) => 0, // Validation error
        Err(_) => 0,
    }
}

/// Read blob data into a caller-allocated buffer.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn cte_c_tag_get_blob(
    tag: *mut c_void,
    name: *const c_char,
    buf: *mut u8,
    size: u64,
    offset: u64,
) -> i32 {
    if tag.is_null() || buf.is_null() {
        return -1;
    }
    let tag_ref = unsafe { &*(tag as *const Tag) };
    let name = match unsafe { cstr_to_str(name) } {
        Ok(s) => s.to_owned(),
        Err(_) => return -1,
    };
    let tag_ptr = std::panic::AssertUnwindSafe(tag_ref as *const Tag);
    let buf_ptr = buf;
    match catch_unwind(move || {
        let tag = unsafe { &*tag_ptr.0 };
        match tag.get_blob(&name, size, offset) {
            Ok(data) => {
                let copy_len = std::cmp::min(data.len(), size as usize);
                unsafe { ptr::copy_nonoverlapping(data.as_ptr(), buf_ptr, copy_len) };
                0i32
            }
            Err(_) => -1i32, // Validation error
        }
    }) {
        Ok(rc) => rc,
        Err(_) => -1,
    }
}

/// List all blob names in a tag. Returns a JSON array string via `out_json`.
/// The caller must free the string with `cte_c_free_string`.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn cte_c_tag_get_contained_blobs(
    tag: *mut c_void,
    out_json: *mut *mut c_char,
) -> i32 {
    if tag.is_null() || out_json.is_null() {
        return -1;
    }
    let tag_ref = unsafe { &*(tag as *const Tag) };
    let tag_ptr = std::panic::AssertUnwindSafe(tag_ref as *const Tag);
    let out = out_json;
    match catch_unwind(move || {
        let tag = unsafe { &*tag_ptr.0 };
        let blobs = tag.get_contained_blobs();

        // Build JSON array manually to avoid serde dependency
        let json = format!(
            "[{}]",
            blobs
                .iter()
                .map(|s| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")))
                .collect::<Vec<_>>()
                .join(",")
        );

        match CString::new(json) {
            Ok(cs) => {
                unsafe { *out = cs.into_raw() };
                0i32
            }
            Err(_) => -1i32,
        }
    }) {
        Ok(rc) => rc,
        Err(_) => -1,
    }
}

/// Delete a tag by name.
/// Returns 0 on success, -1 on failure.
///
/// NOTE: This function requires a Client instance to operate. Create a client first
/// using cte_c_client_new() and use the client's del_tag method. This standalone
/// function returns -1 (not implemented) for backward compatibility.
#[no_mangle]
pub unsafe extern "C" fn cte_c_del_tag(name: *const c_char) -> i32 {
    // TODO: Implement properly by creating a Client instance and calling del_tag
    // For now, return -1 to indicate this function is not implemented
    // The sync::Client struct doesn't have a del_tag method - it needs to be added
    let _name = match unsafe { cstr_to_str(name) } {
        Ok(s) => s,
        Err(_) => return -1,
    };
    -1 // Not implemented - use cte_c_client_del_tag instead
}

/// Register a file-backed storage target.
/// Returns 0 on success, -1 on failure.
///
/// NOTE: This function requires a Client instance to operate. Create a client first
/// using cte_c_client_new() and use the client's register_target method. This standalone
/// function returns -1 (not implemented) for backward compatibility.
#[no_mangle]
pub unsafe extern "C" fn cte_c_register_target(path: *const c_char, size: u64) -> i32 {
    // TODO: Implement properly by creating a Client instance and calling register_target
    // For now, return -1 to indicate this function is not implemented
    // The sync::Client struct doesn't have a register_target method - it needs to be added
    let _path = match unsafe { cstr_to_str(path) } {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let _ = size;
    -1 // Not implemented - use cte_c_client_register_target instead
}

/// Free a string previously allocated by CTE (e.g., from `cte_c_tag_get_contained_blobs`).
#[no_mangle]
pub unsafe extern "C" fn cte_c_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}
