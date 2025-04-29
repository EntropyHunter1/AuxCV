use pyo3::buffer::PyBuffer;
use pyo3::types::PyAny;
use pyo3::{pyfunction, Bound, PyResult};
use std::sync::atomic::Ordering;

#[pyfunction]
#[pyo3(name = "fetch_add_seq_cst")]
pub unsafe fn fetch_add_seq_cst(data_ptr: &Bound<'_, PyAny>, val: u32) -> PyResult<u32> {
    let buffer = PyBuffer::<u8>::get(data_ptr)?;
    let a = std::sync::atomic::AtomicU32::from_ptr(buffer.buf_ptr() as *mut u32);
    Ok(a.fetch_add(val, Ordering::SeqCst))
}

#[pyfunction]
#[pyo3(name = "fetch_sub_seq_cst")]
pub unsafe fn fetch_sub_seq_cst(data_ptr: &Bound<'_, PyAny>, val: u32) -> PyResult<u32> {
    let buffer = PyBuffer::<u8>::get(data_ptr)?;
    let a = std::sync::atomic::AtomicU32::from_ptr(buffer.buf_ptr() as *mut u32);
    Ok(a.fetch_sub(val, Ordering::SeqCst))
}

#[pyfunction]
#[pyo3(name = "fetch_add_release")]
pub unsafe fn fetch_add_acq_rel(data_ptr: &Bound<'_, PyAny>, val: u32) -> PyResult<u32> {
    let buffer = PyBuffer::<u8>::get(data_ptr)?;
    let a = std::sync::atomic::AtomicU32::from_ptr(buffer.buf_ptr() as *mut u32);
    Ok(a.fetch_add(val, Ordering::Release))
}

#[pyfunction]
#[pyo3(name = "fetch_sub_release")]
pub unsafe fn fetch_sub_acq_rel(data_ptr: &Bound<'_, PyAny>, val: u32) -> PyResult<u32> {
    let buffer = PyBuffer::<u8>::get(data_ptr)?;
    let a = std::sync::atomic::AtomicU32::from_ptr(buffer.buf_ptr() as *mut u32);
    Ok(a.fetch_sub(val, Ordering::Release))
}

#[pyfunction]
#[pyo3(name = "compare_exchange")]
pub unsafe fn compare_exchange(data_ptr: &Bound<'_, PyAny>, cur: u32, new: u32) -> PyResult<bool> {
    let buffer = PyBuffer::<u8>::get(data_ptr)?;
    let a = std::sync::atomic::AtomicU32::from_ptr(buffer.buf_ptr() as *mut u32);
    Ok(a.compare_exchange(cur, new, Ordering::SeqCst, Ordering::SeqCst).is_ok())
}

#[pyfunction]
#[pyo3(name = "compare_exchange_add")]
pub unsafe fn compare_exchange_add(data_ptr: &Bound<'_, PyAny>, delta: u32) -> PyResult<(bool, u32)> {
    let buffer = PyBuffer::<u8>::get(data_ptr)?;
    let a = std::sync::atomic::AtomicU32::from_ptr(buffer.buf_ptr() as *mut u32);
    let cur = a.load(Ordering::SeqCst);
    let new = cur + delta;
    match a.compare_exchange(cur, new, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(old) => {
            Ok((true, old))
        }
        Err(old) => {
            Ok((false, old))
        }
    }
}

#[pyfunction]
#[pyo3(name = "compare_exchange_sub")]
pub unsafe fn compare_exchange_sub(data_ptr: &Bound<'_, PyAny>, delta: u32) -> PyResult<(bool, u32)> {
    let buffer = PyBuffer::<u8>::get(data_ptr)?;
    let a = std::sync::atomic::AtomicU32::from_ptr(buffer.buf_ptr() as *mut u32);
    let cur = a.load(Ordering::SeqCst);
    let new = cur - delta;
    match a.compare_exchange(cur, new, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(old) => {
            Ok((true, old))
        }
        Err(old) => {
            Ok((false, old))
        }
    }
}