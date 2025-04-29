use pyo3::prelude::{Bound, PyModule, pymodule, PyModuleMethods, PyResult, wrap_pyfunction};

mod postprocess;
mod preprocess;
mod nms;
mod cas;

#[inline(always)]
fn min(a: f32, b: f32) -> f32 {
    a.min(b)
}
#[inline(always)]
fn max(a: f32, b: f32) -> f32 {
    a.max(b)
}

#[pymodule]
fn computing_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preprocess::yolov5::detect::preprocess_fast, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess::yolov5::classify::preprocess_fast, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess::std::float32::preprocess_fast, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess::rtmpose::float32::preprocess_fast, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess::calc_padding::get_padding, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess::calc_padding::get_auto_resize, m)?)?;

    m.add_function(wrap_pyfunction!(postprocess::yolov5::detect::process, m)?)?;
    m.add_function(wrap_pyfunction!(postprocess::yolov8::detect::process, m)?)?;
    m.add_function(wrap_pyfunction!(postprocess::yolov8::segment::process, m)?)?;
    m.add_function(wrap_pyfunction!(postprocess::yolov8_property::detect::postprocess_batch, m)?)?;
    m.add_function(wrap_pyfunction!(postprocess::alphapose::pose::postprocess, m)?)?;
    m.add_function(wrap_pyfunction!(postprocess::rtmpose::postprocess, m)?)?;

    m.add_function(wrap_pyfunction!(cas::compare_exchange, m)?)?;
    m.add_function(wrap_pyfunction!(cas::fetch_add_seq_cst, m)?)?;
    m.add_function(wrap_pyfunction!(cas::fetch_sub_seq_cst, m)?)?;
    m.add_function(wrap_pyfunction!(cas::fetch_add_acq_rel, m)?)?;
    m.add_function(wrap_pyfunction!(cas::fetch_sub_acq_rel, m)?)?;
    m.add_function(wrap_pyfunction!(cas::compare_exchange_add, m)?)?;
    m.add_function(wrap_pyfunction!(cas::compare_exchange_sub, m)?)?;
    Ok(())
}