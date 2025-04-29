use numpy::PyReadonlyArrayDyn;
use pyo3::pyfunction;

#[pyfunction]
#[pyo3(name = "alphapose_postprocess")]
pub fn postprocess(pred: PyReadonlyArrayDyn<'_, f32>, scale: f32) -> Vec<(f32, f32, f32)> {
    let pred = pred.as_slice().unwrap();
    let mut base = 0;
    let mut kps = Vec::with_capacity(17);
    for _ in 0..17 {
        let x = pred[base] * scale;
        let y = pred[base + 1] * scale;
        let prob = pred[base + 2];
        base += 3;
        kps.push((x, y, prob));
    }
    kps
}