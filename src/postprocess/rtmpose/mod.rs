use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

fn argmax(data: &[f32]) -> (f32, usize) {
    let mut max_value = f32::MIN;
    let mut max_index = 0;

    for (i, &n) in data.iter().enumerate() {
        if n > max_value {
            max_value = n;
            max_index = i;
        }
    }

    (max_value, max_index)
}

#[pyfunction]
#[pyo3(name = "rtmpose_postprocess")]
pub fn postprocess(
    batch_pred_x: PyReadonlyArrayDyn<'_, f32>,
    batch_pred_y: PyReadonlyArrayDyn<'_, f32>,
    batch_scale_pad: Vec<(f32, u32, u32)>,
) -> Vec<Vec<(f32, f32, f32)>> {
    let x_shape = batch_pred_x.shape();
    let y_shape = batch_pred_y.shape();

    let bs = batch_scale_pad.len();
    let batch_pred_x = batch_pred_x.as_slice().unwrap();
    let batch_pred_y = batch_pred_y.as_slice().unwrap();

    let results: Vec<Vec<(f32, f32, f32)>> = (0..bs).into_par_iter().map(|i| {
        let pred_x = &batch_pred_x[i * x_shape[1] * x_shape[2]..(i + 1) * x_shape[1] * x_shape[2]];
        let pred_y = &batch_pred_y[i * y_shape[1] * y_shape[2]..(i + 1) * y_shape[1] * y_shape[2]];
        let (scale, pad_left, pad_top) = batch_scale_pad[i];

        let mut res_ = vec![(0.0, 0.0, 0.0); 17];
        for j in 0..17 {
            let (prob_x, pos_x) = argmax(&pred_x[j * x_shape[2]..(j + 1) * x_shape[2]]);
            let (prob_y, pos_y) = argmax(&pred_y[j * y_shape[2]..(j + 1) * y_shape[2]]);
            let prob = 0.5 * (prob_x + prob_y);
            let original_x = (pos_x as f32 * 0.5 - pad_left as f32) * scale;
            let original_y = (pos_y as f32 * 0.5 - pad_top as f32) * scale;

            res_[j] = (original_x, original_y, prob);
        }

        res_
    }).collect();

    results
}
