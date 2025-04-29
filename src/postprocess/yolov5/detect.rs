use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::pyfunction;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelExtend;

use crate::nms::detect::par_nms;

type DetResult = (f32, f32, f32, f32, u32, f32, usize);
#[pyfunction]
#[pyo3(name = "yolov5_postprocess")]
pub fn process(batch_pred: PyReadonlyArrayDyn<'_, f32>,
               iou_threshold: f32,
               conf_threshold: f32,
) -> Vec<Vec<DetResult>> {
    let shape = batch_pred.shape();
    let batch_pred = batch_pred.as_slice().unwrap();
    // B Grid C
    let grids = shape[1];
    let channels = shape[2];
    let cls_num = channels - 5;
    let batch_size = shape[0];
    let batch_stride = channels * grids;

    let mut results = Vec::with_capacity(batch_size);
    results.par_extend((0..batch_size).into_par_iter().map(|batch_idx| {
        let mut obj_map_cache: Vec<Vec<DetResult>> = vec![vec![]; cls_num];
        let pred = &batch_pred[batch_idx * batch_stride..(batch_idx + 1) * batch_stride];
        for i_grid in 0..grids {
            let pos = i_grid * channels;
            let obj_conf: f32 = pred[4 + pos];
            if obj_conf < conf_threshold {
                continue;
            }
            let mut max_pred_conf = 0_f32;
            let mut max_pred_conf_idx = 0;
            let start = pos + 5;
            for i in 0..cls_num {
                let cls_conf: f32 = pred[start + i];
                let temp_score = obj_conf * cls_conf;
                if max_pred_conf < temp_score {
                    max_pred_conf_idx = i;
                    max_pred_conf = temp_score;
                }
            }
            if max_pred_conf < conf_threshold {
                continue;
            }
            let c_x: f32 = pred[0 + pos];
            let c_y: f32 = pred[1 + pos];
            let c_w: f32 = pred[2 + pos];
            let c_h: f32 = pred[3 + pos];
            let half_w = c_w * 0.5_f32;
            let half_h = c_h * 0.5_f32;
            let object = (
                c_x - half_w,
                c_y - half_h,
                c_x + half_w,
                c_y + half_h,
                max_pred_conf_idx as u32,
                max_pred_conf,
                i_grid,
            );
            obj_map_cache[max_pred_conf_idx].push(object);
        }
        par_nms(iou_threshold, obj_map_cache)
    }));
    results
}
