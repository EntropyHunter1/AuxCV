use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelExtend;

use crate::nms::detect::par_nms;

type DetResult = (f32, f32, f32, f32, u32, f32, usize);

#[pyfunction]
#[pyo3(name = "yolov8_seg_postprocess")]
pub fn process(batch_pred: PyReadonlyArrayDyn<'_, f32>,
               iou_threshold: f32,
               conf_threshold: f32,
               vec_dims: usize,
) -> Vec<Vec<DetResult>> {
    let shape = batch_pred.shape();
    let batch_pred = batch_pred.as_slice().unwrap();
    // B C Grid
    let channels = shape[1];
    let grids = shape[2];
    let batch_size = shape[0];
    let batch_stride = channels * grids;
    const CLS_START_IDX: usize = 4;
    let cls_num = channels - CLS_START_IDX - vec_dims;
    let cls_end_idx = channels - vec_dims;
    let mut results = Vec::with_capacity(batch_size);
    results.par_extend((0..batch_size).into_par_iter().map(|batch_idx| {
        let mut obj_map_cache: Vec<Vec<DetResult>> = vec![vec![]; cls_num];
        let pred = &batch_pred[batch_idx * batch_stride..(batch_idx + 1) * batch_stride];
        for i_grid in 0..grids {
            let mut max_pred_conf = 0_f32;
            let mut max_pred_conf_idx = 0;
            for i in CLS_START_IDX..cls_end_idx {
                let cls_conf: f32 = pred[i * grids + i_grid];
                if cls_conf > max_pred_conf {
                    max_pred_conf_idx = i - CLS_START_IDX;
                    max_pred_conf = cls_conf;
                }
            }
            if max_pred_conf < conf_threshold {
                continue;
            }

            let cx = pred[0 * grids + i_grid];
            let cy = pred[1 * grids + i_grid];
            let cw = pred[2 * grids + i_grid];
            let ch = pred[3 * grids + i_grid];
            let half_w = cw * 0.5;
            let half_h = ch * 0.5;
            let object = (
                cx - half_w,
                cy - half_h,
                cx + half_w,
                cy + half_h,
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


