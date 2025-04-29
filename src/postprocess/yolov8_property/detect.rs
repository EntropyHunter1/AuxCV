use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::pyfunction;
use rayon::prelude::*;

use crate::nms::props::par_nms;

type Props = Vec<(usize, f32)>;
// Define the Object type as a tuple
type Object = ((f32, f32, f32, f32, u32, f32), Props);

#[pyfunction]
#[pyo3(name = "yolov8_property_postprocess")]
pub fn postprocess_batch(
    batch_pred: PyReadonlyArrayDyn<'_, f32>,
    iou_threshold: f32,
    conf_threshold: f32,
    num_cls: usize,
    has_property: Vec<bool>,
    property_groups: Vec<usize>,
) -> Vec<Vec<Object>> {
    let shape = batch_pred.shape();
    let batch_pred = batch_pred.as_slice().unwrap();
    let [bs, channels, grids] = [shape[0], shape[1], shape[2]];
    let batch_stride = channels * grids;
    let cls_end_idx = num_cls + 4;
    let mut results = Vec::with_capacity(bs);
    // Process each batch in parallel
    results.par_extend((0..bs).into_par_iter().map(|batch_idx| {
        let mut obj_map_cache: Vec<Vec<Object>> = vec![Vec::new(); num_cls];
        let pred = &batch_pred[batch_idx * batch_stride..(batch_idx + 1) * batch_stride];
        for i_grid in 0..grids {
            let mut max_pred_conf = 0.0;
            let mut max_pred_conf_idx = 0;
            for i in 4..cls_end_idx {
                let cls_conf = pred[i * grids + i_grid];
                if cls_conf > max_pred_conf {
                    max_pred_conf = cls_conf;
                    max_pred_conf_idx = i - 4;
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
            let mut props = vec![];
            if has_property[max_pred_conf_idx] {
                let mut base_start = cls_end_idx;
                for &group_num in property_groups.iter() {
                    let start = base_start;
                    let end = start + group_num;
                    let mut max_cls = usize::MAX;
                    let mut max_cls_prob = 0_f32;
                    for (cls, i) in (start..end).enumerate() {
                        let prob = pred[i * grids + i_grid];
                        if prob > max_cls_prob {
                            max_cls = cls;
                            max_cls_prob = prob;
                        }
                    }
                    base_start = end;
                    props.push((max_cls, max_cls_prob));
                }
            }
            let obj: Object = (
                (cx - half_w,
                 cy - half_h,
                 cx + half_w,
                 cy + half_h,
                 max_pred_conf_idx as u32, max_pred_conf,
                ),
                props);
            obj_map_cache[max_pred_conf_idx].push(obj);
        }
        // let mut res = Vec::new();
        // res.par_extend(obj_map_cache.into_par_iter().map(|mut boxes| -> Vec<Object>{
        //     if boxes.is_empty() {
        //         return vec![];
        //     }
        //     let mut filter_out_boxes = Vec::new();
        //     boxes.sort_by(|a, b| b.0.5.partial_cmp(&a.0.5).unwrap_or(std::cmp::Ordering::Equal));
        //     let mut idx: Vec<usize> = (0..boxes.len()).collect();
        //     while !idx.is_empty() {
        //         let good_idx = idx[0];
        //         filter_out_boxes.push(boxes[good_idx].clone());
        //         let mut tmp = idx.clone();
        //         idx.clear();
        //         let (good_xmin, good_ymin, good_xmax, good_ymax) = (boxes[good_idx].0.0, boxes[good_idx].0.1, boxes[good_idx].0.2, boxes[good_idx].0.3);
        //         let (good_width, good_height) = (good_xmax - good_xmin, good_ymax - good_ymin);
        //         for &tmp_i in tmp.iter().skip(1) {
        //             let (temp_xmin, temp_ymin, temp_xmax, temp_ymax) = (boxes[tmp_i].0.0, boxes[tmp_i].0.1, boxes[tmp_i].0.2, boxes[tmp_i].0.3);
        //             let (temp_width, temp_height) = (temp_xmax - temp_xmin, temp_ymax - temp_ymin);
        //             let (inter_x1, inter_y1, inter_x2, inter_y2) = (crate::max(good_xmin, temp_xmin), crate::max(good_ymin, temp_ymin), crate::min(good_xmax, temp_xmax), crate::min(good_ymax, temp_ymax));
        //             let (w, h) = (crate::max(inter_x2 - inter_x1, 0.0), crate::max(inter_y2 - inter_y1, 0.0));
        //             let inter_area = w * h;
        //             let area_1 = good_width * good_height;
        //             let area_2 = temp_width * temp_height;
        //             let o = inter_area / (area_1 + area_2 - inter_area + 0.00001);
        //             if o <= iou_threshold {
        //                 idx.push(tmp_i);
        //             }
        //         }
        //     }
        //     filter_out_boxes
        // }));
        // res.into_iter().flatten().collect()
        par_nms(iou_threshold, obj_map_cache)
    }));
    results
}
