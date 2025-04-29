use rayon::iter::{IntoParallelIterator, ParallelExtend, ParallelIterator};
type DetResult = (f32, f32, f32, f32, u32, f32, usize);
/// xmin ymin xmax ymax cls prob grid
/// 0    1    2    3    4   5    6
#[inline]
pub fn nms_cpu(boxes: &mut Vec<DetResult>, iou_threshold: f32, filter_out_boxes: &mut Vec<DetResult>) {
    if boxes.is_empty() {
        return;
    }
    boxes.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal));
    let mut idx: Vec<usize> = (0..boxes.len()).collect();
    while !idx.is_empty() {
        let good_idx = idx[0];
        filter_out_boxes.push(boxes[good_idx].clone());
        let tmp = idx.split_off(1);
        idx.clear();
        let t_good_box = &boxes[good_idx];
        let good_xmin = t_good_box.0;
        let good_ymin = t_good_box.1;
        let good_xmax = t_good_box.2;
        let good_ymax = t_good_box.3;
        let good_width = good_xmax - good_xmin;
        let good_height = good_ymax - good_ymin;

        for tmp_i in tmp {
            let tt_box = &boxes[tmp_i];
            let temp_xmin = tt_box.0;
            let temp_ymin = tt_box.1;
            let temp_xmax = tt_box.2;
            let temp_ymax = tt_box.3;
            let temp_width = temp_xmax - temp_xmin;
            let temp_height = temp_ymax - temp_ymin;

            let inter_x1 = good_xmin.max(temp_xmin);
            let inter_y1 = good_ymin.max(temp_ymin);
            let inter_x2 = good_xmax.min(temp_xmax);
            let inter_y2 = good_ymax.min(temp_ymax);

            let w = (inter_x2 - inter_x1).max(0.0);
            let h = (inter_y2 - inter_y1).max(0.0);

            let inter_area = w * h;
            let area_1 = good_width * good_height;
            let area_2 = temp_width * temp_height;
            let o = inter_area / (area_1 + area_2 - inter_area + 0.00001);
            if o <= iou_threshold {
                idx.push(tmp_i);
            }
        }
    }
}

#[inline]
pub fn par_nms(iou_threshold: f32, mut obj_map_cache: Vec<Vec<DetResult>>) -> Vec<DetResult> {
    let mut res = Vec::new();
    res.par_extend(obj_map_cache.into_par_iter().map(|mut boxes| -> Vec<DetResult>{
        let mut filter_out_boxes = Vec::new();
        nms_cpu(&mut boxes, iou_threshold, &mut filter_out_boxes);
        filter_out_boxes
    }));
    res.into_iter().flatten().collect()
}