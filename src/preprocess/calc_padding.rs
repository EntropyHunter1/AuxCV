use pyo3::pyfunction;

#[pyfunction]
pub fn get_padding(src_w: i32, src_h: i32, width: i32, height: i32) -> (f32, i32, i32, i32, i32) {
    let w = src_w;
    let h = src_h;
    let scale_w = w as f32 / width as f32;
    let scale_h = h as f32 / height as f32;
    let (scale, new_w, new_h) = if scale_w > scale_h {
        (scale_w, width, (h as f32 / scale_w) as i32)
    } else {
        (scale_h, (w as f32 / scale_h) as i32, height)
    };
    let delta_width = width - new_w;
    let pad_left = (delta_width as f32 * 0.5) as i32;
    let pad_right = delta_width - pad_left;
    let delta_height = height - new_h;
    let pad_top = (delta_height as f32 * 0.5) as i32;
    let pad_bottom = delta_height - pad_top;
    (scale, pad_left, pad_top, pad_right, pad_bottom)
}

#[pyfunction]
pub fn get_auto_resize(src_w: i32, src_h: i32, width: i32, height: i32) -> (f32, i32, i32) {
    let w = src_w;
    let h = src_h;
    let scale_w = w as f32 / width as f32;
    let scale_h = h as f32 / height as f32;
    let (scale, new_w, new_h) = if scale_w > scale_h {
        (scale_w, width, (h as f32 / scale_w) as i32)
    } else {
        (scale_h, (w as f32 / scale_h) as i32, height)
    };
    (scale, new_w, new_h)
}