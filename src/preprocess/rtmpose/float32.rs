use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::pyfunction;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use zeroize::Zeroize;

#[pyfunction]
#[pyo3(name = "rtmpose_preprocess")]
pub fn preprocess_fast(
    src: PyReadonlyArrayDyn<'_, u8>,
    mut dst: PyReadwriteArrayDyn<'_, f32>,
    padding_rate: f32,
) -> (f32, u32, u32) {
    let src_height = src.shape()[0];
    let src_width = src.shape()[1];
    let dst_height = dst.shape()[1];
    let dst_width = dst.shape()[2];

    let pad_width = (src_width as f32 * padding_rate).round() as usize;
    let pad_height = (src_height as f32 * padding_rate).round() as usize;

    let padded_width = src_width + 2 * pad_width;
    let padded_height = src_height + 2 * pad_height;

    let scale_w = padded_width as f32 / dst_width as f32;
    let scale_h = padded_height as f32 / dst_height as f32;
    let scale = scale_w.max(scale_h);
    let new_width = (src_width as f32 / scale) as usize;
    let new_height = (src_height as f32 / scale) as usize;

    let delta_width = dst_width - new_width;
    let pad_left = delta_width / 2;
    let delta_height = dst_height - new_height;
    let pad_top = delta_height / 2;

    let dst_channel_pixels = dst_width * dst_height;
    let offset_channel0 = 0 * dst_channel_pixels;
    let offset_channel1 = 1 * dst_channel_pixels;
    let offset_channel2 = 2 * dst_channel_pixels;

    let src_slice = src.as_slice().unwrap();
    let dst_slice = dst.as_slice_mut().unwrap();

    let src_ptr = src_slice.as_ptr() as usize;
    let dst_ptr = dst_slice.as_mut_ptr() as usize;

    (0..new_height).into_par_iter().for_each(|y| {
        let src_ptr = src_ptr as *const u8;
        let dst_ptr = dst_ptr as *mut f32;
        for x in 0..new_width {
            let src_index = ((y as f32 * scale) as usize * src_width + (x as f32 * scale) as usize) * 3;
            let dst_index = (y + pad_top) * dst_width + x + pad_left;
            unsafe {
                *dst_ptr.add(dst_index + offset_channel2) = (*src_ptr.add(src_index + 0) as f32 - 103.530) / 57.375;
                *dst_ptr.add(dst_index + offset_channel1) = (*src_ptr.add(src_index + 1) as f32 - 123.675) / 58.395;
                *dst_ptr.add(dst_index + offset_channel0) = (*src_ptr.add(src_index + 2) as f32 - 116.280) / 57.120;
            }
        }
    });
    (scale, pad_left as u32, pad_top as u32)
}
