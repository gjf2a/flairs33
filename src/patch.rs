use crate::mnist_data::{Image, ImageIterator};

pub fn patchify(img: &Image, patch_size: usize) -> Vec<bool> {
    let mut patch = Vec::new();
    for (x, y) in img.x_y_iter() {
        for (i, j) in ImageIterator::centered(x as isize, y as isize, patch_size as isize, patch_size as isize, 1) {
            patch.push(img.get(x, y) > img.option_get(i, j).unwrap_or(0));
        }
    }
    patch
}