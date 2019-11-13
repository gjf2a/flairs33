use crate::mnist_data::{Image, ImageIterator};
use decorum::R64;
use crate::brief::bitvec_distance;

pub fn patch_distance(img1: &Image, img2: &Image, patch_size: usize) -> R64 {
    assert_eq!(img1.len(), img2.len());
    let mut distance: R64 = R64::from_inner(0.0);
    for (x, y) in img1.x_y_iter() {
        distance += bitvec_distance(&create_patch(&img1, x, y, patch_size),
                                    &create_patch(&img2, x, y, patch_size));
    }
    distance
}

pub fn create_patch(img: &Image, x: usize, y: usize, patch_size: usize) -> Vec<bool> {
    let mut patch = Vec::new();
    for (i, j) in ImageIterator::centered(x as isize, y as isize, patch_size as isize, patch_size as isize, 1) {
        patch.push(img.get(x, y) > img.option_get(i, j).unwrap_or(0));
    }
    patch
}