use crate::mnist_data::{Image, ImageIterator};
use crate::bits::Bits;

pub fn patchify(img: &Image, patch_size: usize) -> Bits {
    let mut patch = Bits::new();
    for (x, y) in img.x_y_iter() {
        for (i, j) in ImageIterator::centered(x as isize, y as isize, patch_size as isize, patch_size as isize, 1) {
            patch.add(img.get(x, y) > img.option_get(i, j).unwrap_or(0));
        }
    }
    patch
}