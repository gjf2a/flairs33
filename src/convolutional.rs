// Write a method that takes an Image and produces another Image with k classes for the pixel values
// Write a distance metric for that method.

use crate::mnist_data::{Image, ImageIterator};
use crate::kmeans::Kmeans;
use crate::kmeans;
use decorum::R64;

// Here's the approach:
// - Use k-means to find k convolution filters. (Imagine using 3x3.)
//   - Use both images being compared to generate the filters.
// - For each of the k filters:
//   - Generate an image based on each pixel's response to the filter.
//     - Response function:
//       - Find distance between pixel neighborhood and the filter.
//       - Somehow scale the distance to a u8, where zero distance yields 255
//   - Shrink the image by a factor of 2 ("pooling")
// - Repeat this process once
//   - Actually, let's try it with both one level and two levels.
// - Caluclate distance based on the filtered images at the final level.
//
//
// Numbers of images:
// - Imagine 8 filters
// - 28x28, 8 x 14x14, 64 x 7x7

pub fn find_filters_from(img1: &Image, img2: &Image, num_filters: usize, kernel_size: usize) -> Vec<Image> {
    let mut raw_filters: Vec<Image> = Vec::new();
    add_kernels_from_to(img1, &mut raw_filters, kernel_size);
    add_kernels_from_to(img2, &mut raw_filters, kernel_size);
    raw_filters
}

fn add_kernels_from_to(img: &Image, raw_filters: &mut Vec<Image>, kernel_size: usize) {
    let iter = img.x_y_iter();
    let offset = kernel_size / 2;
    iter.filter_map(|(x, y)| remix_bounds(kernel_bounds(x, kernel_size, img.side()), kernel_bounds(y, kernel_size, img.side())))
        .for_each(|((x1, y1), (x2, y2))| {
            raw_filters.push(img.subimage(x1, y1, x2, y2));
        });
}

pub fn kernel_bounds(value: usize, kernel_size: usize, max: usize) -> Option<(usize,usize)> {
    let offset = kernel_size / 2;
    if value >= offset {
        let lo = value - offset;
        let hi = lo + kernel_size - 1;
        if hi < max {
            return Some((lo, hi))
        }
    }
    None
}

pub fn remix_bounds(x: Option<(usize,usize)>, y: Option<(usize,usize)>) -> Option<((usize,usize),(usize,usize))> {
    if let Some((x1, x2)) = x {
        if let Some((y1, y2)) = y {
            return Some(((x1, y1),(x2, y2)))
        }
    }
    None
}