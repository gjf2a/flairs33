// Write a method that takes an Image and produces another Image with k classes for the pixel values
// Write a distance metric for that method.

use crate::mnist_data::{Image, image_mean};
use crate::kmeans;
use decorum::R64;
use crate::euclidean_distance::euclidean_distance;

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

pub fn project_through(src: &Image, kernels: &Vec<Image>) -> Vec<Image> {
    let mut distances: Vec<Vec<R64>> = (0..kernels.len()).map(|_| Vec::new()).collect();
    for (x, y) in src.x_y_iter() {
        for k in 0..kernels.len() {
            distances[k].push(apply_to(&kernels[k], src, x, y));
        }
    }

    let min = min_across(&distances);
    let max = max_across(&distances);

    let mut result: Vec<Image> = (0..kernels.len()).map(|_| Image::new()).collect();
    for k in 0..kernels.len() {
        for distance in distances[k].iter() {
            result[k].add(scale(*distance, min, max))
        }
    }
    result
}

pub fn apply_to(kernel: &Image, target: &Image, x: usize, y: usize) -> R64 {
    let offset = kernel.side() / 2;
    let mut distance_sum: R64 = R64::from_inner(0.0);
    for (kx, ky) in kernel.x_y_iter() {
        distance_sum += euclidean_distance(&kernel, &target.subimage(x, y, kernel.side()))
    }
    distance_sum
}

pub fn max_across(distances: &Vec<Vec<R64>>) -> R64 {
    *(distances.iter().filter_map(|v| v.iter().max()).max().unwrap())
}

pub fn min_across(distances: &Vec<Vec<R64>>) -> R64 {
    *(distances.iter().filter_map(|v| v.iter().min()).min().unwrap())
}

pub fn scale(value: R64, min: R64, max: R64) -> u8 {
    let float_scale = 1.0 - ((value - min) / (max - min)).into_inner();
    (float_scale * std::u8::MAX as f64) as u8
}

pub fn find_filters_from(img1: &Image, img2: &Image, num_filters: usize, kernel_size: usize) -> Vec<Image> {
    let mut raw_filters: Vec<Image> = Vec::new();
    add_kernels_from_to(img1, &mut raw_filters, kernel_size);
    add_kernels_from_to(img2, &mut raw_filters, kernel_size);
    kmeans::Kmeans::new(num_filters, &raw_filters, euclidean_distance, image_mean).move_means()
}

fn add_kernels_from_to(img: &Image, raw_filters: &mut Vec<Image>, kernel_size: usize) {
    img.x_y_iter().
        for_each(|(x, y)| raw_filters.push(img.subimage(x, y, kernel_size)));
}