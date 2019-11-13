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

const NUM_KERNELS: usize = 8;
const KERNEL_SIZE: usize = 3;
const STRIDE: usize = 2;

pub fn convolutional_distance(img1: &Image, img2: &Image, levels: usize) -> R64 {
    assert!(levels > 0);
    let layers = make_convolutional_layers(img1, img2, NUM_KERNELS, STRIDE, levels);
    let final_layer_1 = layers.0.last().unwrap();
    let final_layer_2 = layers.1.last().unwrap();
    assert_eq!(final_layer_1.len(), final_layer_2.len());
    (0..final_layer_1.len())
        .map(|i| euclidean_distance(&final_layer_1[i], &final_layer_2[i]))
        .sum()
}

pub fn make_convolutional_layers(img1: &Image, img2: &Image, kernels: usize, stride: usize, levels: usize) -> (Vec<Vec<Image>>,Vec<Vec<Image>>) {
    let mut result = (vec![vec![img1.clone()]], vec![vec![img2.clone()]]);
    for _ in 0..levels {
        let frontier1 = result.0.last().unwrap();
        let frontier2 = result.1.last().unwrap();
        let mut projections1 = Vec::new();
        let mut projections2 = Vec::new();

        assert_eq!(frontier1.len(), frontier2.len());
        for i in 0..frontier1.len() {
            let kernels = find_filters_from(&frontier1[i], &frontier2[i], kernels, KERNEL_SIZE);
            projections1.append(&mut project_through(&frontier1[i], &kernels, stride));
            projections2.append(&mut project_through(&frontier2[i], &kernels, stride));
        }
        result.0.push(projections1);
        result.1.push(projections2);
    }
    result
}

pub fn project_through(src: &Image, kernels: &Vec<Image>, stride: usize) -> Vec<Image> {
    let mut distances: Vec<Vec<R64>> = (0..kernels.len()).map(|_| Vec::new()).collect();
    for (x, y) in src.x_y_step_iter(stride) {
        for k in 0..kernels.len() {
            distances[k].push(euclidean_distance(&kernels[k], &src.subimage(x, y, kernels[k].side())));
        }
    }

    let min = min_across(&distances).into_inner();
    let max = max_across(&distances).into_inner();

    let mut result: Vec<Image> = (0..kernels.len()).map(|_| Image::new()).collect();
    for k in 0..kernels.len() {
        for distance in distances[k].iter() {
            result[k].add(scale(distance.into_inner(), min, max))
        }
    }
    result
}

pub fn max_across(distances: &Vec<Vec<R64>>) -> R64 {
    *(distances.iter().filter_map(|v| v.iter().max()).max().unwrap())
}

pub fn min_across(distances: &Vec<Vec<R64>>) -> R64 {
    *(distances.iter().filter_map(|v| v.iter().min()).min().unwrap())
}

pub fn scale(value: f64, min: f64, max: f64) -> u8 {
    let float_scale = 1.0 - ((value - min) / (max - min));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale() {
        assert_eq!(0, scale(100.0, 50.0, 100.0));
        assert_eq!(255, scale(50.0, 50.0, 100.0));
        assert_eq!(155, scale(69.6078, 50.0, 100.0));
    }

    #[test]
    fn test_kernels() {
        let img = Image::from_vec(&(1..10).collect());
        let filters = find_filters_from(&img, &img, 4, 2);
        let filter_means: Vec<u8> = filters.iter().map(|f| f.pixel_mean()).collect();

        let target_means_1: Vec<u8> = vec![3, 0, 6, 1];
        let target_means_2: Vec<u8> = vec![3, 0, 6, 7];
        assert!(test_filter_means(&target_means_1, &filter_means) ||
                test_filter_means(&target_means_2, &filter_means));
    }

    fn test_filter_means(target_means: &Vec<u8>, filter_means: &Vec<u8>) -> bool {
        for mean in filter_means.iter() {
            if !target_means.contains(mean) && !target_means.contains(&(mean - 1)) && !target_means.contains(&(mean + 1)) {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_projection() {
        let img = Image::from_vec(&(1..10).collect());
        let kernels =
            vec![Image::from_vec(&(0..4).collect()), Image::from_vec(&(6..10).collect())];
        let projections = project_through(&img, &kernels, 1);
        let targets = vec![Image::from_vec(&vec![245, 252, 255, 250, 244, 224, 221, 157, 109]),
            Image::from_vec(&vec![0, 36, 67, 62, 157, 196, 120, 244, 253])];
        assert_eq!(projections.len(), targets.len());
        for i in 0..targets.len() {
            assert_eq!(targets[i], projections[i]);
        }
    }
}