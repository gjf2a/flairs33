use crate::mnist_data::{LabeledImage, IMAGE_DIMENSION};
use decorum::R64;

pub fn euclidean_distance(img1: &LabeledImage, img2: &LabeledImage) -> R64 {
    let mut distance: R64 = R64::from_inner(0.0);

    for y in 0..IMAGE_DIMENSION {
        for x in 0..IMAGE_DIMENSION {
            let difference = img1.value(x, y) as f64 - img2.value(x, y) as f64;
            distance += difference.powf(2.0);
        }
    }

    distance
}