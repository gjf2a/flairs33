use crate::mnist_data::Image;
use decorum::R64;

pub fn euclidean_distance(img1: &Image, img2: &Image) -> R64 {
    assert_eq!(img1.side(), img2.side());
    assert_eq!(img1.len(), img2.len());
    R64::from_inner(img1.x_y_iter()
        .map(|(x, y)| (img1.get(x, y) as f64 - img2.get(x, y) as f64).powf(2.0))
        .sum())
}