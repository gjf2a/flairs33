use crate::mnist_data::Image;
use decorum::R64;

pub fn euclidean_distance(img1: &Image, img2: &Image) -> R64 {
    assert_eq!(img1.side(), img2.side());
    let mut distance: R64 = R64::from_inner(0.0);

    for y in 0..img1.side() {
        for x in 0..img1.side() {
            let difference = img1.get(x, y) as f64 - img2.get(x, y) as f64;
            distance += difference.powf(2.0);
        }
    }

    distance
}