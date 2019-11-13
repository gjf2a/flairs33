use crate::mnist_data::Image;
use decorum::R64;
use crate::euclidean_distance::euclidean_distance;

const REDUCTION: usize = 2;

#[derive(Clone)]
pub struct Pyramid {
    images: Vec<Image>
}

impl Pyramid {
    pub fn new(src: &Image) -> Pyramid {
        let mut result = Pyramid {images: Vec::new()};
        let mut image = src.clone();
        while image.side() >= 2 {
            let shrunk = image.shrunken(REDUCTION);
            result.images.push(image);
            image = shrunk;
        }
        result
    }

    pub fn levels(&self) -> usize {
        self.images.len()
    }
}

pub fn pyramid_distance(img1: &Pyramid, img2: &Pyramid) -> R64 {
    assert_eq!(img1.levels(), img2.levels());

    let mut total: R64 = R64::from_inner(0.0);
    for i in 0..img1.levels() {
        // With the multiplier, it is significantly worse than the regular Euclidean.
        // Without the multiplier, the additional levels become so insignificant that
        // performance is identical.
        let multiplier: R64 = R64::from_inner(REDUCTION.pow(2*i as u32) as f64);
        total += multiplier * euclidean_distance(&img1.images[i], &img2.images[i]);
    }
    total
}