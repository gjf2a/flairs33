use crate::mnist_data::Image;

const REDUCTION: usize = 2;

pub struct Pyramid {
    images: Vec<Image>
}

impl Pyramid {
    pub fn new(src: Image) -> Pyramid {
        let mut result = Pyramid {images: Vec::new()};
        let mut image = src.clone();
        while image.side() >= 2 {
            let shrunk = image.shrunken(REDUCTION);
            result.images.push(image);
            image = shrunk;
        }
        result
    }
}