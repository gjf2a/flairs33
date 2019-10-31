extern crate decorum;
use crate::mnist_data::LabeledImage;
use decorum::R64;
use crate::training_harness::Classifier;
use crate::hash_histogram::HashHistogram;

pub struct Knn<F: Fn(&LabeledImage,&LabeledImage) -> R64> {
    k: usize,
    images: Vec<LabeledImage>,
    distance: F,
}

impl<F: Fn(&LabeledImage,&LabeledImage) -> R64> Knn<F> {
    pub fn new(k: usize, distance: F) -> Knn<F> {
        Knn {k: k, images: Vec::new(), distance: distance}
    }

    pub fn add_example(&mut self, img: LabeledImage) {
        self.images.push(img);
    }
}

impl<F: Fn(&LabeledImage,&LabeledImage) -> R64> Classifier for Knn<F> {
    fn train(&mut self, training_images: &Vec<LabeledImage>) {
        for img in training_images {
            self.add_example(*img);
        }
    }

    fn classify(&self, example: &LabeledImage) -> u8 {
        let mut distances: Vec<(R64, u8)> = self.images.iter()
            .map(|&img| ((self.distance)(example, &img), img.get_label()))
            .collect();
        distances.sort_unstable();

        let mut labels = HashHistogram::new();
        for i in 0..self.k {
            labels.bump(distances[i].1);
        }
        labels.mode()
    }
}