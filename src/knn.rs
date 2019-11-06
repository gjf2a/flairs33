extern crate decorum;
use crate::mnist_data::Image;
use crate::training_harness::Classifier;
use crate::hash_histogram::HashHistogram;
use self::decorum::R64;

pub struct Knn<I, F: Fn(&I,&I) -> R64> {
    k: usize,
    images: Vec<(u8,I)>,
    distance: F,
}

impl<I, F: Fn(&I,&I) -> R64> Knn<I,F> {
    pub fn new(k: usize, distance: F) -> Knn<I,F> {
        Knn {k: k, images: Vec::new(), distance: distance}
    }

    pub fn add_example(&mut self, img: (u8, I)) {
        self.images.push(img);
    }
}

impl<I: Clone, F: Fn(&I,&I) -> R64> Classifier<I> for Knn<I,F> {
    fn train(&mut self, training_images: &Vec<(u8,I)>) {
        for img in training_images {
            //self.add_example(img.clone()); // Flagged as type error by IDE, but compiles fine.
            self.add_example((img.0, img.1.clone()));
        }
    }

    fn classify(&self, example: &I) -> u8 {
        let mut distances: Vec<(R64, u8)> = self.images.iter()
            .map(|img| ((self.distance)(example, &img.1), img.0))
            .collect();
        distances.sort();

        let mut labels = HashHistogram::new();
        for i in 0..self.k {
            labels.bump(distances[i].1);
        }
        labels.mode()
    }
}