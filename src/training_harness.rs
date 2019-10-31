use crate::mnist_data::LabeledImage;
use crate::hash_histogram::HashHistogram;

pub struct ConfusionMatrix {
    label_2_right: HashHistogram<u8>,
    label_2_wrong: HashHistogram<u8>,
}

impl ConfusionMatrix {
    pub fn new() -> ConfusionMatrix {
        ConfusionMatrix {
            label_2_right: HashHistogram::new(),
            label_2_wrong: HashHistogram::new(),
        }
    }

    pub fn record(&mut self, test_img: &LabeledImage, classification: u8) {
        if classification == test_img.get_label() {
            self.label_2_right.bump(classification);
        } else {
            self.label_2_wrong.bump(classification);
        }
    }
}

pub trait Classifier {
    fn train(&mut self, training_images: &Vec<LabeledImage>);

    fn classify(&self, example: &LabeledImage) -> u8;

    fn test(&self, testing_images: &Vec<LabeledImage>) -> ConfusionMatrix {
        let mut result = ConfusionMatrix::new();
        for test_img in testing_images {
            result.record(test_img, self.classify(test_img));
        }
        result
    }
}