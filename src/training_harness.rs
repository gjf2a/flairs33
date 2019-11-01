use crate::mnist_data::LabeledImage;
use crate::hash_histogram::HashHistogram;
use std::fmt;
use std::fmt::Formatter;
use std::collections::HashSet;

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

    pub fn record(&mut self, img_label: u8, classification: u8) {
        if classification == img_label {
            self.label_2_right.bump(img_label);
        } else {
            self.label_2_wrong.bump(img_label);
        }
    }

    pub fn all_labels(&self) -> HashSet<u8> {
        self.label_2_wrong.all_labels()
            .union(&self.label_2_right.all_labels())
            .map(|label| *label)
            .collect()
    }
}

impl fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for label in self.all_labels() {
            writeln!(f, "{}: {} correct, {} incorrect", label, self.label_2_right.get(label), self.label_2_wrong.get(label))?;
        }
        Ok(())
    }
}

pub trait Classifier {
    fn train(&mut self, training_images: &Vec<LabeledImage>);

    fn classify(&self, example: &LabeledImage) -> u8;

    fn test(&self, testing_images: &Vec<LabeledImage>) -> ConfusionMatrix {
        let mut result = ConfusionMatrix::new();
        for test_img in testing_images {
            result.record(test_img.get_label(), self.classify(test_img));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix() {
        let mut matrix = ConfusionMatrix::new();

        let one_ok = 6;
        let one_er = 4;
        let two_ok = 7;
        let two_er = 3;

        for _ in 0..one_ok {
            matrix.record(1, 1);
        }

        for _ in 0..one_er {
            matrix.record(1, 2);
        }

        for _ in 0..two_ok {
            matrix.record(2, 2);
        }

        for _ in 0..two_er {
            matrix.record(2, 1);
        }

        assert_eq!(format!("1: {} correct, {} incorrect\n2: {} correct, {} incorrect\n", one_ok, one_er, two_ok, two_er), matrix.to_string());
    }
}