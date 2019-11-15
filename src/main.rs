mod mnist_data;
mod training_harness;
mod knn;
mod hash_histogram;
mod euclidean_distance;
mod pyramid;
mod permutation;
mod brief;
mod kmeans;
mod patch;
mod convolutional;
mod bits;
mod timing;

use std::io;
use crate::training_harness::Classifier;
use crate::hash_histogram::HashHistogram;
use crate::pyramid::Pyramid;
use crate::mnist_data::Image;
use decorum::R64;
use std::env;
use std::collections::{HashSet, BTreeMap, HashMap};
use crate::brief::Descriptor;
use crate::convolutional::convolutional_distance;
use crate::patch::patchify;
use crate::timing::print_time_milliseconds;

const BASE_PATH: &str = "/Users/ferrer/Desktop/mnist_data/";
const SHRINK_FACTOR: usize = 50;
const K: usize = 7;
const PATCH_SIZE: usize = 3;
const NUM_NEIGHBORS: usize = 16;

const HELP: &str = "help";
const PERMUTE: &str = "permute";
const BASELINE: &str = "baseline";
const PYRAMID: &str = "pyramid";
const BRIEF: &str = "brief";
const CONVOLUTIONAL_1: &str = "convolutional1";
const PATCH: &str = "patch";
const PATCH_7: &str = "patch_7";
const SHRINK: &str = "shrink";
const UNIFORM_NEIGHBORS: &str = "uniform_neighbors";
const GAUSSIAN_NEIGHBORS: &str = "gaussian_neighbors";
const GAUSSIAN_7: &str = "gaussian_7";

fn main() -> io::Result<()> {
    let args: HashSet<String> = env::args().collect();
    if args.contains(HELP) {
        help_message();
    } else {
        train_and_test(&args)?;
    }
    Ok(())
}

fn help_message() {
    println!("Usage: flairs33 [options]:");
    println!("\t{}: print this message", HELP);
    println!("\t{}: runs additional experiment that permutes image pixels", PERMUTE);
    println!("\t{}: Use only 1 out of {} training/testing images", SHRINK, SHRINK_FACTOR);
    println!("\nAlgorithmic options:");
    println!("\t{}: straightforward knn", BASELINE);
    println!("\t{}: knn with pyramid images", PYRAMID);
    println!("\t{}: knn with BRIEF descriptors", BRIEF);
    println!("\t{}: knn with uniform neighbor BRIEF", UNIFORM_NEIGHBORS);
    println!("\t{}: knn with gaussian neighbor BRIEF (stdev 1/3 side)", GAUSSIAN_NEIGHBORS);
    println!("\t{}: knn with gaussian neighbor BRIEF (stdev 1/7 side)", GAUSSIAN_7);
    println!("\t{}: knn with convolutional patch 3x3 BRIEF descriptors", PATCH);
    println!("\t{}: knn with convolutional patch 7x7 BRIEF descriptors", PATCH_7);
    println!("\t{}: knn with convolutional distance metric (1 level)", CONVOLUTIONAL_1);
}

fn train_and_test(args: &HashSet<String>) -> io::Result<()> {
    let mut training_images = load_data_set("train")?;
    let mut testing_images = load_data_set("t10k")?;

    if args.contains(SHRINK) {
        println!("Shrinking by {}", SHRINK_FACTOR);
        training_images = mnist_data::discard(&training_images, SHRINK_FACTOR);
        testing_images = mnist_data::discard(&testing_images, SHRINK_FACTOR);
    }

    see_label_counts(&training_images, "Training");
    see_label_counts(&testing_images, "Testing");

    let mut data = ExperimentData {
        training: training_images,
        testing: testing_images,
        descriptors: Default::default(),
        errors: BTreeMap::new()
    };

    data.add_descriptor(BRIEF, brief::Descriptor::classic_brief(8192, mnist_data::IMAGE_DIMENSION, mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(UNIFORM_NEIGHBORS, brief::Descriptor::uniform_neighbor(NUM_NEIGHBORS, mnist_data::IMAGE_DIMENSION, mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(GAUSSIAN_NEIGHBORS, brief::Descriptor::gaussian_neighbor(NUM_NEIGHBORS, mnist_data::IMAGE_DIMENSION / 3, mnist_data::IMAGE_DIMENSION, mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(GAUSSIAN_7, brief::Descriptor::gaussian_neighbor(NUM_NEIGHBORS, mnist_data::IMAGE_DIMENSION / 7, mnist_data::IMAGE_DIMENSION, mnist_data::IMAGE_DIMENSION));

    data.run_all_tests_with(&args);

    if args.contains(PERMUTE) {
        println!("Permuting images");
        let permutation = permutation::read_permutation("image_permutation_file")?;
        let mut permuted_data = data.permuted(&permutation);
        permuted_data.run_all_tests_with(&args);
        println!("Permuted results");
        permuted_data.print_errors();
    }

    println!("Original results");
    data.print_errors();

    Ok(())
}

fn load_data_set(file_prefix: &str) -> io::Result<Vec<(u8,Image)>> {
    let train_images = format!("{}{}-images-idx3-ubyte", BASE_PATH, file_prefix);
    let train_labels = format!("{}{}-labels-idx1-ubyte", BASE_PATH, file_prefix);

    let training_images = print_time_milliseconds(&format!("loading mnist {} images", file_prefix),
        || mnist_data::init_from_files(train_images.as_str(), train_labels.as_str()))?;

    println!("Number of {} images: {}", file_prefix, training_images.len());
    Ok(training_images)
}

fn permuted_data_set(permutation: &Vec<usize>, data: &Vec<(u8,Image)>) -> Vec<(u8,Image)> {
    data.iter()
        .map(|(label, img)| (*label, img.permuted(permutation)))
        .collect()
}

fn convert_all<I, C: Fn(&Image) -> I>(labeled_list: &Vec<(u8, Image)>, conversion: C) -> Vec<(u8, I)> {
    labeled_list.iter().map(|(label, img)| (*label, conversion(img))).collect()
}

fn see_label_counts<I>(labeled: &Vec<(u8, I)>, label: &str) {
    let mut label_counts = HashHistogram::new();
    for img in labeled {
        label_counts.bump(img.0);
    }
    println!("{} labels: {}", label, label_counts);
}

#[derive(Clone)]
pub struct ExperimentData {
    training: Vec<(u8,Image)>,
    testing: Vec<(u8,Image)>,
    descriptors: HashMap<String,Descriptor>,
    errors: BTreeMap<String,f64>
}

impl ExperimentData {
    pub fn build_and_test_model<I: Clone, C: Fn(&Image) -> I, D: Fn(&I,&I) -> R64>
    (&mut self, label: &str, conversion: C, distance: D) {
        let training_images = print_time_milliseconds(&format!("converting training images to {}", label),
                                                      || convert_all(&self.training, &conversion));

        let testing_images = print_time_milliseconds(&format!("converting testing images to {}", label),
                                                     || convert_all(&self.testing, &conversion));

        let mut model = knn::Knn::new(K, distance);
        print_time_milliseconds(&format!("training {} model (k={})", label, K),
                                || model.train(&training_images));
        let outcome = print_time_milliseconds("testing", || model.test(&testing_images));
        print!("{}", outcome);
        let error_percentage = outcome.error_rate() * 100.0;
        println!("Error rate: {}", error_percentage);
        self.errors.insert(label.to_string(), error_percentage);
    }

    pub fn get_descriptor(&self, name: &str) -> Descriptor {
        self.descriptors.get(name).unwrap().clone()
    }

    pub fn add_descriptor(&mut self, name: &str, d: Descriptor) {
        self.descriptors.insert(name.to_string(), d);
    }

    pub fn run_all_tests_with(&mut self, args: &HashSet<String>) {
        if args.contains(BASELINE) {
            self.build_and_test_model(BASELINE, |v| v.clone(), euclidean_distance::euclidean_distance);
        }
        if args.contains(PYRAMID) {
            self.build_and_test_model(PYRAMID, Pyramid::new, pyramid::pyramid_distance);
        }
        if args.contains(BRIEF) {
            let descriptor = self.get_descriptor(BRIEF);
            self.build_and_test_model(&BRIEF.to_uppercase(), |img| descriptor.apply_to(img), bits::real_distance);
        }
        if args.contains(UNIFORM_NEIGHBORS) {
            let descriptor = self.get_descriptor(UNIFORM_NEIGHBORS);
            self.build_and_test_model(UNIFORM_NEIGHBORS, |img| descriptor.apply_to(img), bits::real_distance);
        }
        if args.contains(GAUSSIAN_NEIGHBORS) {
            let descriptor = self.get_descriptor(GAUSSIAN_NEIGHBORS);
            self.build_and_test_model(GAUSSIAN_NEIGHBORS, |img| descriptor.apply_to(img), bits::real_distance);
        }
        if args.contains(PATCH) {
            self.build_and_test_model(PATCH, |img| patchify(img, PATCH_SIZE), bits::real_distance);
        }
        if args.contains(PATCH_7) {
            self.build_and_test_model(PATCH_7, |img| patchify(img, 7), bits::real_distance);
        }
        if args.contains(CONVOLUTIONAL_1) {
            self.build_and_test_model(CONVOLUTIONAL_1, |v| v.clone(), |img1, img2| convolutional_distance(img1, img2, 1));
        }
    }

    pub fn permuted(&self, permutation: &Vec<usize>) -> ExperimentData {
        ExperimentData {
            training: permuted_data_set(permutation, &self.training),
            testing: permuted_data_set(permutation, &self.testing),
            descriptors: self.descriptors.clone(),
            errors: BTreeMap::new()
        }
    }

    pub fn print_errors(&self) {
        for (k,v) in self.errors.iter() {
            println!("{}: {}%", k, v);
        }
    }
}