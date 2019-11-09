mod mnist_data;
mod training_harness;
mod knn;
mod hash_histogram;
mod euclidean_distance;
mod pyramid;
mod permutation;
mod brief;
mod kmeans;

use std::io;
use crate::training_harness::Classifier;
use std::time::Instant;
use crate::hash_histogram::HashHistogram;
use crate::pyramid::images2pyramids;
use crate::mnist_data::Image;
use decorum::R64;
use std::env;
use std::collections::HashSet;
use crate::brief::Descriptor;

const BASE_PATH: &str = "/Users/ferrer/Desktop/mnist_data/";
const SHRINK: usize = 50;
const K: usize = 7;

macro_rules! timed_op {
    ($label:expr, $line:stmt) => {
        println!("Started {}...", $label);
        let start = Instant::now();
        $line
        println!("Finished {} after {} seconds", $label, Instant::now().duration_since(start).as_secs());
    }
}

fn main() -> io::Result<()> {
    let args: HashSet<String> = env::args().collect();
    if args.contains("help") {
        help_message();
    } else {
        train_and_test(&args)?;
    }
    Ok(())
}

fn help_message() {
    println!("Usage: flairs33 [options]:");
    println!("\thelp: print this message");
    println!("\tpermute: runs additional experiment that permutes image pixels");
    println!("\tbaseline: straightforward knn");
    println!("\tpyramid: knn with pyramid images");
    println!("\tbrief: knn with BRIEF descriptors");
    println!("\tshrink: Use only 1 out of {} training/testing images", SHRINK);
}

fn train_and_test(args: &HashSet<String>) -> io::Result<()> {
    let mut training_images = load_data_set("train")?;
    let mut testing_images = load_data_set("t10k")?;

    if args.contains("shrink") {
        println!("Shrinking by {}", SHRINK);
        training_images = mnist_data::discard(&training_images, SHRINK);
        testing_images = mnist_data::discard(&testing_images, SHRINK);
    }

    see_label_counts(&training_images, "Training");
    see_label_counts(&testing_images, "Testing");

    let descriptor = brief::Descriptor::new(8192, mnist_data::IMAGE_DIMENSION, mnist_data::IMAGE_DIMENSION);

    run_all_tests_with(&args, &training_images, &testing_images, &descriptor);

    if args.contains("permute") {
        println!("Permuting images");
        let permutation = permutation::read_permutation("image_permutation_file")?;
        run_all_tests_with(&args,
                           &permuted_data_set(&permutation, &training_images),
                           &permuted_data_set(&permutation, &testing_images),
                            &descriptor);
    }

    Ok(())
}

fn run_all_tests_with(args: &HashSet<String>, training_images: &Vec<(u8,Image)>, testing_images: &Vec<(u8,Image)>, descriptor: &Descriptor) {
    if args.contains("baseline") {
        build_and_test_model("Baseline", &training_images, &testing_images, |v| v.clone(), euclidean_distance::euclidean_distance);
    }
    if args.contains("pyramid") {
        build_and_test_model("Pyramid", &training_images, &testing_images, images2pyramids, pyramid::pyramid_distance);
    }
    if args.contains("brief") {
        build_and_test_model("BRIEF", &training_images, &testing_images, |images| descriptor.images_2_brief_vecs(images), brief::bitvec_distance);
    }
}

fn load_data_set(file_prefix: &str) -> io::Result<Vec<(u8,Image)>> {
    let train_images = format!("{}{}-images-idx3-ubyte", BASE_PATH, file_prefix);
    let train_labels = format!("{}{}-labels-idx1-ubyte", BASE_PATH, file_prefix);

    timed_op!(format!("loading mnist {} images", file_prefix),
        let training_images = mnist_data::init_from_files(train_images.as_str(), train_labels.as_str())?
    );
    println!("Number of {} images: {}", file_prefix, training_images.len());
    Ok(training_images)
}

fn permuted_data_set(permutation: &Vec<usize>, data: &Vec<(u8,Image)>) -> Vec<(u8,Image)> {
    data.iter()
        .map(|(label, img)| (*label, img.permuted(permutation)))
        .collect()
}

fn build_and_test_model<I: Clone, C: Fn(&Vec<(u8,Image)>) -> Vec<(u8,I)>, D: Fn(&I,&I) -> R64>
(label: &str, training: &Vec<(u8, Image)>, testing: &Vec<(u8,Image)>, conversion: C, distance: D) {
    timed_op!(format!("converting training images to {}", label),
        let training_images = conversion(training)
    );

    timed_op!(format!("converting testing images to {}", label),
        let testing_images = conversion(testing)
    );

    let mut model = knn::Knn::new(K, distance);
    timed_op!(format!("training {} model (k={})", label, K),
        model.train(&training_images)
    );
    timed_op!("testing",
        let outcome = model.test(&testing_images)
    );
    print!("{}", outcome);
    println!("Error rate: {}", outcome.error_rate() * 100.0);
}

fn see_label_counts<I>(labeled: &Vec<(u8, I)>, label: &str) {
    let mut label_counts = HashHistogram::new();
    for img in labeled {
        label_counts.bump(img.0);
    }
    println!("{} labels: {}", label, label_counts);
}