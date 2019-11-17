use crate::bits::BitArray;
use crate::mnist_data::{Grid, Image, add_subgrid_of};
use crate::patch::patchify;
use crate::{kmeans, bits};
use decorum::R64;
use crate::hash_histogram::HashHistogram;
//use crate::brief::Descriptor;

const KERNEL_SIZE: usize = 3;
/*
pub fn kernelize_all(labeled_images: &Vec<(u8,Image)>, levels: usize, num_kernels: usize) -> Vec<(u8,BitImage)> {
    let mut candidates = Vec::new();
    let descriptor = Descriptor::uniform_neighbor(KERNEL_SIZE.pow(2), labeled_images[0].1.side(), labeled_images[0].1.side());
    for (label, img) in labeled_images.iter() {
        candidates.push(BitImage::from(&descriptor.apply_to(img)));
    }
    let kernels = kmeans::Kmeans::new(num_kernels, &candidates, real_distance, image_mean).move_means();

    for _ in 0..levels {

    }
}
*/
pub fn to_kernelized(img: &Image, levels: usize, num_kernels: usize) -> Vec<BitImage> {
    let mut kernelized = vec![binary_via_brief_patch(img, KERNEL_SIZE)];
    for _ in 0..levels {
        let mut iterated = Vec::new();
        for src in kernelized.iter() {
            for kernel in src.find_kernels(num_kernels) {
                iterated.push(src.project_through(&kernel, 2));
            }
        }
        kernelized = iterated;
    }
    kernelized
}

pub fn kernelized_distance(k1: &Vec<BitImage>, k2: &Vec<BitImage>) -> R64 {
    assert_eq!(k1.len(), k2.len());
    (0..k1.len())
        .map(|i| real_distance(&k1[i], &k2[i]))
        .sum()
}

#[derive(Clone)]
pub struct BitImage {
    pixels: BitArray,
    side: usize
}

impl Eq for BitImage {}

impl PartialEq for BitImage {
    fn eq(&self, other: &Self) -> bool {
        self.side == other.side && self.pixels == other.pixels
    }
}

impl Grid<bool> for BitImage {
    fn add(&mut self, pixel: bool) {
        self.pixels.add(pixel);
        if self.pixels.len() > self.side().pow(2) as u64 {
            self.side += 1;
        }
    }

    fn get(&self, x: usize, y: usize) -> bool {
        self.pixels.is_set((y * self.side() + x) as u64)
    }

    fn side(&self) -> usize {
        self.side
    }

    fn len(&self) -> usize {
        self.pixels.len() as usize
    }
}

impl BitImage {
    pub fn new() -> BitImage {
        BitImage { pixels: BitArray::new(), side: 0}
    }

    pub fn from(bits: &BitArray) -> BitImage {
        let mut result = BitImage::new();
        (0..bits.len()).for_each(|b| result.add(bits.is_set(b)));
        result
    }

    pub fn subimage(&self, x_center: usize, y_center: usize, side: usize) -> BitImage {
        let mut result = BitImage::new();
        add_subgrid_of(self, &mut result, false, x_center, y_center, side);
        result
    }

    /*
    pub fn add_kernels_to(&self, kernels: &mut Vec<BitImage>) {
        self.x_y_iter()
            .map(|(x, y)| self.subimage(x, y, KERNEL_SIZE))
            .for_each(|k| kernels.push(k));
    }
    */

    pub fn find_kernels(&self, num_kernels: usize) -> Vec<BitImage> {
        let candidates: Vec<BitImage> = self.x_y_iter()
            .map(|(x, y)| self.subimage(x, y, KERNEL_SIZE))
            .collect();
        kmeans::Kmeans::new(num_kernels, &candidates, real_distance, image_mean).move_means()
    }

    pub fn project_through(&self, kernel: &BitImage, stride: usize) -> BitImage {
        let mut result = BitImage::new();
        self.x_y_step_iter(stride).for_each(|(x, y)| {
            let sub = self.subimage(x, y, KERNEL_SIZE);
            result.add(distance(&sub, kernel) > kernel.len() / 2);
        });
        result
    }
}

pub fn real_distance(img1: &BitImage, img2: &BitImage) -> R64 {
    bits::real_distance(&img1.pixels, &img2.pixels)
}

pub fn distance(img1: &BitImage, img2: &BitImage) -> usize {
    bits::distance(&img1.pixels, &img2.pixels)
}

pub fn image_mean(images: &Vec<BitImage>) -> BitImage {
    let mut value_counts = HashHistogram::new();
    for img in images.iter() {
        for (x, y) in img.x_y_iter() {
            value_counts.bump((x, y, img.get(x, y)));
        }
    }
    let mut result = BitImage::new();
    for (x, y) in images[0].x_y_iter() {
        result.add(value_counts.get((x, y, true)) >= value_counts.get((x, y, false)));
    }
    result
}

pub fn binary_via_brief_patch(src: &Image, patch_size: usize) -> BitImage {
    BitImage::from(&patchify(src, patch_size))
}
