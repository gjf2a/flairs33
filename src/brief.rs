use crate::mnist_data::Image;
use decorum::R64;
use bitvec::prelude::*;
use rand_distr::{Normal, Distribution};

pub struct Descriptor {
    pairs: Vec<((usize,usize),(usize,usize))>,
    width: usize,
    height: usize
}

impl Descriptor {
    pub fn new(n: usize, width: usize, height: usize) -> Descriptor {
        let mut rng = rand::thread_rng();
        let x_dist = Normal::new((width/2) as f64, (width/2) as f64).unwrap();
        let y_dist = Normal::new((height/2) as f64, (height/2) as f64).unwrap();
        let mut result = Descriptor {pairs: Vec::new(), width: width, height: height};
        for _ in 0..n {
            result.pairs.push(((x_dist.sample(&mut rng) as usize, y_dist.sample(&mut rng) as usize),
                              (x_dist.sample(&mut rng) as usize, y_dist.sample(&mut rng) as usize)));
        }
        result
    }

    pub fn apply_to(&self, img: &Image) -> BitVec<BigEndian,u8> {
        assert_eq!(img.side(), self.width());
        assert_eq!(img.side(), self.height());

        self.pairs.iter()
            .map(|((x1, y1), (x2, y2))| img.get(*x1, *y1) < img.get(*x2, *y2))
            .collect()
    }

    pub fn images_2_brief_vecs(&self, images: &Vec<(u8,Image)>) -> Vec<(u8,BitVec<BigEndian,u8>)> {
        images.iter()
            .map(|(label, img)| (*label, self.apply_to(img)))
            .collect()
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

pub fn bitvec_distance(bv1: &BitVec<BigEndian,u8>, bv2: &BitVec<BigEndian,u8>) -> R64 {
    let xor = bv1.clone() ^ bv2.clone();
    R64::from_inner(xor.iter().filter(|b| *b).count() as f64)
}