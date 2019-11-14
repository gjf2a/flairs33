use crate::mnist_data::{Image, ImageIterator};
use rand_distr::{Normal, Distribution};
use rand::prelude::ThreadRng;
use crate::bits::BitArray;
use rand::distributions::Uniform;

#[derive(Clone)]
pub struct Descriptor {
    pairs: Vec<((usize,usize),(usize,usize))>,
    width: usize,
    height: usize
}

fn constrained_random(dist: &Normal<f64>, rng: &mut ThreadRng, max: usize) -> usize {
    let mut value = dist.sample(rng);
    value = value.max(0 as f64);
    value = value.min((max - 1) as f64);
    value as usize
}

impl Descriptor {
    pub fn classic_brief(n: usize, width: usize, height: usize) -> Descriptor {
        let mut rng = rand::thread_rng();
        let x_dist = Normal::new((width/2) as f64, (width/6) as f64).unwrap();
        let y_dist = Normal::new((height/2) as f64, (height/6) as f64).unwrap();
        let mut result = Descriptor {pairs: Vec::new(), width: width, height: height};
        for _ in 0..n {
            result.pairs.push(((constrained_random(&x_dist, &mut rng, width),
                                constrained_random(&y_dist, &mut rng, height)),
                              (constrained_random(&x_dist, &mut rng, width),
                                constrained_random(&y_dist, &mut rng, height))));
        }
        result
    }

    pub fn uniform_neighbor(neighbors: usize, width: usize, height: usize) -> Descriptor {
        let mut rng = rand::thread_rng();
        let x_dist = Uniform::new(0, width);
        let y_dist = Uniform::new(0, height);
        let mut result = Descriptor {pairs: Vec::new(), width: width, height: height};
        ImageIterator::new(0, 0, width, height, 1)
            .for_each(|(x, y)|
                for _ in 0..neighbors {
                    result.pairs.push(((x, y), (x_dist.sample(&mut rng), y_dist.sample(&mut rng))));
                });
        result
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn apply_to(&self, img: &Image) -> BitArray {
        assert_eq!(img.side(), self.width());
        assert_eq!(img.side(), self.height());

        let mut bits = BitArray::new();
        self.pairs.iter()
            .for_each(|((x1, y1), (x2, y2))|
                bits.add(img.get(*x1, *y1) < img.get(*x2, *y2)));
        bits
    }
}