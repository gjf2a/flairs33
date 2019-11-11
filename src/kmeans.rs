use decorum::R64;
use rand::thread_rng;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use std::iter::Sum;

// These were a nice idea, but they don't compile.
//
//pub trait Data: Clone + Eq {}
//pub trait Dist<T>: Fn(&T,&T) -> R64 {}
//pub trait Mean<T>: Fn(&Vec<T>) -> T {}

pub struct Kmeans<T, D: Fn(&T,&T) -> R64, M: Fn(&Vec<T>) -> T> {
    k: usize,
    means: Vec<T>,
    distance: D,
    mean: M
}

impl <T: Clone + Eq, D: Fn(&T,&T) -> R64, M: Fn(&Vec<T>) -> T> Kmeans<T,D,M> {
    pub fn new(k: usize, data: &Vec<T>, distance: D, mean: M) -> Kmeans<T,D,M> {
        Kmeans {k: k, means: kmeans_iterate(k, data, &distance, &mean), distance: distance, mean: mean}
    }

    pub fn k(&self) -> usize {self.k}

    pub fn classification(&self, sample: &T) -> usize {
        classify(sample, &self.means, &self.distance)
    }

    pub fn copy_means(&self) -> Vec<T> {self.means.clone()}

    pub fn move_means(self) -> Vec<T> {self.means}
}

fn initial_plus_plus<T: Clone + Eq, D: Fn(&T,&T) -> R64>(k: usize, distance: &D, data: &Vec<T>) -> Vec<T> {
    let mut result = Vec::new();
    let mut rng = thread_rng();
    let range = Uniform::new(0, data.len());
    result.push(data[range.sample(&mut rng)].clone());
    while result.len() < k {
        let squared_distances: Vec<f64> = data.iter()
            .map(|d| distance(d, result.last().unwrap()).into_inner().powf(2.0))
            .collect();
        let dist = WeightedIndex::new(&squared_distances).unwrap();
        result.push(data[dist.sample(&mut rng)].clone());
    }
    result
}

fn kmeans_iterate<T: Clone + Eq, D: Fn(&T,&T) -> R64, M: Fn(&Vec<T>) -> T>(k: usize, data: &Vec<T>, distance: &D, mean: &M) -> Vec<T> {
    let mut result = initial_plus_plus(k, distance, data);
    loop {
        let mut classifications: Vec<Vec<T>> = (0..k).map(|_| Vec::new()).collect();
        for datum in data {
            let category = classify(datum, &result, distance);
            classifications[category].push(datum.clone());
        }
        let prev = result;
        result = classifications.iter().map(|v| mean(v)).collect();

        if (0..result.len()).all(|i| prev[i] == result[i]) {
            return result;
        }
    }
}

fn classify<T: Clone + Eq, D: Fn(&T,&T) -> R64>(target: &T, means: &Vec<T>, distance: &D) -> usize {
    let distances: Vec<(R64,usize)> = (0..means.len())
        .map(|i| (distance(&target, &means[i]), i))
        .collect();
    distances.iter().min().unwrap().1
}