use decorum::R64;
use rand::thread_rng;
use rand_distr::{Distribution, Uniform};
use rand::distributions::WeightedIndex;
use std::ops::Add;

pub struct Kmeans<T, D: Fn(&T,&T) -> R64, M: Fn(&T,usize) -> &T> {
    k: usize,
    means: Vec<T>,
    distance: D,
    scalar_div: M
}

impl <T: Clone + Add, D: Fn(&T,&T) -> R64, M: Fn(&T,usize) -> &T> Kmeans<T,D,M> {
    pub fn new(k: usize, data: &Vec<T>, distance: D, scalar_div: M) -> Kmeans<T,D,M> {
        Kmeans {k: k, means: kmeans_iterate(k, data, &distance, &scalar_div), distance: distance, scalar_div: scalar_div}
    }

    fn classify(target: &T, means: &Vec<T>, distance: &D) -> usize {
        let distances: Vec<(R64,usize)> = (0..means.len())
            .map(|i| (distance(&target, &means[i]), i))
            .collect();
        distances.iter().min().unwrap().1
    }
}

fn initial_plus_plus<T: Clone + Add, D: Fn(&T,&T) -> R64>(k: usize, distance: &D, data: &Vec<T>) -> Vec<T> {
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

fn kmeans_iterate<T: Clone + Add, D: Fn(&T,&T) -> R64, M: Fn(&T,usize) -> &T>(k: usize, data: &Vec<T>, distance: &D, scalar_div: &M) -> Vec<T> {
    let mut result = initial_plus_plus(k, distance, data);

    result
}
