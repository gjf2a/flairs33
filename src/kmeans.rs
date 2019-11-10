use decorum::R64;
use rand::thread_rng;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use std::iter::Sum;

//pub trait Data: Clone + Sum + Eq {} // Needed a type parameter for Sum, but there was no way to express that.
pub trait Dist<T>: Fn(&T,&T) -> R64 {}
pub trait ScalarDiv<T>: Fn(&T,usize) -> T {}

pub struct Kmeans<T, D: Dist<T>, M: ScalarDiv<T>> {
    k: usize,
    means: Vec<T>,
    distance: D,
    scalar_div: M
}

impl <T: Clone + Sum<T> + Eq, D: Dist<T>, M: ScalarDiv<T>> Kmeans<T,D,M> {
    pub fn new(k: usize, data: &Vec<T>, distance: D, scalar_div: M) -> Kmeans<T,D,M> {
        Kmeans {k: k, means: kmeans_iterate(k, data, &distance, &scalar_div), distance: distance, scalar_div: scalar_div}
    }

    pub fn k(&self) -> usize {self.k}

    pub fn classification(&self, sample: &T) -> usize {
        classify(sample, &self.means, &self.distance)
    }
}

fn initial_plus_plus<T: Clone + Sum<T> + Eq, D: Dist<T>>(k: usize, distance: &D, data: &Vec<T>) -> Vec<T> {
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

fn kmeans_iterate<T: Clone + Sum<T> + Eq, D: Dist<T>, M: ScalarDiv<T>>(k: usize, data: &Vec<T>, distance: &D, scalar_div: &M) -> Vec<T> {
    let mut result = initial_plus_plus(k, distance, data);
    loop {
        let mut classifications: Vec<Vec<T>> = (0..k).map(|_| Vec::new()).collect();
        for datum in data {
            let category = classify(datum, &result, distance);
            classifications[category].push(datum.clone());
        }
        let prev = result;

        let sums: Vec<(T,usize)> = classifications.iter().map(|v| (v.iter().map(|t| t.clone()).sum(), v.len())).collect();
        result = sums.iter().map(|(n,d)| scalar_div(n, *d)).collect();

        if (0..result.len()).all(|i| prev[i] == result[i]) {
            return result;
        }
    }
}

fn classify<T: Clone + Sum<T> + Eq, D: Dist<T>>(target: &T, means: &Vec<T>, distance: &D) -> usize {
    let distances: Vec<(R64,usize)> = (0..means.len())
        .map(|i| (distance(&target, &means[i]), i))
        .collect();
    distances.iter().min().unwrap().1
}