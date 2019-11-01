use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;

pub struct HashHistogram<K: Hash + Eq + Copy> {
    map: HashMap<K,usize>
}

impl<K: Hash + Eq + Copy> HashHistogram<K> {
    pub fn new() -> HashHistogram<K> {HashHistogram {map: HashMap::new()}}

    pub fn get(&self, key: K) -> usize {
        *(self.map.get(&key).unwrap_or(&0))
    }

    pub fn bump(&mut self, key: K) {
        let value: usize = self.map.get(&key).unwrap_or(&0) + 1;
        self.map.insert(key, value);
    }

    pub fn all_labels(&self) -> HashSet<K> {
        self.map.iter()
            .map(|entry| *entry.0)
            .collect()
    }

    pub fn mode(&self) -> K {
        *(self.map.iter()
            .max_by_key(|entry| entry.1)
            .unwrap().0)
    }
}