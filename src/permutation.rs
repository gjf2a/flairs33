use rand::thread_rng;
use rand::seq::SliceRandom;
use std::io;
use std::fs::File;
use std::io::{Write, Read};

pub fn make_permutation(n: usize) -> Vec<usize> {
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    indices
}

pub fn write_permutation(filename: &str, nums: &Vec<usize>) -> io::Result<()> {
    let mut file = File::create(filename)?;
    for elt in nums {
        file.write(elt.to_string().as_bytes())?;
        file.write(b",")?;
    }
    Ok(())
}

pub fn read_permutation(filename: &str) -> io::Result<Vec<usize>> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents.split(",")
        .map(|n| n.parse())
        .filter_map(Result::ok)
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::permutation::{make_permutation, write_permutation, read_permutation};
    use std::io;

    #[test]
    fn test_size() {
        let perm = make_permutation(10);
        assert_eq!(10, perm.len());
    }

    #[test]
    fn test_file() -> io::Result<()> {
        let perm = make_permutation(28*28);
        assert!(all_values_present(&perm));
        write_permutation("test_file", &perm)?;
        let read = read_permutation("test_file")?;
        assert_eq!(read, perm);
        Ok(())
    }

    fn all_values_present(nums: &Vec<usize>) -> bool {
        for i in 0..nums.len() {
            if !nums.contains(&i) {
                return false;
            }
        }
        true
    }
}