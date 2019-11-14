use std::ops::BitXor;
use decorum::R64;
use bitvec::prelude::*;

const NUM_BITS: u64 = 64;

const LOOKUP_8_BIT: [u8; 256] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8];

#[cfg(test)]
pub fn log2(n: u64) -> u8 {
    assert!(n >= 1);
    let mut log = 0;
    let mut n = n;
    while n > 1 {
        log += 1;
        n >>= 1;
    }
    log
}

#[cfg(test)]
pub fn num_bits(n: u64) -> u8 {
    let mut bits = 0;
    let mut leftover = n;
    while leftover > 0 {
        leftover -= 2_u64.pow(log2(leftover).into());
        bits += 1;
    }
    bits
}

#[derive(Clone, Debug)]
pub struct BitArray {
    bits: Vec<u64>,
    size: u64
}

impl BitArray {
    pub fn new() -> BitArray { BitArray {bits: Vec::new(), size: 0}}

    pub fn len(&self) -> u64 {self.size}

    pub fn add(&mut self, value: bool) {
        if get_offset(self.size) == 0 {
            self.bits.push(0);
        }
        self.set(self.size, value);
        self.size += 1;
    }

    pub fn set(&mut self, index: u64, value: bool) {
        let mask = get_mask(index);
        if value {
            self.bits[get_word(index)] |= mask;
        } else {
            self.bits[get_word(index)] &= !mask;
        }
    }

    pub fn is_set(&self, index: u64) -> bool {
        self.bits[get_word(index)] & get_mask(index) > 0
    }

    pub fn count_bits_on(&self) -> usize {
        self.bits.iter().map(|word| count_lookup(*word) as usize).sum()
    }
}

impl PartialEq for BitArray {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && (0..self.bits.len()).all(|i| self.bits[i] == other.bits[i])
    }
}

impl Eq for BitArray {}

fn get_mask(index: u64) -> u64 {
    1 << get_offset(index)
}

fn get_offset(index: u64) -> u64 {
    index % NUM_BITS
}

fn get_word(index: u64) -> usize {
    (index / NUM_BITS) as usize
}

fn count_lookup(value: u64) -> u8 {
    let mut count = 0;
    for i in 0..8 {
        let shift = i * 8;
        let mask: u64 = 0xFF << shift;
        let bits = (value & mask) >> shift;
        count += LOOKUP_8_BIT[bits as usize];
    }
    count
}

pub fn distance(b1: &BitArray, b2: &BitArray) -> usize {
    (b1 ^ b2).count_bits_on()
}

pub fn real_distance(b1: &BitArray, b2: &BitArray) -> R64 {
    R64::from_inner(distance(b1, b2) as f64)
}

impl BitXor for &BitArray {
    type Output = BitArray;

    fn bitxor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let mut result = BitArray::new();
        for i in 0..self.bits.len() {
            result.bits.push(self.bits[i] ^ rhs.bits[i]);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::print_time_milliseconds;

    #[test]
    fn test_bits() {
        let mut b = BitArray::new();
        assert_eq!(0, b.len());
        b.add(false);
        assert_eq!(1, b.len());
        assert!(!b.is_set(0));
        assert_eq!(0, b.count_bits_on());

        b.add(true);
        assert_eq!(2, b.len());
        assert!(b.is_set(1));
        assert_eq!(1, b.count_bits_on());

        for _ in 0..NUM_BITS {
            b.add(true);
        }
        assert_eq!(NUM_BITS + 2, b.len());
        for i in 1..b.len() {
            assert!(b.is_set(i));
        }
        assert_eq!(b.len() as usize - 1, b.count_bits_on());

        let mut b2 = BitArray::new();
        for i in 0..b.len() {
            b2.add(i % 2 == 0);
        }

        let b3 = &b ^ &b2;
        assert_eq!((b.len() as usize / 2) + 1, b3.count_bits_on());
        assert_eq!(b3.count_bits_on(), distance(&b, &b2));

        assert_ne!(b, b2);
        assert_ne!(b2, b3);
        assert_ne!(b, b3);

        assert_eq!(b, b.clone());
        assert_eq!(b2, b2.clone());
        assert_eq!(b3, b3.clone());
    }

    #[test]
    fn test_log2() {
        assert_eq!(1, log2(2));
        assert_eq!(2, log2(4));
        assert_eq!(3, log2(8));
        assert_eq!(16, log2(65536));
        assert_eq!(2, log2(7));
        assert_eq!(3, log2(15));
    }

    #[test]
    fn test_num_bits() {
        assert_eq!(1, num_bits(1));
        assert_eq!(1, num_bits(2));
        assert_eq!(2, num_bits(3));
        assert_eq!(1, num_bits(4));
        assert_eq!(2, num_bits(5));
        assert_eq!(2, num_bits(6));
        assert_eq!(3, num_bits(7));
        assert_eq!(1, num_bits(8));
    }

    pub fn bool_vec_distance(bv1: &Vec<bool>, bv2: &Vec<bool>) -> R64 {
        assert_eq!(bv1.len(), bv2.len());
        R64::from_inner((0..bv1.len())
            .filter(|i| bv1[*i] != bv2[*i])
            .count() as f64)
    }

    pub fn bitvec_distance_1(bv1: &BitVec<BigEndian,u8>, bv2: &BitVec<BigEndian,u8>) -> R64 {
        assert_eq!(bv1.len(), bv2.len());
        let xor = bv1.clone() ^ bv2.clone();
        R64::from_inner(xor.iter().filter(|b| *b).count() as f64)
    }

    pub fn bitvec_distance_2(bv1: &BitVec<BigEndian,u8>, bv2: &BitVec<BigEndian,u8>) -> R64 {
        assert_eq!(bv1.len(), bv2.len());
        R64::from_inner((0..bv1.len())
            .map(|i| bv1[i] != bv2[i])
            .filter(|b| *b)
            .count() as f64)
    }

    #[test]
    fn test_time() {
        let num_items = 1000000;
        let baseline_1: Vec<bool> = (0..num_items).map(|_| rand::random()).collect();
        let baseline_2: Vec<bool> = (0..num_items).map(|_| rand::random()).collect();

        let mut bits_1 = BitArray::new();
        baseline_1.iter().for_each(|b| bits_1.add(*b));
        let mut bits_2 = BitArray::new();
        baseline_2.iter().for_each(|b| bits_2.add(*b));

        let mut bitvec_1 = BitVec::new();
        baseline_1.iter().for_each(|b| bitvec_1.push(*b));
        let mut bitvec_2 = BitVec::new();
        baseline_2.iter().for_each(|b| bitvec_2.push(*b));

        let baseline_distance = print_time_milliseconds("baseline distance", || bool_vec_distance(&baseline_1, &baseline_2));
        let bits_distance = print_time_milliseconds("bits distance", || real_distance(&bits_1, &bits_2));
        let bitvec_distance_1 = print_time_milliseconds("bitvec distance 1", || bitvec_distance_1(&bitvec_1, &bitvec_2));
        let bitvec_distance_2 = print_time_milliseconds("bitvec distance 2", || bitvec_distance_2(&bitvec_1, &bitvec_2));
        assert_eq!(baseline_distance, bits_distance);
    }
}