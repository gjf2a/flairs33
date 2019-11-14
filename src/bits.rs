use std::ops::BitXor;
use decorum::R64;
use std::time::Instant;

const NUM_BITS: u64 = 64;

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
pub struct Bits {
    bits: Vec<u64>,
    size: u64
}

impl Bits {
    pub fn new() -> Bits {Bits {bits: Vec::new(), size: 0}}

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
        //self.bits.iter().map(|word| count_bits_on(*word)).sum()
        self.bits.iter().map(|word| num_bits(*word) as usize).sum()
    }
}

impl PartialEq for Bits {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && (0..self.bits.len()).all(|i| self.bits[i] == other.bits[i])
    }
}

impl Eq for Bits {}

fn get_mask(index: u64) -> u64 {
    1 << get_offset(index)
}

fn get_offset(index: u64) -> u64 {
    index % NUM_BITS
}

fn get_word(index: u64) -> usize {
    (index / NUM_BITS) as usize
}

fn count_bits_on(value: u64) -> usize {
    let mut count = 0;
    for i in 0..NUM_BITS {
        if value & get_mask(i) > 0 {
            count += 1;
        }
    }
    count
}

pub fn distance(b1: &Bits, b2: &Bits) -> usize {
    (b1 ^ b2).count_bits_on()
}

pub fn real_distance(b1: &Bits, b2: &Bits) -> R64 {
    R64::from_inner(distance(b1, b2) as f64)
}

impl BitXor for &Bits {
    type Output = Bits;

    fn bitxor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let mut result = Bits::new();
        for i in 0..self.bits.len() {
            result.bits.push(self.bits[i] ^ rhs.bits[i]);
        }
        result
    }
}

pub fn time<F: Fn()>(label: &str, f: F) {
    println!("Started {}...", label);
    let start = Instant::now();
    f();
    println!("Finished {} after {} milliseconds", label,
             Instant::now().duration_since(start).as_millis());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brief::bitvec_distance;

    #[test]
    fn test_bits() {
        let mut b = Bits::new();
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

        let mut b2 = Bits::new();
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

    #[test]
    fn test_time() {
        let num_items = 1000000;
        let baseline_1: Vec<bool> = (0..num_items).map(|_| rand::random()).collect();
        let baseline_2: Vec<bool> = (0..num_items).map(|_| rand::random()).collect();

        let mut bits_1 = Bits::new();
        baseline_1.iter().for_each(|b| bits_1.add(*b));
        let mut bits_2 = Bits::new();
        baseline_2.iter().for_each(|b| bits_2.add(*b));

        time("baseline distance", || println!("{}", bitvec_distance(&baseline_1, &baseline_2)));
        time("bits distance", || println!("{}", real_distance(&bits_1, &bits_2)));
    }
}