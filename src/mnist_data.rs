use std::fs;
use std::io;
use std::io::Read;

pub const IMAGE_DIMENSION: usize = 28;
pub const IMAGE_BYTES: usize = IMAGE_DIMENSION * IMAGE_DIMENSION;

#[derive(Clone)]
pub struct Image {
    pixels: Vec<u8>,
    side_size: usize,
}

impl Image {
    pub fn new() -> Image {
        Image {pixels: Vec::new(), side_size: 0}
    }

    pub fn add(&mut self, pixel: u8) {
        self.pixels.push(pixel);
        if self.pixels.len() > self.side_size.pow(2) {
            self.side_size += 1;
        }
    }

    pub fn in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.side() && y < self.side()
    }

    pub fn get(&self, x: usize, y: usize) -> u8 {
        assert!(self.in_bounds(x, y));
        self.pixels[y * self.side_size + x]
    }

    pub fn x_y_iter(&self) -> ImageIterator {
        ImageIterator::new(0, 0, self.side() - 1, self.side() - 1)
    }

    pub fn side(&self) -> usize {
        self.side_size
    }

    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    pub fn permuted(&self, permutation: &Vec<usize>) -> Image {
        assert_eq!(self.pixels.len(), permutation.len());
        let mut result = Image::new();
        for index in permutation {
            result.add(self.pixels[*index]);
        }
        result
    }

    pub fn shrunken(&self, shrink: usize) -> Image {
        let mut result = Image::new();
        let target_side = self.side() / shrink;
        for x in 0..target_side {
            for y in 0..target_side {
                result.add(self.subimage_mean(x, y, shrink));
            }
        }
        result
    }

    pub fn subimage(&self, x1: usize, y1: usize, x2: usize, y2: usize) -> Image {
        let mut result = Image::new();
        ImageIterator::new(x1, y1, x2, y2).for_each(|(x, y)| result.add(self.get(x, y)));
        result
    }

    fn subimage_mean(&self, x: usize, y: usize, side: usize) -> u8 {
        let mut sum: u16 = 0;
        for i in x..x + side {
            for j in y..y + side {
                sum += self.get(i, j) as u16;
            }
        }
        (sum / side.pow(2) as u16) as u8
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.side_size == other.side_size && self.pixels.len() == other.pixels.len() && (0..self.pixels.len()).all(|i| self.pixels[i] == other.pixels[i])
    }
}

impl Eq for Image {}

pub fn image_mean(images: &Vec<Image>) -> Image {
    assert!(images.len() > 0);
    assert!(images.iter().all(|img| img.pixels.len() == images[0].pixels.len()));
    let mut sums: Vec<usize> = (0..images[0].pixels.len()).map(|_| 0).collect();
    for image in images.iter() {
        for p in 0..image.pixels.len() {
            sums[p] += image.pixels[p] as usize;
        }
    }

    let mut result = Image::new();
    for sum in sums {
        result.add((sum / images.len()) as u8);
    }
    result
}

pub struct ImageIterator {
    width: usize,
    height: usize,
    x: usize,
    y: usize
}

impl ImageIterator {
    pub fn new(x1: usize, y1: usize, x2: usize, y2: usize) -> ImageIterator {
        ImageIterator {x: x1, y: y1, width: x2 + 1, height: y2 + 1}
    }
}

impl Iterator for ImageIterator {
    type Item = (usize,usize);

    fn next(&mut self) -> Option<Self::Item> {
        let result = (self.x, self.y);
        self.x += 1;
        if self.x == self.width {
            self.x = 0;
            self.y += 1;
            if self.y == self.height {
                return None
            }
        }
        Some(result)
    }
}

pub fn init_from_files(image_file_name: &str, label_file_name: &str) -> io::Result<Vec<(u8,Image)>> {
    let bytes: Vec<u8> = read_label_file(label_file_name)?;
    read_image_file(image_file_name, &bytes)
}

pub fn discard(items: &Vec<(u8,Image)>, shrink: usize) -> Vec<(u8,Image)> {
    let mut result: Vec<(u8,Image)> = Vec::new();
    for i in 0..items.len() {
        if i % shrink == 0 {
            result.push(items[i].clone())
        }
    }
    result
}

fn read_label_file(label_file_name: &str) -> io::Result<Vec<u8>> {
    let fin = fs::File::open(label_file_name)?;
    let mut bytes: Vec<u8> = Vec::new();
    let mut header_bytes_left = 8;
    for b in fin.bytes() {
        if header_bytes_left > 0 {
            header_bytes_left -= 1;
        } else {
            let b = b?;
            bytes.push(b);
        }
    }
    Ok(bytes)
}

fn read_image_file(image_file_name: &str, labels: &Vec<u8>) -> io::Result<Vec<(u8,Image)>> {
    let fin = fs::File::open(image_file_name)?;
    let mut images: Vec<(u8,Image)> = Vec::new();
    let mut image: Image = Image::new();
    let mut header_bytes_left = 16;
    for b in fin.bytes() {
        if header_bytes_left > 0 {
            header_bytes_left -= 1;
        } else {
            image.add(b?);
            if image.len() == IMAGE_BYTES {
                images.push((labels[images.len()], image));
                image = Image::new();
            }
        }
    }
    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_img() {
        let mut img = Image::new();
        assert_eq!(0, img.side());
        img.add(10);
        assert_eq!(1, img.side());
        assert_eq!(10, img.get(0, 0));
        img.add(20);
        assert_eq!(2, img.side());
        assert_eq!(20, img.get(1, 0));
        img.add(30);
        assert_eq!(2, img.side());
        assert_eq!(30, img.get(0, 1));
        img.add(40);
        assert_eq!(2, img.side());
        assert_eq!(40, img.get(1, 1));
        img.add(50);
        assert_eq!(3, img.side());
        assert_eq!(30, img.get(2, 0));
        assert_eq!(40, img.get(0, 1));
        assert_eq!(50, img.get(1, 1));
    }
}