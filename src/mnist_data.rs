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

    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.pixels[y * self.side_size + x]
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