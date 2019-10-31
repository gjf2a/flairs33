use std::fs;
use std::io;
use std::io::Read;

pub const IMAGE_DIMENSION: usize = 28;
pub const IMAGE_BYTES: usize = IMAGE_DIMENSION * IMAGE_DIMENSION;

#[derive(Copy, Clone)]
pub struct LabeledImage {
    pixels: [[u8; IMAGE_DIMENSION]; IMAGE_DIMENSION],
    label: u8,
}

impl LabeledImage {
    pub fn new(pixels: &Vec<u8>, label: u8) -> LabeledImage {
        let mut result = LabeledImage { pixels: [[0; IMAGE_DIMENSION]; IMAGE_DIMENSION], label: label};
        let mut index = 0;
        for y in 0..IMAGE_DIMENSION {
            for x in 0..IMAGE_DIMENSION {
                result.pixels[y][x] = pixels[index];
                index += 1;
            }
        }
        result
    }

    pub fn get_label(&self) -> u8 {self.label}

    pub fn value(&self, x: usize, y: usize) -> u8 {
        self.pixels[y][x]
    }
}

pub fn init_from_files(image_file_name: &str, label_file_name: &str) -> io::Result<Vec<LabeledImage>> {
    let bytes: Vec<u8> = read_label_file(label_file_name)?;
    read_image_file(image_file_name, &bytes)
}

fn read_label_file(label_file_name: &str) -> io::Result<Vec<u8>> {
    let fin = fs::File::open(label_file_name)?;
    let mut bytes: Vec<u8> = Vec::new();
    let mut header_bytes_left = 8;
    for b in fin.bytes() {
        if header_bytes_left > 0 {
            header_bytes_left -= 1;
        } else {
            bytes.push(b?);
        }
    }
    Ok(bytes)
}

fn read_image_file(image_file_name: &str, labels: &Vec<u8>) -> io::Result<Vec<LabeledImage>> {
    let fin = fs::File::open(image_file_name)?;
    let mut images: Vec<LabeledImage> = Vec::new();
    let mut header_bytes_left = 16;
    let mut buffer: Vec<u8> = Vec::new();
    for b in fin.bytes() {
        if header_bytes_left > 0 {
            header_bytes_left -= 1;
        } else {
            buffer.push(b?);
            if buffer.len() == IMAGE_BYTES {
                images.push(LabeledImage::new(&buffer, labels[buffer.len()]));
                buffer.clear();
            }
        }
    }
    Ok(images)
}
