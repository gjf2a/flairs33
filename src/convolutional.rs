// Write a method that takes an Image and produces another Image with k classes for the pixel values
// Write a distance metric for that method.

use crate::mnist_data::Image;
use crate::kmeans::Kmeans;
use crate::kmeans;
use decorum::R64;

// Here's the approach:
// - Use k-means to find k convolution filters.
//   - Use both images being compared to generate the filters.
// - For each of the k filters:
//   - Generate an image based on each pixel's response to the filter.
//     - Response function:
//       - Find distance between pixel neighborhood and the filter.
//       - Somehow scale the distance to a u8, where zero distance yields 255
//   - Shrink the image by a factor of 2 ("pooling")
// - Repeat this process once
//   - Actually, let's try it with both one level and two levels.
// - Caluclate distance based on the filtered images at the final level.
//
//
// Numbers of images:
// - Imagine 8 filters
// - 28x28, 8 x 14x14, 64 x 7x7


