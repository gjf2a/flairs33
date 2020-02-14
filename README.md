# Code from FLAIRS33 paper: A Preliminary Study of Spatial Bias in Knn Distance Metrics

A machine learning algorithm for image classification exhibits spatial bias if permuting the order of image pixels significantly alters its classification accuracy. In this paper, we explore the spatial bias of a number of different distance metrics for k-nearest-neighbor image classification. One distance metric is inspired by the convolutional kernels employed in convolutional neural networks. The other metrics are based on BRIEF descriptors, which generate bit vectors corresponding to images based on comparisons of pixel intensity values. We found that the convolutional distance metric exhibited a strong positive spatial bias, as did one of the BRIEF descriptors. Another BRIEF descriptor exhibited a negative spatial bias, and the remainder exhibited little or no spatial bias. These results lay a foundation for future work that would involve larger numbers of convolutional iterations, potentially synergized with BRIEF-style image preprocessing.

This repository contains the code I wrote to perform the experiments from this paper.

## Getting Started

Install the Rust programming language and compile the code using Cargo. Then run it on the command line with a single command-line argument: "help". It will then display the command-line arguments to run each of the variations given in the paper.

See the instructions at https://www.rust-lang.org/tools/install to set up Rust.


## Authors

* **Gabriel J. Ferrer, PhD**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
