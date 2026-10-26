# DD-Net lite source

`classifiers.py` ports the architecture in [SHREC coarse 1D lite](https://github.com/fandulu/DD-Net/blob/master/SHREC/SHREC_coarse_1D_lite.ipynb), under the repository's MIT license copied to `licenses/DD-Net-LICENSE.txt`.

The Jester adaptation uses 21 image-space 2D joints, 210 pairwise distances, a pose-validity flag, and separate full-rate and half-rate motion streams with endpoint-validity flags. It preserves the reference's temporal resize, convolution blocks, pooling, 16-channel base width, dense layers, activations, and dropout. The classifier returns logits; softmax is applied only when exporting probabilities.
