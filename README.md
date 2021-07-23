# GPU-based 2D autocorrelation method using a sliding window

In order to process multi-exposure datasets a dense image autocorrelation approach can be used. This approach computes autocorrelation coefficients for a small patch around every pixel of the input image. Based on location of the peaks in the correlated images one can determine displacements of substructures.


## Examples

![alt text](https://github.com/axruff/cuda-flow2d/raw/master/images/correlation.png "Examples")

**Figure**: **Top Left**: Flat-field corrected input image depicting repetitive spray droplets. **Top Right**: Correlation peaks analysis. The peak corresponding to the best displacement match is shown as a red marker. **Bottom Left**: Correlation coefficients for each pixel. **Bottom Right**: Displacement amplitude computed from the positions of correlation peaks.  

