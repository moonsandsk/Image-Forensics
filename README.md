# Digital Image Forensics Dashboard

## Project Description:
### Summary -
This project is a Digital Image Tampering Detector built in MATLAB for the Digital Image Processing (DIP) course. It addresses the need for robust image authentication by implementing a multi-method forensic dashboard. The system analyzes a suspect image using five distinct passive forensic techniques (ELA, JPEG Artifacts, Noise Analysis, Edge Analysis, and Copy-Move) and generates a unified visual report. The final output is a **Fused Suspicion Map** that highlights the most probable regions of manipulation.

### Course concepts used -
1.  **Image Fundamentals:** Handling image data types (uint8, double), color space conversion (RGB to YCbCr/Grayscale), and reading/writing image files.
2.  **Filtering & Frequency Domain:** Implementation of Wiener filtering (`wiener2`) for noise extraction and Block-based processing which relies on Discrete Cosine Transform (DCT) principles.
3.  **Morphological Operations:** Using morphological closing and opening to clean up binary detection maps and connect fragmented components.

### Additional concepts used -
1.  **Unsupervised Machine Learning (K-Means Clustering):** Used to automatically segment the noise residual map into "authentic" and "tampered" clusters without prior training data.
2.  **Statistical Analysis:** Calculation of Z-scores and Correlation Matrices for the block-based Copy-Move forgery detection algorithm.

### Dataset -
1.  **CASIA 2.0 Image Tampering Detection Dataset:** A standard benchmark dataset for splicing and copy-move forgery. Available at: [Kaggle - CASIA 2.0](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset).
2.  **Custom Dataset:** Created by manually splicing objects (e.g., inserting a centipede into a image of a sand, adding a cat to a backround image) using photo editing tools like Photoshop, GIMP, or edit with Microsoft paint saved as high-quality JPEGs to simulate realistic forgeries.

### Novelty -
1.  **Evidence Fusion Algorithm:** Unlike single-method detectors, this project implements a weighted summation of five different forensic maps to reduce false positives and create a highly confident "Combined Suspicion Map."
2.  **Modular OOP Architecture:** The entire system is encapsulated in a MATLAB `classdef`, making the code modular, scalable, and easy to integrate into larger forensic pipelines compared to standard procedural scripts.


## Outputs:
* **Forensic Dashboard:** The system generates a 3x4 grid of subplots showing intermediate results (ELA, Noise Map, Edge Map) and the final decision.
* **Final Output Image:** A fused "Tampering Overlay" where red pixels indicate high probability of forgery.

![Example Output Dashboard](Outputs/bird_output.jpg)
*(Note: This image serves as a visual proof of the tool's functionality on the custom dataset)*

## References:
1.  Farid, H. (2009). "Image forgery detection." *IEEE Signal Processing Magazine*, 26(2), 16-25.
2.  Fridrich, J., Soukal, D., & Lukáš, J. (2003). "Detection of Copy-Move Forgery in Digital Images." *Proceedings of DFRWS*.

## Limitations and Future Work:
* **Computation Speed:** The Copy-Move detection algorithm is computationally expensive (O(n^2)) due to block matching. Future work could implement keypoint-based methods (SIFT/SURF) for faster performance.
* **Robustness:** The current ELA and Artifact methods rely on JPEG compression traces. Future improvements could include deep learning models (CNNs) to detect manipulation in uncompressed (TIFF/PNG) images.](https://github.com/moonsandsk/Image-Forensics.git)
