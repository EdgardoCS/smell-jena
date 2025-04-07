"""
author: EdgardoCS @FSU Jena
date: 31/03/2025
"""

import pyclesperanto_prototype as cle

cle.get_device()

import numpy as np
from skimage.io import imshow

random_image = np.random.random([512,512])
binary_image = random_image > 0.9995

input_image = cle.push(binary_image * random_image)

imshow(input_image)

sigma = 3
starting_point = cle.gaussian_blur(input_image, sigma_x=sigma, sigma_y=sigma)

maxima = cle.detect_maxima_box(starting_point)

# Label maxima
labeled_maxima = cle.label_spots(maxima)

# read out intensities at the maxima
max_intensities = cle.read_intensities_from_map(labeled_maxima, starting_point)
print(max_intensities)

# calculate thresholds
thresholds = cle.multiply_image_and_scalar(max_intensities, scalar=0.5)
print(thresholds)

# Extend labeled maxima until they touch
voronoi_label_image = cle.extend_labeling_via_voronoi(labeled_maxima)

# Replace labels with thresholds
threshold_image = cle.replace_intensities(voronoi_label_image, thresholds)

# Apply threshold
binary_segmented = cle.greater(starting_point, threshold_image)

# Label objects
labels = cle.connected_components_labeling_box(binary_segmented)

