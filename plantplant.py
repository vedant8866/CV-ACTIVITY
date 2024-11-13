import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to extract GLCM texture features
def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

# Function to apply various effects
def enhance_image(image):
    # 1. Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # 2. Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)

    # 3. Canny Edge Detection
    edges = cv2.Canny(image, 100, 200)

    # 4. Contours Detection
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.copy(image)
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)  # Blue contours

    return blurred_image, adaptive_thresh, edges, contour_image

# Function for texture segmentation using K-means
def texture_segmentation(image):
    pixels = image.reshape(-1, 1)  # Flatten the image for clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
    segmented_image = kmeans.labels_.reshape(image.shape)
    return segmented_image

# Load an image (replace with your path)
image = cv2.imread("C:/Users/9254g/Downloads/Real-Time-Face-Recognition-master/Real-Time-Face-Recognition-master/plant_leaf.jpg", cv2.IMREAD_GRAYSCALE)

# Resize for faster processing
image = cv2.resize(image, (200, 200))

# Step 1: Extract GLCM features from the image
texture_features = extract_glcm_features(image)

# Step 2: Apply K-means texture-based segmentation
segmented_image = texture_segmentation(image)

# Apply additional effects
blurred_image, adaptive_thresh, edges_image, contour_image = enhance_image(image)

# Display the results
plt.figure(figsize=(15, 12))

# Displaying the effects
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(2, 3, 2)
plt.title("Gaussian Blurred Image")
plt.imshow(blurred_image, cmap='gray')

plt.subplot(2, 3, 3)
plt.title("Adaptive Thresholding")
plt.imshow(adaptive_thresh, cmap='gray')

plt.subplot(2, 3, 4)
plt.title("Edge Detection (Canny)")
plt.imshow(edges_image, cmap='gray')

plt.subplot(2, 3, 5)
plt.title("Contour Detection")
plt.imshow(contour_image, cmap='gray')

plt.subplot(2, 3, 6)
plt.title("Segmented Image")
plt.imshow(segmented_image, cmap='gray')

plt.tight_layout()
plt.show()
