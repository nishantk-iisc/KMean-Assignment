from matplotlib import pyplot as plt
import numpy as np

def get_image(image_path):
    image = plt.imread(image_path)
    return image/255.0


def show_image(image):
    plt.imshow(image)
    plt.show()

def save_image(image, image_path):
    plt.imsave(image_path, image)


def error(original_image: np.ndarray, clustered_image: np.ndarray) -> float:
    # Returns the Mean Squared Error between the original image and the clustered image
    # Flatten the images to 1D arrays
    original = original_image.reshape(-1)
    clustered = clustered_image.reshape(-1)

    # Compute the squared difference between the original and clustered images
    squared_difference = np.square(original - clustered)

    # Compute the mean squared error
    mse = np.mean(squared_difference)

    return mse
    # raise NotIcleamplementedError