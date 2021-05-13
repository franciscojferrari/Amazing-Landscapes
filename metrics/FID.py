import numpy
from numpy import asarray
from numpy import cov
from numpy import iscomplexobj
from numpy import trace
from scipy.linalg import sqrtm
from skimage.transform import resize
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation

        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis = 0), cov(act1, rowvar = False)
    mu2, sigma2 = act2.mean(axis = 0), cov(act2, rowvar = False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def run_fid(images1, images2):
    model = InceptionV3(include_top = False, pooling = 'avg', input_shape = (256, 256, 3))

    # Image1 and image2 should be tensor or numpy array with 3 color channels, with values in the range [0, 255]
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    fid = calculate_fid(model, images1, images1)
    return fid
