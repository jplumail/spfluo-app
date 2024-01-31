from sklearn.cluster import KMeans
from sklearn.svm import SVC

from spfluo.utils.array import Array, array_namespace


def separate_centrioles(im: Array, threshold_percentage: float = 0.5):
    xp = array_namespace(im)

    # Thresholding
    points = xp.stack((im > xp.max(im) * threshold_percentage).nonzero(), axis=-1)
    points = xp.to_device(points, "cpu")

    # Clustering
    kmeans = KMeans(n_clusters=2, n_init="auto")
    kmeans.fit(points)

    # Separate clusters
    svc = SVC(kernel="linear")
    svc.fit(points, kmeans.labels_)

    # Compute hyperplane
    image_coords = xp.stack(
        xp.meshgrid(*[xp.arange(s) for s in im.shape], indexing="ij"), axis=-1
    )
    proj = (image_coords @ xp.asarray(svc.coef_.T))[..., 0] + xp.asarray(svc.intercept_)

    # Output 2 images
    mask1 = proj > 0
    im1 = xp.zeros([(x.max() - x.min() + 1) for x in xp.nonzero(mask1)])
    mask11 = mask1[tuple([slice(x.min(), x.max() + 1) for x in xp.nonzero(mask1)])]
    im1[xp.to_device(mask11, xp.device(im))] = im[xp.to_device(mask1, xp.device(im))]

    mask2 = ~mask1
    im2 = xp.zeros([(x.max() - x.min() + 1) for x in xp.nonzero(mask2)])
    mask22 = mask2[tuple([slice(x.min(), x.max() + 1) for x in xp.nonzero(mask2)])]
    im2[xp.to_device(mask22, xp.device(im))] = im[xp.to_device(mask2, xp.device(im))]

    return im1, im2
