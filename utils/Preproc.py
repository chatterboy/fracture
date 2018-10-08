import numpy as np

class Preproc:
    def subtract_in_each_dim(self, images):
        """
            각 차원에서 평균을 계산하여 mean subtract을 적용함
        :param images: a numpy, NHWC
        :return subtracted images: a numpy ,NHWC
        """
        means = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        subtracted = np.array(images, dtype=np.float32)
        for n in range(images.shape[0]):
            for h in range(images.shape[1]):
                for w in range(images.shape[2]):
                    means += images[n, h, w]
        means /= images.shape[0] * images.shape[1] * images.shape[2] * images.shape[3]
        for n in range(images.shape[0]):
            for h in range(images.shape[1]):
                for w in range(images.shape[2]):
                    subtracted[n, h, w] = images[n, h, w] - means
        return subtracted

    def subtract_in_all_dim(self, images):
        """
            모든 차원에 대한 평균을 계산하여 mean subtract을 적용함
        :param images: a numpy, NHWC
        :return subtracted images: a numpy, NHWC
        """
        mean = 0.0
        subtracted = np.array(images, dtype=np.float32)
        for n in range(images.shape[0]):
            for h in range(images.shape[0]):
                for w in range(images.shape[0]):
                    for c in range(images.shape[0]):
                        mean += images[n, h, w, c]
        mean /= images.shape[0] * images.shape[1] * images.shape[2] * images.shape[3]
        for n in range(images.shape[0]):
            for h in range(images.shape[0]):
                for w in range(images.shape[0]):
                    for c in range(images.shape[0]):
                        subtracted[n, h, w, c] = images[n, h, w, c] - mean
        return subtracted