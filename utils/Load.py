import os
import cv2

class Load:
    def load(self, path):
        """
        :param path: a string
        :return self.images: a list of numpy
                self.labels: a list of list of integer
        """
        self.images = []
        self.labels = []
        # path내에 labels.txt가 있다는 것을 보장해야함
        with open(os.path.join(path, 'labels.txt'), 'r') as f:
            for line in f.readlines():
                line = line[:-1]
                fname, labels = line.split('\t')
                img = cv2.imread(os.path.join(path, fname))
                self.images.append(img)
                self.labels.append([int(i) for i in labels.split(',')])

    def get(self):
        """
        :return self.images: a list of numpy
                self.labels: a list of list of integer
        """
        return self.images, self.labels