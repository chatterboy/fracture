import os
import cv2

class Load:
    def load_with_labels(self, path):
        """
        :param path: a string
        :return self.images: a list of numpy
                self.labels: a list of list of integer
        """
        self.images = []
        self.labels = []
        # path내에 labels.txt가 있다는 것을 보장해야함
        with open(os.path.join(path, 'labels.txt'), 'r') as f:
            for l in f.read().split('\n'):
                p = l.split('\t')
                img = cv2.imread(os.path.join(path, p[0]))
                # 우선, CT 이미지만 로드
                if img.shape[0] == 512 and img.shape[1] == 512:
                    self.images.append(img)
                    self.labels.append([int(i) for i in p[1].split(',')])

    def get(self):
        """
        :return self.images: a list of numpy
                self.labels: a list of list of integer
        """
        return self.images, self.labels