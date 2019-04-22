import numpy as np


dataPath = 'C:\\Users\\Student\\Desktop\\data\\data\\'

class knn:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        if metric == 'l2':
            distances = np.sqrt(np.sum((self.train_images - test_image)**2, axis=1))
        else: #l1
            distances = np.sum(np.abs((self.train_images - test_image)), axis=1)

        sorted_index = np.argsort(distances)
        sorted_index = sorted_index[:num_neighbors]

        nearest_neighbors = self.train_labels[sorted_index]
        h = np.bincount(nearest_neighbors)

        return np.argmax(h)

    def classify_images(self, test_images, num_neighbors=3, metrics='l2'):
        num_images = test_images.shape[0]
        predicted_labels = np.zeros(num_images)

        for i in range(num_images):
            predicted_labels[i] = self.classify_image(test_images[i], num_neighbors, metrics)
        return predicted_labels

    def accuracy(self, predicted, labels):
        return np.mean(predicted == labels)


test_images = np.loadtxt(dataPath + 'test_images.txt')
test_labels = np.loadtxt(dataPath + 'test_labels.txt', 'int')
train_images = np.loadtxt(dataPath + 'train_images.txt')
train_labels = np.loadtxt(dataPath + 'train_labels.txt', 'int')

knn_classifier = knn(train_images, train_labels)
classified = knn_classifier.classify_images(test_images, 3, 'l2')
print(knn_classifier.accuracy(classified, test_labels))
