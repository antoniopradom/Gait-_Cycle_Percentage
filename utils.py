import numpy as np
import os
import tensorflow as tf
import pandas
from sklearn import linear_model



class DatasetProto(object):
    """
    This is dataset proto object, it has a training and a test/validation set
    """
    def __init__(self, x, y, xT, yT, batchSize, numclases=None):
        self.batchsize = batchSize
        self.train = self.DatasetSubclass(x=x, y=y, n=batchSize)
        self.test = self.DatasetSubclass(x=xT, y=yT, n=batchSize)
        dShape = x.shape[1:]
        self.DATA_POINTS = dShape[0]
        self.DATA_SHAPE = dShape
        if numclases is None:
            self.CLASSES_NUM = y.shape[1]
        else:
            self.CLASSES_NUM = numclases

    class DatasetSubclass(object):
        """docstring for DatasetSubclass"""

        def __init__(self, x, y, n, shuffleAfterEpoch=True):
            """
            Creates dataset for training
            :param x: Inputs to the network
            :param y: Outputs of the network
            :param n: batchsize
            """
            self.shuffleAfterEpoch = shuffleAfterEpoch
            self.x = x
            self.y = y
            self.index = 0
            self.n = n
            self.subX = []
            self.subY = []
            self.sameBatch = True
            self.batchStart = 0
            self.batchEnd = 0


class HealthyDataset(DatasetProto):
    """
    Dataset with an option of data split of Leave-One-Out testing
    """
    def __init__(self, allSubs, trainsize, testSize, batchSize, db2use):
        self.allSubs = allSubs
        xUse = db2use[0]
        yUse = db2use[1]
        x = []
        y = []
        xT = []
        yT = []
        useSubTestMode = type(testSize) is not int
        if useSubTestMode:
            # This means that you want to split
            totalSamsPerSub = np.ceil(trainsize / len(allSubs)).astype(np.int)
        else:
            totalSamsPerSub = np.ceil((testSize + trainsize) / len(allSubs)).astype(np.int)

        for subName in allSubs:
            sub = allSubs[subName]
            if useSubTestMode and subName in testSize:
                xT.append(sub[xUse])
                yT.append(sub[yUse][:, -1])
            else:
                shu = randomOrder(sub[xUse].shape[0])
                x.append(sub[xUse][shu[:totalSamsPerSub]])
                y.append(sub[yUse][shu[:totalSamsPerSub], -1])

        if useSubTestMode:
            xx = np.concatenate(x, axis=0)
            yy = np.concatenate(y, axis=0)
            shu = randomOrder(xx.shape[0])
            x = xx[shu].copy()
            y = yy[shu].copy()
            xx = np.concatenate(xT, axis=0)
            yy = np.concatenate(yT, axis=0)
            shu = randomOrder(xx.shape[0])
            xT = xx[shu]
            yT = yy[shu]
        else:
            xx = np.concatenate(x, axis=0)
            yy = np.concatenate(y, axis=0)
            shu = randomOrder(xx.shape[0])
            xT = xx[shu][trainsize:]
            yT = yy[shu][trainsize:]

            x = xx[shu][:trainsize]
            y = yy[shu][:trainsize]

        numClasses = 1

        DatasetProto.__init__(self, x, y, xT, yT, batchSize, numclases=numClasses)


class modelContainer(object):
    """
    This is a container for all models tested, this is just a convenient way of storing the data
    """
    def __init__(self, dataset: DatasetProto, kernels1D, kernels1Dpost, num_gru_units, fullyCon, containerName):
        self.dataset = dataset
        self.learningRate = 1e-3
        self.loss = 'mean_absolute_error'
        self.metrics = ['mean_squared_error', 'mean_absolute_error']
        self.ED_RNN = self._buildCRNN(kernels1D, kernels1Dpost, num_gru_units, fullyCon, self.learningRate)
        self.RNN = self._buildRNN(num_gru_units, fullyCon, self.learningRate)
        self.Ln = self._buildLn(fullyCon, self.learningRate)
        self.ED_Ln = self._buildEDLn(kernels1D, kernels1Dpost, fullyCon, self.learningRate)
        self.Linear = linear_model.LinearRegression()
        self.AllModel = {containerName + '_ED_RNN': self.ED_RNN, containerName + '_RNN': self.RNN,
                         containerName + '_Ln': self.Ln, containerName + '_ED_Ln': self.ED_Ln,
                         containerName + '_Linear': self.Linear}

    def _buildCRNN(self, kernels1D, kernels1Dpost, num_gru_units, fullyCon, learningRate) -> tf.keras.Model:
        dataset = self.dataset
        model = tf.keras.Sequential()
        # Encoder
        model.add(tf.keras.layers.Conv1D(dataset.DATA_SHAPE[1], kernels1D[0], activation=tf.nn.relu, padding='same',
                                         input_shape=dataset.DATA_SHAPE))
        for k in kernels1D[1:]:
            model.add(tf.keras.layers.Conv1D(dataset.DATA_SHAPE[1], k, activation=tf.nn.relu, padding='same'))
        # RNN
        model.add(tf.keras.layers.RNN([tf.keras.layers.GRUCell(n, dropout=0.5) for n in num_gru_units],
                                      return_sequences=True))

        # dense layers
        for k in fullyCon:
            model.add(tf.keras.layers.Dense(k, activation=tf.nn.relu))

        model.add(tf.keras.layers.Dense(dataset.CLASSES_NUM, activation=tf.nn.relu))

        # Decoder
        for k in kernels1Dpost:
            model.add(tf.keras.layers.Conv1D(dataset.DATA_SHAPE[1], k, activation=tf.nn.relu, padding='same'))

        # Flatten
        model.add(tf.keras.layers.Flatten())

        # sigmoid
        model.add(tf.keras.layers.Dense(dataset.CLASSES_NUM, activation=tf.nn.sigmoid))
        opt = tf.keras.optimizers.Adam(lr=learningRate)

        model.compile(optimizer=opt, loss=self.loss,
                      metrics=self.metrics)

        return model

    def _buildEDLn(self, kernels1D, kernels1Dpost, fullyCon, learningRate) -> tf.keras.Model:
        dataset = self.dataset
        model = tf.keras.Sequential()
        # Encoder
        model.add(tf.keras.layers.Conv1D(dataset.DATA_SHAPE[1], kernels1D[0], activation=tf.nn.relu, padding='same',
                                         input_shape=dataset.DATA_SHAPE))
        for k in kernels1D[1:]:
            model.add(tf.keras.layers.Conv1D(dataset.DATA_SHAPE[1], k, activation=tf.nn.relu, padding='same'))

        # dense layers
        for k in fullyCon:
            model.add(tf.keras.layers.Dense(k, activation=tf.nn.relu))

        model.add(tf.keras.layers.Dense(dataset.CLASSES_NUM, activation=tf.nn.relu))

        # Decoder
        for k in kernels1Dpost:
            model.add(tf.keras.layers.Conv1D(dataset.DATA_SHAPE[1], k, activation=tf.nn.relu, padding='same'))

        # Flatten
        model.add(tf.keras.layers.Flatten())

        # sigmoid
        model.add(tf.keras.layers.Dense(dataset.CLASSES_NUM, activation=tf.nn.sigmoid))
        opt = tf.keras.optimizers.Adam(lr=learningRate)

        model.compile(optimizer=opt, loss=self.loss,
                      metrics=self.metrics)

        return model

    def _buildRNN(self, num_gru_units, fullyCon, learningRate) -> tf.keras.Model:
        dataset = self.dataset
        model = tf.keras.Sequential()

        # RNN
        model.add(tf.keras.layers.RNN([tf.keras.layers.GRUCell(n, dropout=0.5) for n in num_gru_units],
                                      return_sequences=True, input_shape=dataset.DATA_SHAPE))

        # dense layers
        for k in fullyCon:
            model.add(tf.keras.layers.Dense(k, activation=tf.nn.relu))

        # Flatten
        model.add(tf.keras.layers.Flatten())

        # sigmoid
        model.add(tf.keras.layers.Dense(dataset.CLASSES_NUM, activation=tf.nn.sigmoid))
        opt = tf.keras.optimizers.Adam(lr=learningRate)

        model.compile(optimizer=opt, loss=self.loss,
                      metrics=self.metrics)

        return model

    def _buildLn(self, fullyCon, learningRate) -> tf.keras.Model:
        dataset = self.dataset
        model = tf.keras.Sequential()

        # Flatten
        model.add(tf.keras.layers.Flatten(input_shape=dataset.DATA_SHAPE))

        # dense layers
        for k in fullyCon:
            model.add(tf.keras.layers.Dense(k, activation=tf.nn.relu))

        # sigmoid
        model.add(tf.keras.layers.Dense(dataset.CLASSES_NUM, activation=tf.nn.sigmoid))

        opt = tf.keras.optimizers.Adam(lr=learningRate)
        model.compile(optimizer=opt, loss=self.loss,
                      metrics=self.metrics)

        return model


def randomOrder(n):
    """
    this function returns an array of index with a random order
    :param n:
    :return:
    """
    shu = np.arange(n)
    np.random.shuffle(shu)
    return shu


def createbatch(x, timeSize, aslist=False):
    """
    This function create samples with a window size of timeSize.
    :param x: original recording as a matrix
    :param timeSize: window size
    :param aslist: return a list or a numpy array
    :return:
    """
    actualSamples = np.arange(x.shape[0] - timeSize) + timeSize
    x0 = np.zeros((timeSize, x.shape[1]), dtype=x.dtype)
    xi = x0.copy()
    xi[-1] = x[0]
    xii = [xi.copy()]
    for i in range(1, timeSize):
        xi = x0.copy()
        xi[-i:] = x[:i]
        xii.append(xi.copy())
    xx = xii + [x[j - timeSize:j, :] for j in actualSamples]
    if not aslist:
        xx = np.stack(xx, axis=0)
    return xx


def createDirIfNotExist(filePath):
    if not os.path.exists(filePath):
        os.makedirs(filePath)


def evaluateSubject(leftShoe, rightShoe, groundTruth, model, freq, batchSize, makeVector=False):
    """
    Evaluated the subject with the given model
    :param leftShoe:
    :param rightShoe:
    :param groundTruth:
    :param model:
    :param freq:
    :param batchSize:
    :param makeVector:
    :return:
    """
    rBatch = createbatch(rightShoe, 50)
    lBatch = createbatch(leftShoe, 50)

    if makeVector:
        rBatch = rBatch.reshape(-1, 12 * 50)
        lBatch = lBatch.reshape(-1, 12 * 50)

    if batchSize is None:
        yRp = model.predict(rBatch)
        yLp = model.predict(lBatch)
    else:
        yRp = model.predict(rBatch, batch_size=batchSize)
        yLp = model.predict(lBatch, batch_size=batchSize)
    mLen = min([yRp.shape[0], yLp.shape[0]])
    yR = np.zeros(mLen)
    yR[:mLen] = np.squeeze(yRp[:mLen])
    yL = np.zeros(mLen)
    yL[:mLen] = np.squeeze(yLp[:mLen])
    preds = {'R': yR, 'L': yL}
    t = np.arange(mLen) / freq
    db = {'gt': groundTruth.y, 'pred': pandas.DataFrame(data=preds, index=t), 'laps': groundTruth.lap}

    return db
