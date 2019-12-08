import pickle

import tensorflow as tf

import utils as tu

data_path = './'
allSubsPath = data_path + 'allSubs.pickle'
resultsParent = "./results/"

with open(allSubsPath, 'rb') as f:
    allSubs = pickle.load(f)

kernels1D = [20, 10, 5]
kernels1Dpost = [20, 10, 5]
fullyCon = [32, 64, 512]
EVALUATE_SUB = True
# num_gru_units = None
num_gru_units = [5 for _ in range(1)]
batchSize = 5000
trainSize = len(allSubs) * 5000
freq = 100.0
maxEpochs = 500
# for each subject of the LOO
modelName = 'LOO'
resultsPath = resultsParent + modelName + '/'
LOOmodels = {}
LOOevals = {}

for sub in allSubs:
    testSize = [sub]
    dataset = tu.HealthyDataset(allSubs, trainSize, testSize, batchSize, db2use=['xP', 'yP'])
    contAux = tu.modelContainer(dataset, kernels1D, kernels1Dpost, num_gru_units, fullyCon, sub)
    # for each model
    for modelName in contAux.AllModel:
        print('Doing %s' % modelName)
        model = contAux.AllModel[modelName]
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=resultsParent + 'scalars/' + modelName)
        if 'Linear' in modelName:
            model.fit(dataset.train.x.reshape((-1, dataset.DATA_SHAPE[0]*dataset.DATA_SHAPE[1])), dataset.train.y)
            makeVector = True
        else:
            model.fit(dataset.train.x, dataset.train.y, batch_size=batchSize, epochs=maxEpochs,
                      validation_data=(dataset.test.x, dataset.test.y), callbacks=[tensorboard_callback])
            makeVector = False

        tu.createDirIfNotExist(resultsPath)
        model.save(resultsPath + modelName + '.h5')
        if EVALUATE_SUB:
            LOOevals[modelName] = tu.evaluateSubject(allSubs[sub].left, allSubs[sub].right, allSubs[sub].mat, model,
                                                     freq, batchSize, makeVector=makeVector)


if EVALUATE_SUB:
    with open(resultsPath + 'evalSubs.pickle', 'wb') as f:
        pickle.dump(LOOevals, f)
