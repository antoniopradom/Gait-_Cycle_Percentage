import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from AnalysisTools import *

tb_Parent = './results/scalars'
doScalarsPlots(dataParent=tb_Parent, regenerate=True)
evalPath = "./results/evalSubs.pickle"
lapsPath = "./results/evalSubsLaps.pickle"
perModelSubPath = "./results//perModelSub.pickle"

redoLapsPickle = False
if redoLapsPickle:
    with open(evalPath, 'rb') as f:
        allSubs = pickle.load(f)

    justLaps = []
    for test in allSubs:
        pdAux = useJustLaps(allSubs[test])
        subID = test[:6]
        modelType = test[7:]
        pdAux['SubID'] = subID
        pdAux['Model'] = modelType
        justLaps.append(pdAux.copy())

    justLaps = pandas.concat(justLaps)
    justLaps['Error'] = justLaps['gt'] - justLaps['pred']
    justLaps['Abs_Error'] = np.abs(justLaps['gt'] - justLaps['pred'])
    justLaps['Square_Error'] = np.power(justLaps['gt'] - justLaps['pred'], 2)
    justLaps.reset_index(inplace=True, drop=True)

    with open(lapsPath, 'wb') as f:
        pickle.dump(justLaps, f)
else:
    with open(lapsPath, 'rb') as f:
        justLaps = pickle.load(f)

subs = justLaps['SubID'].unique()
modelTypes = justLaps['Model'].unique()
redoPerModelSub = False
if redoPerModelSub:
    perModelSubsData = []
    perModelSubsDataTraj = []

    for subID in subs:
        for model in modelTypes:
            m = np.logical_and(justLaps['Model'] == model, justLaps['SubID'] == subID)
            s1 = justLaps.where(m).dropna()
            rets = calculateErrors(s1)
            pdAux = pandas.DataFrame(columns=['meanError', 'meanAbsError', 'rms',
                                              'lags_meanError', 'lags_meanAbsError', 'lags_rms',
                                              'false_positive', 'false_negative'], index=[model + subID])
            pdAuxTraj = pandas.DataFrame()
            if rets[0] is not np.nan:
                # save the data
                errorDF, falsePos, falseNeg, lagError, predsMean = rets
                pdAux[['meanError', 'meanAbsError', 'rms']] = errorDF.mean().values
                pdAux[['lags_meanError', 'lags_meanAbsError', 'lags_rms']] = lagError.mean().values
                pdAux['false_positive'] = falsePos
                pdAux['false_negative'] = falseNeg
                pdAuxTraj['Values'] = predsMean
                pdAuxTraj['Time'] = np.arange(predsMean.shape[0])
                pdAuxTraj['SubID'] = subID
                pdAuxTraj['Model'] = model

            else:
                errorDF, falsePos, falseNeg, lagError, predsMean = rets
                pdAux[['meanError', 'meanAbsError', 'rms']] = 1
                pdAux[['lags_meanError', 'lags_meanAbsError', 'lags_rms']] = 1
                pdAux['false_positive'] = falsePos
                pdAux['false_negative'] = falseNeg

            pdAux['SubID'] = subID
            pdAux['Model'] = model

            perModelSubsData.append(pdAux.copy())
            perModelSubsDataTraj.append(pdAuxTraj.copy())

    perModelSubsData = pandas.concat(perModelSubsData)
    perModelSubsDataTraj = pandas.concat(perModelSubsDataTraj)
    perModelSubsDataDict = {'scalars': perModelSubsData, 'Traj': perModelSubsDataTraj}

    with open(perModelSubPath, 'wb') as f:
        pickle.dump(perModelSubsDataDict, f)

else:
    with open(perModelSubPath, 'rb') as f:
        perModelSubsDataDict = pickle.load(f)

    perModelSubsData = perModelSubsDataDict['scalars']
    perModelSubsDataTraj = perModelSubsDataDict['Traj']

perModelSubsDataTraj2 = perModelSubsDataTraj.copy()
perModelSubsDataTraj2['Time'] /= 400

sns.lineplot(x='Time', y='Values', hue='Model', data=perModelSubsDataTraj2)
plt.title('Mean cycle per Model')
plt.xlabel('Gait Cycle Percentage')
plt.ylabel('Gait Cycle Prediction')

falsePredPD = []
for fType in ['false_positive', 'false_negative']:
    falsePredPDaux = pandas.DataFrame()
    falsePredPDaux[['Error', 'SubID', 'Model']] = perModelSubsData[[fType, 'SubID', 'Model']]
    falsePredPDaux['Type'] = fType
    falsePredPD.append(falsePredPDaux.copy())

falsePredPD = pandas.concat(falsePredPD)

data = falsePredPD.where(falsePredPD['Type'] == 'false_positive')
ax = sns.boxplot(x='Model', y='Error', data=data)
plt.title('False Positive Heel Strike Identification')
plt.ylabel('Frequency [%]')
plt.tight_layout()

significance = [[0, 1],
                [0, 2],
                [0, 3],
                [0, 4]]
maxHeight = 0.1
minHeight = 0.03
for j, ((c1, c2), h) in enumerate(zip(significance, np.linspace(minHeight, maxHeight, len(significance)))):
    addSignificanceLevel(ax, c1, c2, data, '*', 'Error', offsetPercentage=h)

plt.tight_layout()

data = falsePredPD.where(falsePredPD['Type']=='false_negative')
ax = sns.boxplot(x='Model', y='Error', data=data)
plt.title('False Negative Heel Strike Identification')
plt.ylabel('Frequency [%]')
plt.tight_layout()

significance = [[3, 4],
                [2, 3],
                [1, 2],
                [1, 3]]
maxHeight = 0.1
minHeight = 0.03
for j, ((c1, c2), h) in enumerate(zip(significance, np.linspace(minHeight, maxHeight, len(significance)))):
    addSignificanceLevel(ax, c1, c2, data, '*', 'Error', offsetPercentage=h)

plt.tight_layout()

lagsPD = []
for fType in ['meanError', 'meanAbsError', 'rms']:
    falsePredPDaux = pandas.DataFrame()
    falsePredPDaux[['Error', 'SubID', 'Model']] = perModelSubsData[['lags_' + fType, 'SubID', 'Model']]
    falsePredPDaux['Type'] = fType
    lagsPD.append(falsePredPDaux.copy())

lagsPD = pandas.concat(lagsPD)

perModelSubsData['lags_rms'] = perModelSubsData['lags_rms'].astype(np.float)
ax = sns.boxplot(x='Model', y='lags_rms', data=perModelSubsData)
plt.title('Event ID lag')
plt.ylabel('Delay [ms]')

significance = [[1, 3],
                [1, 4]]
maxHeight = 0.1
minHeight = 0.03
for j, ((c1, c2), h) in enumerate(zip(significance, np.linspace(minHeight, maxHeight, len(significance)))):
    addSignificanceLevel(ax, c1, c2, perModelSubsData, '*', 'lags_rms', offsetPercentage=h)
plt.tight_layout()


errsPD = []
for fType in ['meanError', 'meanAbsError', 'rms']:
    falsePredPDaux = pandas.DataFrame()
    falsePredPDaux[['Error', 'SubID', 'Model']] = perModelSubsData[[fType, 'SubID', 'Model']]
    falsePredPDaux['Type'] = fType
    errsPD.append(falsePredPDaux.copy())

errsPD = pandas.concat(errsPD)

perModelSubsData['rms'] = perModelSubsData['rms'].astype(np.float) * 100
ax = sns.boxplot(x='Model', y='rms', data=perModelSubsData, fliersize=0, whis=0.99)
plt.title('Root Mean Square Error')
plt.ylabel('% Error')

significance = [[3, 4],
                [1, 2],
                [1, 3],
                [1, 4],
                [0, 2],
                [0, 3],
                [0, 4]]
maxHeight = 0.15
minHeight = 0.03
for j, ((c1, c2), h) in enumerate(zip(significance, np.linspace(minHeight, maxHeight, len(significance)))):
    addSignificanceLevel(ax, c1, c2, perModelSubsData, '*', 'rms', offsetPercentage=h)
plt.tight_layout()
