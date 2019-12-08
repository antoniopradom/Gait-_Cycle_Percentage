import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import warnings

import utils as tu


def doScalarsPlots(dataParent, regenerate=False):
    # Do training plots
    if regenerate:
        allData = []

        for folder in os.listdir(dataParent):
            subID = folder[:6]
            for file in os.listdir(dataParent + '/' + folder):
                a2 = readTensorBoard(dataParent + '/' + folder + '/' + file)
                a2['SubID'] = subID
                a2['Test'] = folder[7:]
                a2.reset_index(inplace=True)
                allData.append(a2.copy())

        allData = pandas.concat(allData)

        with open(dataParent + '/trainingScalars.pickle', 'wb') as f:
            pickle.dump(allData, f)
    else:
        with open(dataParent + '/trainingScalars.pickle', 'rb') as f:
            allData = pickle.load(f)

    allData['order'] = 0
    modelTypes = allData['Test'].unique()
    modelTypeNew = ['EDM', 'ERM', 'DM', 'RM']
    desOrder = [2, 3, 0, 1]

    for oldN, newN, orN in zip(modelTypes, modelTypeNew, desOrder):
        m = allData['Test'] == oldN
        allData.loc[m, 'Test'] = newN
        allData.loc[m, 'order'] = orN

    allData.sort_values(['order', 'SubID', 'index'], inplace=True)

    sns.lineplot(x='index', y='epoch_val_mean_absolute_error', hue='Test', data=allData)
    plt.title('Training LOO Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.tight_layout()
    plt.show()

    sns.lineplot(x='index', y='epoch_mean_squared_error', hue='Test', data=allData)
    plt.title('Training  Mean Square Error')
    plt.xlabel('Epoch')
    plt.ylabel('RMS')
    plt.tight_layout()
    plt.show()

    sns.lineplot(x='index', y='epoch_val_mean_squared_error', hue='Test', data=allData)
    plt.title('Training LOO Mean Square Error')
    plt.xlabel('Epoch')
    plt.ylabel('RMS')
    plt.tight_layout()
    plt.show()


def readTensorBoard(f):
    ea = event_accumulator.EventAccumulator(f,
                                            size_guidance={  # see below regarding this argument
                                                event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                event_accumulator.IMAGES: 4,
                                                event_accumulator.AUDIO: 4,
                                                event_accumulator.SCALARS: 0,
                                                event_accumulator.HISTOGRAMS: 1,
                                            })
    ea.Reload()
    a = {}
    for tag in ea.Tags()['scalars']:
        a[tag] = pandas.DataFrame(ea.Scalars(tag))['value'].values

    aP = pandas.DataFrame(a)
    return aP


def useJustLaps(data):
    """
    This function removes all strides that are outside the instrumented section of the Walkway. This is the only section 
    that we can actually compare
    :param data: all the data
    :return: new data
    """
    
    freq = 100.0
    gt = data['gt']
    pred = data['pred']
    
    # if gt and pred are different lengths, cut them
    mLen = min([gt.shape[0], pred.shape[0]])
    gt = gt.iloc[:mLen, :]
    pred = pred.iloc[:mLen, :]

    df = gt[['R', 'L']].diff()
    newData = []
    for lab in ['R', 'L']:
        # lab = 'R'
        hs = df.where(df == -1)[lab].dropna().index.values
        st = np.diff(hs)
        heel_off = np.where(st > st.mean())[0]
        heel_strike = np.concatenate((np.array([0]), heel_off + 1))
        heel_off = np.concatenate((heel_off, np.array([hs.shape[0] - 1])))

        heel_strikeI = (hs[heel_strike] * freq).astype(np.int)
        heel_offI = (hs[heel_off] * freq).astype(np.int)

        tI = np.concatenate([np.arange(stt, ent) for stt, ent in zip(heel_strikeI, heel_offI)])
        pdAux = pandas.DataFrame()
        pdAux['Time'] = tI / freq
        pdAux['gt'] = gt[lab].values[tI]
        pdAux['pred'] = pred[lab].values[tI]
        pdAux['Side'] = lab
        newData.append(pdAux.copy())

    newData = pandas.concat(newData)
    newData.reset_index(inplace=True, drop=True)
    return newData


def calculateErrors(contData):
    freq = 100.0
    n = 400
    gt = contData['gt']
    pred = contData['pred']
    df = gt.diff()
    df.iloc[0] = -1
    dfPred = pred.diff()
    ddfPred = dfPred.diff()
    hsRef = df.where(df == -1).dropna().index.values
    m = np.logical_and(dfPred < -0.2, ddfPred < 0)
    m = m.astype(np.int).diff() > 0
    m.iloc[0] = True
    hsPred = df.where(m).dropna().index.values
    if hsPred.shape[0] is 0:
        warnings.warn('No HS detected on test')
        return np.nan, np.nan, hsRef.shape[0], np.nan, np.zeros(n) * np.nan
    falsePos = 0
    falseNeg = 0
    if hsPred.shape[0] > hsRef.shape[0]:
        falsePos += hsPred.shape[0] - hsRef.shape[0]

    match = [np.argmin(np.abs(hs - hsPred)) for hs in hsRef]
    lag = hsRef - hsPred[match]
    falsePos += (np.abs(lag) > 20).sum()
    hsRef = hsRef[np.abs(lag) < 20]
    hsPred = hsPred[match][np.abs(lag) < 20]
    falseNeg += lag.shape[0] - hsRef.shape[0]
    lag = lag[np.abs(lag) < 20]
    assert hsRef.shape[0] == hsPred.shape[0]
    errorDF = pandas.DataFrame(columns=['meanError', 'meanAbsError', 'rms'], index=np.arange(hsRef.shape[0] - 1),
                               dtype=np.float)

    preds = np.zeros(n)
    c = 0
    for hsRs, hsRe, hsPs, hsPe in zip(hsRef[:-1], hsRef[1:], hsPred[:-1], hsPred[1:]):
        refAux = gt[np.arange(hsRs, hsRe)]
        predAux = pred[np.arange(hsPs + 1, hsPe - 1)]
        preds += tu.reSampleMatrix(predAux.values, n)
        m = np.logical_and(refAux > 0.1, refAux < 0.9)
        errorAux = refAux - predAux
        errorAux = refAux[m] - predAux
        # errorDF['meanError'][c] = errorAux.mean()
        errorDF['meanAbsError'][c] = np.abs(errorAux).mean()
        errorDF['rms'][c] = np.sqrt(np.power(errorAux, 2).mean())
        c += 1
    lagError = pandas.DataFrame(columns=['meanError', 'meanAbsError', 'rms'], index=[0])
    lagError['meanError'][0] = lag.mean() / freq
    lagError['meanAbsError'][0] = np.abs(lag).mean() / freq
    lagError['rms'][0] = np.sqrt(np.power(lag, 2).mean()) / freq
    predsMean = preds / c
    return errorDF, falsePos, falseNeg, lagError, predsMean


def addSignificanceLevel(ax, col1, col2, data, annotation, plotY, yLevel=None, offsetPercentage=0.1, color='k'):
    '''

    :param ax: axis handle
    :type ax: matplotlib.axes._subplots.AxesSubplot
    :param col1: index of column 1, see plt.xticks()
    :param col2: index of column 2
    :param data: pandas dataframe
    :type data: pandas.DataFrame
    :param annotation: string with the annotation
    :param color: line and text color
    '''
    x1, x2 = col1, col2  # columns
    if yLevel is None:
        y = data[plotY].max() + data[plotY].max() * offsetPercentage
    else:
        y = yLevel
    h, col = data[plotY].max() * offsetPercentage, color
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * .5, y + h, annotation, ha='center', va='bottom', color=col)
