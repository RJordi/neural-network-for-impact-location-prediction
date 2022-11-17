print("Loading Libraries")
import scipy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt
import datetime
from scipy.ndimage.filters import gaussian_filter
print("Libraries Loaded")

print("Loading Function Files")
from prepro_dataConversion import *
from prepro_Augmentation import *
from plot_heatmap import plot_heatmap
from gaussian_augmentation import gaussian_augmentation
print("Function files loaded")

cwd = os.getcwd() #Get the current directory
numDataDir = cwd + "\\Data" + "\\Numerical_Data" # "Access" the numerical data
numDataDir_2 = cwd + "\\Data" + "\\Experimental_Validation_Data\\" # "Access" the validation data
numDataDir_3 = cwd + "\\Data" + "\\Experimental_Group_Data\\" # "Access" the testing data

experimentList,labelList,labelListNormalised = dataConversion(numDataDir)

#experimentList,labelList, labelListNormalised = gaussian_augmentation(experimentList,labelList,labelListNormalised)

labListA,labListNA = Augmentation(experimentList,labelList,labelListNormalised)
#labListA and labListNA are in local coordinate form after the augmentation
#Also, note that experiment.x and experiment.y are localized and normalized
#Signals are not normalized

val_labelList, val_experimentList = ExpdataConversionVal(numDataDir_2, labListA)
test_labelList, test_experimentList = ExpdataConversionVal(numDataDir_3, labListA)


#Concstruction of the Matrix for the ANN input
labels = np.linspace(0,len(experimentList)-1,num = len(experimentList))
coordinateNo = len(experimentList)
lenData = 200
sensorNo = 8
matANN = np.zeros((coordinateNo,sensorNo,lenData))
for i in range(coordinateNo):
    normalize = np.amax(np.abs([-experimentList[i].s1[263:-1:99],-experimentList[i].s2[263:-1:99],-experimentList[i].s3[263:-1:99],-experimentList[i].s4[263:-1:99]]))
    matANN[i][0] = gaussian_filter(-experimentList[i].s1[263:-1:99]/normalize,sigma = 3)
    matANN[i][1] = gaussian_filter(-experimentList[i].s2[263:-1:99]/normalize,sigma = 3)
    matANN[i][2] = gaussian_filter(-experimentList[i].s3[263:-1:99]/normalize,sigma = 3)
    matANN[i][3] = gaussian_filter(-experimentList[i].s4[263:-1:99]/normalize,sigma = 3)

    
    matANN[i][4] = np.gradient(matANN[i][0])
    matANN[i][5] = np.gradient(matANN[i][1])
    matANN[i][6] = np.gradient(matANN[i][2])
    matANN[i][7] = np.gradient(matANN[i][3])
    '''
    normalize = np.amax(np.abs([matANN[i][4], matANN[i][5], matANN[i][6], matANN[i][7]]))
    matANN[i][4] = matANN[i][4]/normalize
    matANN[i][5] = matANN[i][5]/normalize
    matANN[i][6] = matANN[i][6]/normalize
    matANN[i][7] = matANN[i][7]/normalize
    '''
    '''
for i in range(18):
    plt.figure()
    plt.plot(test_experimentList[i][0])
    plt.plot(test_experimentList[i][1])
    plt.plot(test_experimentList[i][2])
    plt.plot(test_experimentList[i][3])
    plt.grid()
    plt.title("Experiment " + str(i) + " Data")
    plt.show()
    plt.close()
'''
matANN = np.reshape(matANN,(coordinateNo,lenData,sensorNo))
val_experimentList = np.reshape(val_experimentList,(len(val_experimentList),lenData,sensorNo))
test_experimentList = np.reshape(test_experimentList,(len(test_experimentList),lenData,sensorNo))

print("\nTraining data shape (matANN): " + str(matANN.shape))
print("Validation data shape (val_experimentList): " + str(val_experimentList.shape))
print("Testing data shape (test_experimentList): " + str(test_experimentList.shape) + '\n')

num_filters = 32
filter_size = 8
pool_size = 2
#act = tf.keras.activations.elu
act = 'relu'
################################   NEURAL NETWORK MODEL   ################################
'''
def call_existing_code(num_layers, num_filters, filter_size, pool_size, stride, lr, padding):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(num_filters,filter_size,input_shape = (lenData,sensorNo),padding = padding,name = 'conv1',activation = 'relu'))
    for i in range(num_layers):
        model.add(tf.keras.layers.Conv1D(num_filters, filter_size,padding=padding,name=f'conv{i+2}',activation='relu'))
        model.add(tf.keras.layers.Dropout(0.05))
        model.add(tf.keras.layers.MaxPooling1D(strides=stride,pool_size = pool_size))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(len(labels),activation = 'softmax'))
    learning_rate = lr
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = opt,loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])
    return model

def call_error_code():
    model = tf.keras.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(len(labels),activation = 'softmax')])
    model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
    return model

model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(num_filters,filter_size,input_shape = (lenData,sensorNo),padding = 'same',name = 'conv1',activation = 'relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.MaxPooling1D(strides=4,pool_size = pool_size),
            tf.keras.layers.Conv1D(num_filters,filter_size,padding = 'same',name = 'conv2',activation = 'relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.MaxPooling1D(strides=4,pool_size = pool_size),
            tf.keras.layers.Flatten(),
            #tf.keras.layers.Dense(32,activation = 'relu'),
            tf.keras.layers.Dense(len(labels),activation = 'softmax')
            ])
'''
model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(num_filters,filter_size,input_shape = (lenData,sensorNo),padding = 'same',activation = act),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.MaxPooling1D(strides=4,pool_size = pool_size),
            tf.keras.layers.Conv1D(num_filters,filter_size,padding = 'same', activation = act),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.MaxPooling1D(strides=4,pool_size = pool_size),
            tf.keras.layers.Conv1D(num_filters,filter_size,padding = 'same',activation = act),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.MaxPooling1D(strides=4,pool_size = pool_size),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64,activation = act),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64,activation = act),
            tf.keras.layers.Dense(len(labels),activation = 'softmax')
            ])

#opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

#model.summary()
'''
def build_model(hp):
    num_layers = hp.Int('num_layers', 1,4)
    num_filters = hp.Int('num_filters,', min_value=10, max_value=40, step=10)
    filter_size = hp.Int('filter_size', min_value=10, max_value=30, step=5)
    pool_size = hp.Choice('pool_size', [4, 8, 16])
    stride = hp.Choice('stride', [4, 8, 16, 32])
    learning_rate = hp.Fixed('lr', 0.001)
    padding = hp.Fixed('padding', 'same')
    # call existing model-building code with the hyperparameter values.
    try:
        model = call_existing_code(num_layers=num_layers, num_filters=num_filters,
            filter_size=filter_size, pool_size=pool_size, stride=stride,
            lr=learning_rate, padding=padding)
    except:
        model = call_error_code()
        print('ERROR  -->  Invalid dimensions!\n')
    return model

#build_model(kt.HyperParameters())

tuner = kt.RandomSearch(
    hypermodel=build_model, tune_new_entries=True,
    max_trials=5, executions_per_trial=2, overwrite=True,
    objective='val_loss', directory=cwd, project_name='hyperparameter tuning CIE21')
early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tensorboard_cb = tf.keras.callbacks.TensorBoard(cwd+'/TensorBoard/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tuner.search(matANN, labels, validation_data=(val_experimentList, val_labelList),
    epochs=50, batch_size=128, callbacks=[early_stop_cb, tensorboard_cb])

best_hp = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hp)
'''
################################   NEURAL NETWORK MODEL   ################################

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=cwd+'\\checkpoint\\weights.best.hdf5',
    save_best_only=True, monitor='val_loss', mode='min')
model.fit(matANN,labels, validation_data=(val_experimentList, val_labelList),
    epochs = 40, batch_size = round(len(labels)/10), callbacks=[early_stop, checkpoint])
model.summary()

model.load_weights(cwd+'\\checkpoint\\weights.best.hdf5')

predictions = model.predict(val_experimentList)

#print('Predictions: \n', predictions, '\n')
print('\nShape of predictions: ', np.shape(predictions), '\n')

print('Shape of test_labelList: ', np.shape(test_labelList), '\n')
#print('test_labelList: ', test_labelList, '\n')

print('PREDICTION | REALITY')

errorArray = np.zeros((1,len(predictions)))
sum = 0

for i in range(len(predictions)):
    index = np.argmax(predictions[i])
    predVector = np.array([labListA[0,index]+250,labListA[1,index]+250])
    realVector = np.array([labListA[0,int(val_labelList[i])]+250,labListA[1,int(val_labelList[i])]+250])
    print(predVector, realVector)

    #Weighted interpolation of predicted coordinates
    highest_probs = np.argsort(predictions[i])[-7:]
    best_predictions_x = [labListA[0,index]+250 for index in highest_probs]
    best_predictions_y = [labListA[1,index]+250 for index in highest_probs]
    weighted_x = np.average(best_predictions_x, weights=predictions[i][highest_probs])
    weighted_y = np.average(best_predictions_y, weights=predictions[i][highest_probs])
    interp_prediction = np.array([weighted_x, weighted_y])
    #interp_prediction = np.around(np.array([weighted_x, weighted_y])/5, decimals=0)*5 # round to nearest multiple of 5
    #plot_heatmap(predictions[i], realVector, labListA, interp_prediction)    

    #Next parts are to plot the error/accuracy: applies only to validation experimental data
    coord,coordreal = kConvert(int(val_labelList[i]),labListNA,150,250)
    errorVector = interp_prediction - coordreal # change predVector with interp_prediction to have error with interpolation
    normError = np.sqrt(errorVector[0]**2+errorVector[1]**2)
    if normError > 10:
        #print(interp_prediction, coordreal)
        plot_heatmap(predictions[i], coordreal, labListA, interp_prediction)
    errorArray[0,i] = normError
    sum = sum + normError

avNorm = sum/len(predictions)
print("Average Error Norm = ",str(avNorm))

#Taking Percentage of Norm under certain radius
errorTol = 5
sum = 0
for i in range(len(predictions)):
    if errorArray[0,i] <= errorTol:
        sum = sum+1
percNorm = sum/len(predictions)*100

#Plotting Histogram
print("Percentage of error below",str(errorTol),"is:",str(percNorm),"%")

hist, bin_edges = np.histogram(errorArray)
plt.bar(bin_edges[:-1], hist, width = (max(bin_edges)-min(bin_edges))/10)
plt.xlim(min(bin_edges), max(bin_edges))
xticks = np.arange(min(bin_edges), max(bin_edges), (max(bin_edges)-min(bin_edges))/20)
for i in range(int((len(xticks))/2)):
    xticks = np.delete(xticks,i*2-i)
plt.xticks(xticks)
plt.grid(axis = 'x')
plt.xlabel("Error Norm")
plt.ylabel("Frequency")
titleStr = "Error distribution of experimental data"
plt.title(titleStr)
figStr = titleStr + ".png"
plt.savefig(figStr)
plt.show()
