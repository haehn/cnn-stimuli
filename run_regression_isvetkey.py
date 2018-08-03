from keras import models, layers, optimizers, applications, callbacks
from keras import backend as keras
import sklearn.metrics

import cPickle as pickle 
import numpy as np 
import os
import sys
import time
import stimuli as s

EXPERIMENT = sys.argv[1]
DATASET = int(sys.argv[2]) #true flags (encoded in binary)
CLASSIFIER = sys.argv[3]
NOISE = sys.argv[4]

print 'Running', EXPERIMENT, 'with flags', DATASET, 'with', CLASSIFIER

DATATYPE = eval(EXPERIMENT)

# Flags are encoded in binary (i.e. 5 = 101) and the binary representation is the flags in order:
# ex: 0 = 000 = [False, False, False]
# 1 = 001 = [False, False, True]
# 3 = 011 = [False, True, True]
# 6 = 110 = [True, True, False]
# etcetera.

DST = DATASET #temporary variable
FLAGS = [False] * 3 #there are 3 flags
for f in range(len(FLAGS)):
	if(DST % 2) == 1:
		FLAGS[len(FLAGS)-1-f] = True 
	DST = DST // 2

SUFFIX = '.'
if NOISE == 'True':
	NOISE = True
	SUFFIX = '_noise.'
else:
	NOISE = False

PREFIX = '/n/home05/isvetkey/cnn-stimuli/'
RESULTS_DIR = PREFIX + 'RESULTS/'
OUTPUT_DIR = RESULTS_DIR + EXPERIMENT + '/' + str(DATASET) + '/' + CLASSIFIER
if not os.path.exists(OUTPUT_DIR):
	try:
		os.makedirs(OUTPUT_DIR)
	except:
		print 'Race condition, what are you?', os.path.exists(OUTPUT_DIR)

STATSFILE = OUTPUT_DIR + SUFFIX + 'p' #job index?
MODELFILE = OUTPUT_DIR + SUFFIX + 'h5'

print 'Working in', OUTPUT_DIR
print 'Storing', STATSFILE
print 'Storing', MODELFILE

if os.path.exists(STATSFILE) and os.path.exists(MODELFILE):
	print 'we already did this one :\'('
	sys.exit(0)

#
#
# actual data stuff
#
#

train_target = 60000
val_target = 20000
test_target = 20000

global_min = s.Figure5._min(DATATYPE)
global_max = s.Figure5._max(DATATYPE)

X_train = np.zeros((train_target, 100, 150), dtype=np.float32)
y_train = np.zeros((train_target, 4), dtype=np.float32)
train_counter = 0

X_val = np.zeros((val_target, 100, 150), dtype=np.float32)
y_val = np.zeros((val_target, 4), dtype=np.float32)
val_counter = 0

X_test = np.zeros((test_target, 100, 150), dtype=np.float32)
y_test = np.zeros((test_target, 4), dtype=np.float32)
test_counter = 0

train_set = set()
val_set = set()
test_set = set()

t0 = time.time()

all_counter = 0
while train_counter < train_target or val_counter < val_target or test_counter < test_target:
	all_counter += 1

	sparse, image, label, parameters = DATATYPE(FLAGS)

	image = image.astype(np.float32)

	pot = np.random.choice(3, p=[0.6, 0.2, 0.2])

	if pot == 0 and train_counter < train_target:
		#sort it to check whether same angles are in a different order somewhere
		if tuple(np.sort(label)) in val_set or tuple(np.sort(label)) in test_set:
			continue
		#add noise?
		if NOISE:
			image += np.random.uniform(0, 0.05, (100, 150))

		X_train[train_counter] = image
		y_train[train_counter] = label
		train_set.add(tuple(np.sort(label)))
		train_counter += 1

	#repeat process with other 2 sets of data
	elif pot == 1 and val_counter < val_target:
		if tuple(np.sort(label)) in train_set or tuple(np.sort(label)) in test_set:
			continue

		if NOISE:
			image += np.random.uniform(0, 0.05, (100, 150))

		X_val[val_counter] = image
		y_val[val_counter] = label
		val_set.add(tuple(np.sort(label)))
		val_counter += 1

	elif pot == 2 and test_counter < test_target:
		if tuple(np.sort(label)) in train_set or tuple(np.sort(label)) in val_set:
			continue

		if NOISE:
			image += np.random.uniform(0, 0.05, (100, 150))

		X_test[test_counter] = image
		y_test[test_counter] = label
		test_set.add(tuple(np.sort(label)))
		test_counter += 1
	print train_counter, val_counter, test_counter

print 'Done', time.time()-t0, 'seconds (', all_counter, 'iterations)'
#
#
# NORMALIZATION
#
#
X_min = min(X_train.min(), X_val.min(), X_test.min())
X_max = max(X_train.max(), X_val.max(), X_test.max())
y_min = min(y_train.min(), y_val.min(), y_test.min())
y_max = max(y_train.max(), y_val.max(), y_test.max())

# scale in place
X_train -= X_min
X_train /= (X_max - X_min)
y_train -= y_min
y_train /= (y_max - y_min)

X_val -= X_min
X_val /= (X_max - X_min)
y_val -= y_min
y_val /= (y_max - y_min)

X_test -= X_min
X_test /= (X_max - X_min)
y_test -= y_min
y_test /= (y_max - y_min)

# normalize to -.5 .. .5
X_train -= .5
X_val -= .5
X_test -= .5

print 'memory usage', (X_train.nbytes + X_val.nbytes + X_test.nbytes + y_train.nbytes + y_val.nbytes + y_test.nbytes) / 1000000., 'MB'

feature_time = 0
if CLASSIFIER == 'VGG19' or CLASSIFIER == 'XCEPTION':
	X_train_3D = np.stack((X_train,)*3, -1)
  	X_val_3D = np.stack((X_val,)*3, -1)
  	X_test_3D = np.stack((X_test,)*3, -1)

  	#if this gives me some errors: transpose (0, 3, 1, 2) and put the 3 in front on the VGG19 thing
  	if CLASSIFIER == 'VGG19':
  		feature_generator = applications.VGG19(include_top=False, weights=None, input_shape=(100,150,3))
  	elif CLASSIFIER == 'XCEPTION':
  		feature_generator = applications.Xception(include_top=False, weights=None, input_shape=(100,150,3))
  	
  	t0 = time.time()
  	X_train_3D_features = feature_generator.predict(X_train_3D, verbose=True)
 	X_val_3D_features = feature_generator.predict(X_val_3D, verbose=True)
  	feature_time = time.time()-t0

  	X_test_3D_features = feature_generator.predict(X_test_3D, verbose=True)
  	print CLASSIFIER, 'features done after', time.time()-t0
  	print 'memory usage', (X_train_3D_features.nbytes + X_val_3D_features.nbytes + X_test_3D_features.nbytes) / 1000000., 'MB'

  	# update the shape
  	feature_shape = X_train_3D_features.shape[1] * X_train_3D_features.shape[2] * X_train_3D_features.shape[3]

  	X_train = X_train_3D_features.reshape(len(X_train_3D_features), feature_shape)
  	X_val = X_val_3D_features.reshape(len(X_val_3D_features), feature_shape)
  	X_test = X_test_3D_features.reshape(len(X_test_3D_features), feature_shape)

  	MLP = models.Sequential()

elif CLASSIFIER == 'LeNet':
	classifier = models.Sequential()
	classifier.add(layers.Convolution2D(20, (5, 5), padding="same", input_shape=(100, 100, 1)))
	classifier.add(layers.Activation("relu"))
	classifier.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	classifier.add(layers.Dropout(0.2))
	classifier.add(layers.Convolution2D(50, (5, 5), padding="same"))
	classifier.add(layers.Activation("relu"))
	classifier.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	classifier.add(layers.Dropout(0.2))
	classifier.add(layers.Flatten())

	feature_shape = classifier.output_shape
	MLP = classifier

	X_train = X_train.reshape(len(X_train), 100, 150, 1)
	X_val = X_val.reshape(len(X_val), 100, 150, 1)
	X_test = X_test.reshape(len(X_test), 100, 150, 1)

elif CLASSIFIER == 'MLP':
	MLP = models.Sequential()

	# flatten the data
	X_train = X_train.reshape(len(X_train), 100*150)
	X_val = X_val.reshape(len(X_val), 100*150)
	X_test = X_test.reshape(len(X_test), 100*150)

	feature_shape = 100*150

MLP.add(layers.Dense(256, activation='relu', input_dim=feature_shape))
MLP.add(layers.Dropout(0.5))
MLP.add(layers.Dense(4, activation='linear')) # this gave me an error last time, keep?

sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
MLP.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse', 'mae']) # MSE for regression

t0 = time.time()
callbacks = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'), \
             callbacks.ModelCheckpoint(MODELFILE, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

history = MLP.fit(X_train, \
                  y_train, \
                  epochs=1000, \
                  batch_size=32, \
                  validation_data=(X_val, y_val),
                  callbacks=callbacks,
                  verbose=True)
fit_time = time.time()-t0

print 'Fitting done', time.time()-t0

#actually predict something now
y_pred = MLP.predict(X_test)

# MLAE (from what I understand it's only there to compare with CMG85)
MLAE = np.log2(sklearn.metrics.mean_absolute_error(y_pred*100, y_test*100)+.125)

# store stats
stats = dict(history.history)
# 1. the training history
# 2. the y_pred and y_test values
# 3. the MLAE
stats['time'] = feature_time + fit_time
stats['y_test'] = y_test
stats['y_pred'] = y_pred
stats['MLAE'] = MLAE

with open(STATSFILE, 'w') as f:
  pickle.dump(stats, f)

print 'MLAE', MLAE
print 'Written', STATSFILE
print 'Written', MODELFILE
print 'All done. Give your computer a pat on the back.'
time.sleep(.7)
print 'actually don\'t do that.'
