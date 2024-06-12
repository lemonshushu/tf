import os
import random
import numpy as np
import pickle
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dot
from keras import optimizers
from keras.callbacks import CSVLogger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# This is tuned hyper-parameters
alpha = 0.1
batch_size_value = 128
emb_size = 64
number_epoch = 30

description = 'Triplet_Model'
Training_Data_PATH = '../../dataset/extracted_AWF775/'
print(Training_Data_PATH)
print(f"with parameters, Alpha: {alpha}, Batch_size: {batch_size_value}, Embedded_size: {emb_size}, Epoch_num: {number_epoch}")

alpha_value = float(alpha)
print(description)

# ================================================================================
# This part is to prepare the files' index for generating triplet examples
# and formulating each epoch inputs

# Extract all folders' names
dirs = sorted(os.listdir(Training_Data_PATH))

# Each given folder name (URL of each class), we assign class id
# e.g. {'adp.com' : 23, ...}
name_to_classid = {d:i for i,d in enumerate(dirs)}

# Just reverse from previous step
# Each given class id, show the folder name (URL of each class)
# e.g. {23 : 'adp.com', ...}
classid_to_name = {v:k for k,v in name_to_classid.items()}

num_classes = len(name_to_classid)
print(f"number of classes: {num_classes}")

# Each directory, there are n traces corresponding to the identity
# We map each trace path with an integer id, then build dictionaries
# We are mapping
#   path_to_id and id_to_path
#   classid_to_ids and id_to_classid

# read all directories
# c is class
# name_to_classid.items() contains [(directory, classid), ('slickdeals.net', 547), ...]

trace_paths = {c: Training_Data_PATH + directory + '/data.pkl' for directory, c in name_to_classid.items()}
# trace_paths --> {0: '104.com.tw/data.pkl', 1: 'slickdeals.net/data.pkl', ...}

# retrieve all traces
all_traces = []
for path in trace_paths.values():
    with open(path, 'rb') as handle:
        each_trace = pickle.load(handle)
    all_traces.append(each_trace)

all_traces = np.concatenate(all_traces)
all_traces = all_traces[:, :, np.newaxis]
print("Load traces with ", all_traces.shape)
print("Total size allocated on RAM : ", str(all_traces.nbytes / 1e6) + ' MB')

# map to integers
path_to_id = {i: i for i in range(len(all_traces))}
id_to_path = {v: k for k, v in path_to_id.items()}

# build mapping between traces and class
classid_to_ids = {k: list(range(sum(len(trace) for trace in all_traces[:k]), sum(len(trace) for trace in all_traces[:k + 1]))) for k in trace_paths.keys()}
id_to_classid = {v: c for c, traces in classid_to_ids.items() for v in traces}

def build_pos_pairs_for_id(classid):
    traces = classid_to_ids[classid]
    pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))]
    random.shuffle(pos_pairs)
    return pos_pairs

def build_positive_pairs(class_id_range):
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id)
        for pair in pos:
            listX1 += [pair[0]]
            listX2 += [pair[1]]
    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]

Xa_train, Xp_train = build_positive_pairs(range(0, num_classes))

# Gather the ids of all network traces that are used for training
all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
print("X_train Anchor: ", Xa_train.shape)
print("X_train Positive: ", Xp_train.shape)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def cosine_triplet_loss(X):
    _alpha = alpha_value
    positive_sim, negative_sim = X

    losses = K.maximum(0.0, negative_sim - positive_sim + _alpha)
    return K.mean(losses)

def build_similarities(conv, all_imgs):
    embs = conv.predict(all_imgs)
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    all_sims = np.dot(embs, embs.T)
    return all_sims

def intersect(a, b):
    return list(set(a) & set(b))

def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, num_retries=50):
    if similarities is None:
        return random.sample(neg_imgs_idx, len(anc_idxs))
    final_neg = []
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
        anchor_class = id_to_classid[anc_idx]
        sim = similarities[anc_idx, pos_idx]
        possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
        possible_ids = intersect(neg_imgs_idx, possible_ids)
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(neg_imgs_idx))
    return final_neg

class SemiHardTripletGenerator():
    def __init__(self, Xa_train, Xp_train, batch_size, all_traces, neg_traces_idx, conv):
        self.batch_size = batch_size

        self.traces = all_traces
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_traces_idx = neg_traces_idx
        self.all_anchors = list(set(Xa_train))
        self.mapping_pos = {v: k for k, v in enumerate(self.all_anchors)}
        self.mapping_neg = {k: v for k, v in enumerate(self.neg_traces_idx)}
        if conv:
            self.similarities = build_similarities(conv, self.traces)
        else:
            self.similarities = None

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.num_samples:
                self.cur_train_index = 0

            traces_a = self.Xa[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_p = self.Xp[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx)

            yield ([self.traces[traces_a],
                    self.traces[traces_p],
                    self.traces[traces_n]],
                   np.zeros(shape=(traces_a.shape[0])))

# Training the Triplet Model
from DF_model import DF
shared_conv2 = DF(input_shape=(5000, 1), emb_size=emb_size)

anchor = Input((5000, 1), name='anchor')
positive = Input((5000, 1), name='positive')
negative = Input((5000, 1), name='negative')

a = shared_conv2(anchor)
p = shared_conv2(positive)
n = shared_conv2(negative)

pos_sim = Dot(axes=-1, normalize=True)([a, p])
neg_sim = Dot(axes=-1, normalize=True)([a, n])

loss = Lambda(cosine_triplet_loss, output_shape=(1,))([pos_sim, neg_sim])

model_triplet = Model(inputs=[anchor, positive, negative], outputs=loss)
print(model_triplet.summary())

opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model_triplet.compile(loss=identity_loss, optimizer=opt)

batch_size = batch_size_value
gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, None)
nb_epochs = number_epoch
csv_logger = CSVLogger(f'log/Training_Log_{description}.csv', append=True, separator=';')
for epoch in range(nb_epochs):
    print("built new hard generator for epoch " + str(epoch))
    model_triplet.fit_generator(generator=gen_hard.next_train(),
                                steps_per_epoch=Xa_train.shape[0] // batch_size,
                                epochs=1, verbose=1, callbacks=[csv_logger])
    gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, shared_conv2)
    # For no semi-hard_triplet
    # gen_hard = HardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, None)
shared_conv2.save(f'trained_model/{description}.h5')
