# Helper Files and classes for SoundTransfer

import numpy as np
import resampy
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import mel_features
import vggish_params
import glob
import os
import pickle
import random
from collections import Counter

label = {
    'dog-bark':0,
    'drill':1,
    'hazard-alarm':2,
    'phone-ring':3,
    'speech':4,
    'vacuum':5,
    'baby-cry':6,
    'chopping':7,
    'cough':8,
    'door':9,
    'water-running':10,
    'knock':11,
    'microwave':12,
    'shaver':13,
    'toothbrush':14,
    'blender':15,
    'dishwasher':16,
    'doorbell':17,
    'flush':18,
    'hair-dryer':19,
    'laugh':20,
    'snore':21,
    'typing':22,
    'hammer':23,
    'car-horn':24,
    'engine':25,
    'saw':26,
    'cat-meow':27,
    'alarm-clock':28,
    'cooking':29,
    'dribble': 30,  # SpokeSense
    'whistle': 31,  # SpokeSense
    'buzzer': 32    # SpokeSense
}

label_inv = {i:k for i,k in enumerate(list(label.keys()))}

def waveform_to_examples(data, sample_rate):
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(log_mel, window_length=example_window_length, hop_length=example_hop_length)
    return log_mel_examples

def wavfile_to_examples(wav_file):
    sr, wav_data = wavfile.read(wav_file)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr)
    

class FeatureCollection():
    # Associate Features for a particular WAV file
    def __init__(self, event, wavpath, split=5):
        self.label = event
        self.x = wavfile_to_examples(wavpath)
        self.y = [event] * self.x.shape[0]
        self.filename = wavpath
        self.indices = []
        
    # Split feature list into five-second chunks
    def split_into(self, sections):
        self.indices = []
        num_sections = 0
        if (sections > 0):
            num_sections = int(self.x.shape[0] / sections)
        if (num_sections==0):
            self.indices.append(np.array(range(self.x.shape[0])))
        else:
            for k in range(num_sections):
                self.indices.append(np.array(range(sections*k, min(sections*(k+1),self.x.shape[0]))))
        
class Contextualizer():
    
    def __init__(self, context, other):
        self.context = context  # Context mapping
        self.other = other #Whether to include other class or not
        self.features = []
        self.weights = [1.0] * len(label)
        self.labels = {self.context[k]: k for k in range(len(self.context))}
        self.labels[len(self.context)] = 'other'
        self.negative_class_limit = 100
        
    def label_for(self, index):
        res =  self.context[index] if index < len(self.context) else 'other'
        return res
        
    def index_for(self, label):
        res = self.labels[label] if label in self.labels.keys() else -1
        return res
        
    def compute_weights(self):
        # context ['water-running', 'flush', 'toothbrush', 'shaver', 'hair-dryer', 'other']
        # labels {'water-running': 0, 'flush': 1, 'toothbrush': 2, 'shaver': 3, 'hair-dryer': 4, 'other': 5, 6: 'other'}
        self.weights = [1.0] * len(label)
        for k,v in label.items():
            if (k not in self.context):
                self.weights[v] = 0
    
    def extract_from(self, pipeline, split=5):
        self.features = []
        for c in self.context:
            print("Context", c)
            if (c in pipeline.labels_to_index):
                for index in pipeline.labels_to_index[c]:
                    pipeline.features[index].split_into(split)
                    self.features.append(pipeline.features[index])

        if(self.other):
            other_class = []
            for c in pipeline.labels_to_index:
                if (c not in self.context):
                    k = 0
                    for index in pipeline.labels_to_index[c]:
                        pipeline.features[index].split_into(split)
                        pipeline.features[index].label = 'other'
                        other_class.append(pipeline.features[index])
            
            # Shuffle Other Class
            random.seed(442)
            random.shuffle(other_class)
            other_class = other_class[:self.negative_class_limit]
            
            # Append
            self.features = self.features + other_class
        print("Extraction complete. Total: %d (from %d)" % (len(self.features), len(pipeline.features)))
        
    def test_accuracy(self, model):
        N = 0
        acc = 0.0
        target = []
        prediction = []
        files = []
        confidence = []
        for f in self.features:
            for chunk in f.indices:
                if (len(chunk)>0):
                    x = f.x[chunk]
                    x = x.reshape(len(x), 96, 64, 1)
                    # Test Model Here
                    y = model.predict(x)
                    y_avg = np.argmax(np.sum(y,axis=0))
                    #conf = (np.sum(y,axis=0) / float(len(x))).tolist()
                    #conf = [ (self.context[k],conf[k]) for k in range(len(conf)) ]
                    conf = np.sum(y,axis=0) / float(len(x))
                    target.append(f.label)
                    prediction.append(self.context[y_avg])
                    files.append(f.filename)
                    confidence.append(conf)
        return np.array(target), np.array(prediction), files, np.array(confidence)
        
    def test_accuracy_meta(self, model, thres=0.1):
        N = 0
        acc = 0.0
        target = []
        prediction = []
        files = []
        confidence = []
        other_avg = []
        mapper = [k for k in label.keys()]
        for f in self.features:
            for chunk in f.indices:
                if (len(chunk)>0):
                    x = f.x[chunk]
                    x = x.reshape(len(x), 96, 64, 1)
                    
                    # Test Model Here
                    y = model.predict(x)
                    
                    W = np.ones(y.shape) * self.weights
                    y = y * W

                    y_avg = np.argmax(np.sum(y,axis=0))
                    conf = np.sum(y,axis=0) / float(len(x))
                    
                    best_conf = np.argmax(conf, axis=0)
                    target.append(f.label)
                    
                    final_prediction = mapper[y_avg] if conf[best_conf] > thres else 'other'
                    prediction.append(final_prediction)
                    
                    files.append(f.filename)
                    confidence.append(conf)
                    if (final_prediction=='other'):
                        other_avg.append(conf[best_conf])

        return np.array(target), np.array(prediction), files, np.array(confidence)
        
    def test_accuracy_thres(self, model, thres=0.1):
        N = 0
        acc = 0.0
        target = []
        prediction = []
        files = []
        confidence = []
        other_avg = []
        mapper = [k for k in label.keys()]
        for f in self.features:
            for chunk in f.indices:
                if (len(chunk)>0):
                    x = f.x[chunk]
                    x = x.reshape(len(x), 96, 64, 1)
                    
                    # Test Model Here
                    y = model.predict(x)
                    
                    y_avg = np.argmax(np.sum(y,axis=0))
                    conf = np.sum(y,axis=0) / float(len(x))
                    
                    best_conf = np.argmax(conf, axis=0)
                    target.append(f.label)
                    
                    final_prediction = mapper[y_avg] if conf[best_conf] > thres else 'other'
                    prediction.append(final_prediction)
                    
                    files.append(f.filename)
                    confidence.append(conf)
                    if (final_prediction=='other'):
                        other_avg.append(conf[best_conf])

        return np.array(target), np.array(prediction), files, np.array(confidence)
    
    # Update
    def test_voting_accuracy(self, model, thres=0):
        N = 0
        acc = 0.0
        target = []
        prediction = []
        files = []
        confidence = []
        for f in self.features:
            for chunk in f.indices:
                if (len(chunk)>0):
                    x = f.x[chunk]
                    x = x.reshape(len(x), 96, 64, 1)
                    # Test Model Here
                    y = model.predict(x)
                    
                    N = len(x)
                    votes = []
                    for k in range(N):
                        p = y[k,:]
                        # Send Prediction to Server
                        m = np.argmax(p)
                        if (m < len(self.context)):
                            votes.append((self.context[m], p[m]))
                        else:
                            print("KeyError: %s" % m)

                    v = []
                    for pred in votes:
                        # print("Prediction: %s (%0.4f)" % (p[0],p[1]))
                        if (pred[1] > thres):
                            v.append(pred[0])
                    res = Counter(v)
                    conf = np.sum(y,axis=0) / float(len(x))
                    target.append(f.label)
                    prediction.append(res.most_common(1)[0])
                    files.append(f.filename)
                    confidence.append(conf)

        return np.array(target), np.array(prediction), files, np.array(confidence)
                        
class TestingPipeline():
    
    def __init__(self, root_dir):
        global label
        self.root_dir = root_dir
        self.features = []
        self.labels = []
        self.label_mapping = label
        self.labels_to_index = dict()
        self.MAX_COUNT_LEN = 20000
        
    # Go through each directory, and create FeatureCollections per file
    def assemble(self, sub_dirs):
        index_tracker = 0
        for l, sub_dir in enumerate(sub_dirs):
            class_count = 0
            for fn in glob.glob(os.path.join(self.root_dir, sub_dir, "*.wav")):
                print("Processing file",fn)
                try:
                    feature_collection = FeatureCollection(sub_dir, fn)
                    self.features.append(feature_collection)
                    
                    if (sub_dir not in self.labels_to_index):
                        self.labels_to_index[sub_dir] = []
                        
                    # Save the key, we'll use it later
                    self.labels_to_index[sub_dir].append(index_tracker)
                    index_tracker = index_tracker + 1
                    
                    # Increment count
                    class_count  = class_count + feature_collection.x.shape[0]
                except:
                    print("Oooops! Meta-data on WAV file invalid.")
                
                #if (class_count > self.MAX_COUNT_LEN):
                #    break
                        
                                
    def save(self, filepath):
        print("Saving as: '%s'" % filepath)
        pickle.dump( self, open(filepath, "wb" ) )
        print("Saved.")