
# coding: utf-8

# In[1]:
import tensorflow as tf
import numpy as np
import json
import os
import pickle
import csv

from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, GRU, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.models import Model
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History

from Attention import Attention_Layer
from Multimodel_layer import Multimodel_Layer
from Utilities import Utilities


# In[2]:

class Caption_Generator(Utilities):
    def __init__(self,vid_ids,training_labels,train_epochs):#constructor
        self.captions = []
        self.captions_in_each_video = []
        self.word2id = {}
        self.id2word = {}
        self.max_sentence_length = 0
        self.vocabulary_size = 0
        self.batch_size = 10
        self.embedding_output_shape = 512
        self.json_file = "Data/train.json"
        self.csv_file = 'Data/labels_compact.csv'
        self.pkl_file = "Data/ids_no_keys.pkl" # include id dictionary to row index  of csv file and keys not in csv but in json file
        self.dic_file = "Data/dic_file.pkl" #included id2word max_sentence_length vocabulary_size dictionaries
        self.feature_file = '/mnt/data/c3d_features/feature.c3d.hdf5' #feature file is loaded for gere
        self.num_frames_out = 200 #number of frames per video on output feature pool
        self.num_features = 500 #number of features per frame 
        self.additional_save_path = "Data" # new data like processed features will be saved here
        self.vid_ids = vid_ids #video ids in caption dataset dictionary. Given as a list
        self.training_labels = training_labels #training json file
        self.size = 2000
        self.train_epochs = train_epochs
        
        #######################################
        if(os.path.exists(self.pkl_file)):
            with open(self.pkl_file,'rb') as f:
                ids_no_keys = pickle.load(f)
                removing_set = ids_no_keys['no_keys'] 
                self.frame_dic = ids_no_keys['frame_dic']
        else:
            removing_set, self.frame_dic = self.save_no_keys()
            print("Creating pikle for csv dic and ids not existed")
        for e in removing_set:
            self.vid_ids.remove(e)
        print("vid_ids verified")
        #######################################
        
###############################################################################################
    def read_data(self,n_batch): 
        print("loading Data for new Batch... ")
        
        self.captions_in_each_video = []  

        for i in n_batch:
            try:
                for j in range(len(self.training_labels[self.vid_ids[i]]['sentences'])):
                    self.training_labels[self.vid_ids[i]]['sentences'][j] = "<s> "+self.training_labels[self.vid_ids[i]]['sentences'][j]+" <e>"
                    self.captions.append(self.training_labels[self.vid_ids[i]]['sentences'][j].lower().split(' '))
                self.captions_in_each_video.append(len(self.training_labels[self.vid_ids[i]]['sentences']))
            except KeyError:
                print("\tError Caption: %s"%self.vid_ids[i])
        
        #reading video features
        
        video_features = np.zeros((len(n_batch),self.num_frames_out,self.num_features))
        for i in range(len(n_batch)):
            frame_list = self.frame_dic[self.vid_ids[n_batch[i]]]
            init = frame_list[0]
            num_frames = frame_list[1]
            vid_frames = frame_list[2]
            video_features[i] = self.extract_video_features(file = self.vid_ids[n_batch[i]], init = init, num_frames = num_frames, vid_frames = vid_frames, frame_limit = 200)
        print("Done loading Data")
        return video_features
####################################################################################################
    def create_vocabulary(self):#dumping a dic

        print("creating vocabulary...")
        labels = []
        for i in self.vid_ids:
            try:
                for j in range(len(self.training_labels[i]['sentences'])):
                    self.training_labels[i]['sentences'][j] = "<s> "+self.training_labels[i]['sentences'][j]+" <e>" 
                    labels.append(self.training_labels[i]['sentences'][j].lower().split(' '))
            except KeyError:
                print("\tKey Error:%s"%i)
        self.max_sentence_length = 1 + max([len(caption) for caption in labels])
        print("Max sentence length : ", self.max_sentence_length)
         
        #computing char2id and id2char vocabulary
        index = 0
        for caption in labels:
            for word in caption:
                if word not in self.word2id:
                    self.word2id[word] = index
                    self.id2word[index] = word
                    index += 1
        self.vocabulary_size = len(self.word2id)
        print("Vocabulary Size %d"%(self.vocabulary_size)) 
        print("Done creating vocabulary")
        #save_dic = {'dic':self.id2word, 'max_len':self.max_sentence_length, 'voca':self.vocabulary_size}
        #with open(self.dic_file, 'wb') as f:
        #    pickle.dump(save_dic, f)
################################################################################################
    def transform_inputs(self, video_features):
        print("Transforming Inputs...")
        #transforming the no of samples of video features equal to no of samples of captions
        #new_features = np.zeros((len(self.captions), 80, 4096))
        new_features = np.zeros((len(self.captions), self.num_frames_out, self.num_features))
        ######################################
        
        last = 0
        for i in range(len(self.captions_in_each_video)):
            num_caps = self.captions_in_each_video[i]
            for j in range(last,last+num_caps-1):
                new_features[j] = video_features[i]
            last = last+num_caps       
        print("Done Transforming Inputs...")
        return new_features
            
    
################################################################################################
    def one_of_N_encoding(self): 
        print("encoding inputs...")      
        #creating caption tensor that is a matrix of size numCaptions x maximumSentenceLength x wordVocabularySize
        encoded_tensor = np.zeros((len(self.captions), self.max_sentence_length, self.vocabulary_size), dtype=np.float16)
        label_tensor = np.zeros((len(self.captions), self.max_sentence_length, self.vocabulary_size), dtype =np.float16)
        #one-hot-encoding
        for i in range(len(self.captions)):
            for j in range(len(self.captions[i])):
                encoded_tensor[i, j, self.word2id[self.captions[i][j]]] = 1 #convert each vector into to index
                if j<len(self.captions[i])-1:
                    label_tensor[i,j,self.word2id[self.captions[i][j+1]]] = 1
        print("Done encoding inputs...")
        return encoded_tensor, label_tensor
    
################################################################################################
    def embedding_layer(self, input_data):
        print("embedding inputs....")
        model = Sequential()
        model.add(Dense(self.embedding_output_shape, input_shape = (self.max_sentence_length, self.vocabulary_size)))
        model.add(Activation('relu'))
        model.compile('rmsprop','mse')
        embedding_weights = model.get_weights()
        output_array = model.predict(input_data)
        self.embedding_weights = model.get_weights()
        output_weights = np.asarray(self.embedding_weights[0]).T
        self.embedding_weights[0] = output_weights
        self.embedding_weights[1] = np.ones((self.vocabulary_size,))
        print("Done embedding inputs....")
        #for i in range(output_array.shape[0]):
        #    if(i%200):
        #        print("max: ",np.max(output_array[i]),"min: ",np.min(output_array[i]), "embed")
        return output_array
    
################################################################################################
    def data_preprocessing(self, n_batch):
        #########################Preprocessing Data##############################
        #print("Data Preprocessing.......")
        #print("\tReading data.......")
        print("Data Preprocessing....")
        video_features = self.read_data(n_batch)
        video_features = self.transform_inputs(video_features)
        #print("\tvideo features : ",video_features.shape)
        #print("\tCaptions : ", len(self.captions))
        #print("\tCreating Vocabulary......")
        self.create_vocabulary()

        # one-hot encoding of captions
        #print("\tEncoding Captions......")
        encoded_tensor, label_tensor = self.one_of_N_encoding()
        #print("\tEncoded Captions : ",encoded_tensor.shape)

        # embedding the one-hot encoding of each word into 512
        #print("\tEmbedding Captions.......")
        embedded_input = self.embedding_layer(encoded_tensor)

        #print("\tEmbedding Weights : ", np.asarray(self.embedding_weights[0]).shape)

        #print("\tEmbedded_captions : ",embedded_input.shape)
        print("Done Data Preprocessing....")
        return video_features, embedded_input, label_tensor
        
    ################################################################################################    
    def build_model(self, video_features, embedded_input):
        #########################training model##################################
        print('Building Sentence Generator Model...')

        input1 = Input(shape=(embedded_input.shape[1],embedded_input.shape[2]), dtype='float32')
        #input2 = Input(shape=(visual_features.shape[0],visual_features.shape[1]), dtype='float32')
        input2 = Input(shape=(video_features.shape[1], video_features.shape[2]), dtype='float32')
        model = Sequential()
        
        #layer1 = GRU(512, return_sequences = True, input_shape = (embedded_input.shape[1],embedded_input.shape[2]), activation = 'relu')(input1)
        layer1 = LSTM(512, return_sequences = True, input_shape = (embedded_input.shape[1],embedded_input.shape[2]), activation = 'relu')(input1)
        
        attention_layer = Attention_Layer(output_dim = 32)([layer1, input2])

        multimodel_layer = Multimodel_Layer(output_dim = 1024)([layer1,attention_layer])

        dropout = Dropout(0.5)(multimodel_layer)

        layer2 = TimeDistributed(Dense(activation = 'tanh', units = 512))(dropout)

        softmax_layer = Dense(units = self.vocabulary_size, activation = 'softmax', weights = self.embedding_weights)(layer2)
        
        
        model = Model(inputs = [input1, input2], outputs = [softmax_layer])
        
        '''
        # We also specify here the optimization we will use, in this case we use RMSprop with learning rate 0.001.
        # RMSprop is commonly used for RNNs instead of regular SGD.
        # categorical_crossentropy is the same loss used for classification problems using softmax. (nn.ClassNLLCriterion)
        '''
        #model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr=0.001), metrics=['accuracy'])
        model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr=0.001), metrics=['accuracy'])

        model.summary() # Convenient function to see details about the network model.
        print('Done Building...')
        return model
    
    ################################################################################################    
    def train(self):       
        print('Start Training...')
        video_features, embedded_input, label_tensor = self.data_preprocessing(np.arange(self.size));
        model = self.build_model(video_features, embedded_input)
        filepath="Data/model_results/word-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        hist = model.fit(x = [embedded_input,video_features], y = label_tensor, batch_size = 20, epochs= self.train_epochs, callbacks = callbacks_list)
        model.trainable = False
        with open("Data/histfile.pkl", 'wb') as f:
            pickle.dump(hist.history, f)
        self.save_model(model,1)
        print('Done Training...')
        return model,hist
    
    ################################################################################################    
    def save_model(self, model, epoch):
        # serialize model to JSON
        filename = "Data/model_results/model_epoch_"+str(epoch)+".h5"
        #with open("batch_model.json", "w") as json_file:
            #json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(filename)
        print("Saved model to disk")
        
    ################################################################################################    
    def load_model(self, model, epoch):
        # load weights into new model
        filename = "Data/model_results/model_epoch_"+str(epoch)+".h5"
        model.load_weights(filename)
        print("Loaded model from disk")
        return model
    
    ################################################################################################    
    #def test(self, model, epoch):
##############################################################################################################################################
    def test(self, model):   
        test_captions = []
        files = []
        self.captions_in_each_video = []
        
        for i in range(2600,3000):
            try:
                files.append(self.vid_ids[i])
                for j in range(len(self.training_labels[self.vid_ids[i]]['sentences'])):
                    test_captions.append(self.training_labels[self.vid_ids[i]]['sentences'][j].lower().split(' '))
                self.captions_in_each_video.append(j)
            except KeyError:
                print("\tError Caption: %s"%self.vid_ids[i])
        
        #reading video features
        encoded_tensor = np.zeros((len(test_captions), self.max_sentence_length, self.vocabulary_size), dtype=np.float16)
        encoded_tensor[:,0,0] = 1
                
        print("number of files : ",len(files))
        video_features = np.zeros((len(files),self.num_frames_out,self.num_features))
        new_features = np.zeros((len(test_captions),self.num_frames_out,self.num_features))
        
        
        for i in range(len(files)):
            frame_list = self.frame_dic[files[i]]
            init = frame_list[0]
            num_frames = frame_list[1]
            vid_frames = frame_list[2]
            video_features[i] = self.extract_video_features(file = files[i] , init = init, num_frames = num_frames, vid_frames = vid_frames, frame_limit = 200)
        
        ###################################################
        last = 0
        for i in range(len(self.captions_in_each_video)):
            num_caps = self.captions_in_each_video[i]
            for j in range(last,last+num_caps):
                new_features[j] = video_features[i]
            last = last+num_caps       
        ###################################################
        
        embedded_input = self.embedding_layer(encoded_tensor)
        print("embedded_input : ", embedded_input.shape)
        print("video_features : ", new_features.shape)
        #model = self.load_model(model,1)
        #model  = self.build_model(new_features, embedded_input)
        #model = self.load_model(model,1)
        output = model.predict(x = [embedded_input, new_features])
        
        
        ms, sents, words = output.shape
        
        for i in range(20):
            string = "%d:\n\t"%i
            for j in range(output[i].shape[0]):
                try:
                    idWord = np.argmax(output[i,j,:])
                    word = self.id2word[idWord]
                    string += "%s "%word
                    if(word == "<e>"):
                        break
                except KeyError:
                    string += "xxx \n\n"
            print(string)
        
        return output
##############################################################################################################################################        
    def combine(self):
        print('Start Training...')
        video_features, embedded_input, label_tensor = self.data_preprocessing(np.arange(self.size));
        model = self.build_model(video_features, embedded_input)
        filepath="Data/model_results/word-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        hist = model.fit(x = [embedded_input,video_features], y = label_tensor, batch_size = 20, epochs= self.train_epochs, callbacks = callbacks_list)
        model.trainable = False
        with open("Data/histfile.pkl", 'wb') as f:
            pickle.dump(hist.history, f)
        self.save_model(model,1)
        print('Done Training...')
        test_captions = []
        files = []
        self.captions_in_each_video = []
        
        for i in range(2600,3000):
            try:
                files.append(self.vid_ids[i])
                for j in range(len(self.training_labels[self.vid_ids[i]]['sentences'])):
                    test_captions.append(self.training_labels[self.vid_ids[i]]['sentences'][j].lower().split(' '))
                self.captions_in_each_video.append(j)
            except KeyError:
                print("\tError Caption: %s"%self.vid_ids[i])
        
        #reading video features
        encoded_tensor = np.zeros((len(test_captions), self.max_sentence_length, self.vocabulary_size), dtype=np.float16)
        encoded_tensor[:,0,0] = 1
                
        print("number of files : ",len(files))
        video_features = np.zeros((len(files),self.num_frames_out,self.num_features))
        new_features = np.zeros((len(test_captions),self.num_frames_out,self.num_features))
        
        
        for i in range(len(files)):
            frame_list = self.frame_dic[files[i]]
            init = frame_list[0]
            num_frames = frame_list[1]
            vid_frames = frame_list[2]
            video_features[i] = self.extract_video_features(file = files[i] , init = init, num_frames = num_frames, vid_frames = vid_frames, frame_limit = 200)
        
        ###################################################
        last = 0
        for i in range(len(self.captions_in_each_video)):
            num_caps = self.captions_in_each_video[i]
            for j in range(last,last+num_caps):
                new_features[j] = video_features[i]
            last = last+num_caps       
        ###################################################
        
        embedded_input = self.embedding_layer(encoded_tensor)
        print("embedded_input : ", embedded_input.shape)
        print("video_features : ", new_features.shape)
        #model = self.load_model(model,1)
        #model  = self.build_model(new_features, embedded_input)
        #model = self.load_model(model,1)
        output = model.predict(x = [embedded_input, new_features], verbose = 1)
        
        
        ms, sents, words = output.shape
        
        for i in range(20):
            string = "%d:\n\t"%i
            for j in range(output[i].shape[0]):
                try:
                    idWord = np.argmax(output[i,j,:])
                    word = self.id2word[idWord]
                    string += "%s "%word
                    if(word == "<e>"):
                        break
                except KeyError:
                    string += "xxx \n\n"
            print(string)
        
        return output