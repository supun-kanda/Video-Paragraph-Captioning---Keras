
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
from keras.layers import Dense, Activation, Input, GRU, Dropout
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.models import Model
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History

from Attention import Attention_Layer
from Multimodel_layer import Multimodel_Layer

import sys
sys.path.insert(0,'/mnt/data/Proposals/Activity_proposals/sparseprop/sparseprop') #This path contains the C3D feature reading code
from feature import C3D


# In[2]:

class Caption_Generator:
    def __init__(self,vid_ids,training_labels,train_epochs):
        self.captions = []
        self.captions_in_each_video = []
        self.word2id = {}
        self.id2word = {}
        self.max_sentence_length = 0
        self.vocabulary_size = 0
        self.batch_size = 10
        self.embedding_output_shape = 512
        self.json_file = "Data/train.json"
        self.csv_file = "Data/labels_dense.csv" #frame information csv since the json is from time
        self.pkl_file = "Data/no_keys.pkl"
        self.dic_file = "Data/dic_file.pkl"
        self.feature_file = '/mnt/data/c3d_features/feature.c3d.hdf5' #feature file is loaded for gere
        self.num_frames_out = 200 #number of frames per video on output feature pool
        self.num_features = 500 #number of features per frame 
        self.additional_save_path = "Data" # new data like processed features will be saved here
        self.vid_ids = vid_ids #video ids in caption dataset dictionary. Given as a list
        self.training_labels = training_labels #training json file
        self.size = 1500
        self.train_epochs = train_epochs
        
        #######################################
        if(os.path.exists(self.pkl_file)):
            with open(self.pkl_file,'rb') as f:
                removing_set = pickle.load(f)
        else:
            removing_set = self.save_no_keys()
        for e in removing_set:
            self.vid_ids.remove(e)
        print("vid_ids verified")
        #######################################
        
################################################################################################
    def save_no_keys(self):#Which creates a list of ids which is only presented in id file_ not in csv file
        no_keys = []
        cap_labs = []
        with open(self.csv_file) as csvfile:
            captions = csv.reader(csvfile, delimiter=',')
            stat = 1
            for row in captions:
                if(stat):
                    stat = 0
                else:
                    cap_labs.append(row[0])
        cap_labs = list(set(cap_labs))
                
        for key in self.vid_ids:
            if(key not in self.training_labels):
                print(key,"1")
                no_keys.append(key)
            elif(key not in cap_labs):
                print(key,"2")
                no_keys.append(key)

        with open(self.pkl_file, 'wb') as f:
            pickle.dump(no_keys, f)
        
        return no_keys
################################################################################################
    def video_features(filename, init, num_feats, total_feats):
        obj = C3D(filename=self.feature_file, t_stride=1,t_size=1)
        obj.open_instance()
        video = obj.read_feat(video_name=filename)
        m = video.shape[0]
        ratio = 1.0*m/total_feats
        init_n = int(ratio*init)
        nums_n = int(ratio*num_feats)
        features = obj.read_feat(video_name=filename, f_init=init_n, duration=nums_n)
        obj.close_instance()
        return features
        
################################################################################################
    def read_data(self,n_batch): 
        print("loading Data for new Batch... ")
        ##files = [] 
    
        #reading captions
        #with open('MLDS_HW2/MLDS_hw2_data/training_label.json') as data_file:
        #    self.training_labels = json.load(data_file)
        self.captions_in_each_video = []  

        for i in n_batch:
            try:
                for j in range(len(self.training_labels[self.vid_ids[i]]['sentences'])):
                    ######training_labels[i]['caption'][j]#####
                    self.training_labels[self.vid_ids[i]]['sentences'][j] = "<s> "+self.training_labels[self.vid_ids[i]]['sentences'][j]+" <e>"
                    self.captions.append(self.training_labels[self.vid_ids[i]]['sentences'][j].lower().split(' '))
                self.captions_in_each_video.append(len(self.training_labels[self.vid_ids[i]]['sentences']))
            except KeyError:
                print("\tError Caption: %s"%self.vid_ids[i])
        
        #reading video features
        
        video_features = np.zeros((len(n_batch),self.num_frames_out,self.num_features))
        for i in range(len(n_batch)):
            obj = C3D(filename=self.feature_file, t_stride=1,t_size=5)
            obj.open_instance()
            video = obj.read_feat(video_name=self.vid_ids[n_batch[i]])
            m,n = video.shape
            #print("%s:%dx%d"%(self.vid_ids[n_batch[i]],m,n))
            if(m>=self.num_frames_out):
                video_features[i] = video[:self.num_frames_out,:]
            else:
                video_features[i,:m,:] = video
                video_features[i,m:,:].fill(0)
            obj.close_instance()
        #np.save("%s/processed_features"%self.additional_save_path, video_features, allow_pickle=True)
        ##return video_features
        return video_features
####################################################################################################
    def create_vocabulary(self):

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
        save_dic = {'dic':self.id2word, 'max_len':self.max_sentence_length, 'voca':self.vocabulary_size}
        with open(self.dic_file, 'wb') as f:
            pickle.dump(save_dic, f)
################################################################################################
    def transform_inputs(self, video_features):
        #transforming the no of samples of video features equal to no of samples of captions
        #new_features = np.zeros((len(self.captions), 80, 4096))
        new_features = np.zeros((len(self.captions), self.num_frames_out, self.num_features))
        ######################################
        
        last = 0
        for i in range(len(self.captions_in_each_video)):
            num_caps = self.captions_in_each_video[i]
            for j in range(last,last+num_caps):
                new_features[j] = video_features[i]
            last = last+num_caps       
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
        return output_array
    
################################################################################################
    def data_preprocessing(self, n_batch):
        #########################Preprocessing Data##############################
        #print("Data Preprocessing.......")
        #print("\tReading data.......")
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
        
        return video_features, embedded_input, label_tensor
        
    ################################################################################################    
    def build_model(self, video_features, embedded_input):
        #########################training model##################################
        print('Building Sentence Generator Model...')

        input1 = Input(shape=(embedded_input.shape[1],embedded_input.shape[2]), dtype='float32')
        #input2 = Input(shape=(visual_features.shape[0],visual_features.shape[1]), dtype='float32')
        input2 = Input(shape=(video_features.shape[1], video_features.shape[2]), dtype='float32')
        
        model = Sequential()
        
        layer1 = GRU(512, return_sequences = True, input_shape = (embedded_input.shape[1],embedded_input.shape[2]), activation = 'relu')(input1)
        
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
        model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr=0.001), metrics=['accuracy'])

        model.summary() # Convenient function to see details about the network model.

        return model
    
    ################################################################################################    
    def train(self):       
        '''
        batches = np.arange(1450)
        #########################training model##################################
        for epoch in range(10):
            print("\n\n\nEpoch : ",epoch+1)
            np.random.shuffle(batches)
            batch = 0
            for iteration in range(int(1450/self.batch_size)):
                if batch+self.batch_size >= 1450:
                    n_batch = batches[batch:-1]
                else:    
                    n_batch = batches[batch:(batch+self.batch_size)]
                batch += self.batch_size
                self.captions = []
                video_features, embedded_input, label_tensor = self.data_preprocessing(n_batch)
                if(iteration == 0 and epoch == 0):
                    #model = caption_generator.build_model(video_features, embedded_input)
                    model = self.build_model(video_features, embedded_input)
                # define the checkpoint
                filepath="Data/model_results/word-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
                checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
                callbacks_list = [checkpoint]

                print("\n\n###########Training the model on epoch : ", epoch+1, " batch : ", iteration+1 ," ###########\n\n")
                model.fit(x = [embedded_input,video_features], y = label_tensor, batch_size = 256, epochs= 5, callbacks = callbacks_list)
            #self.save_model(model,epoch)
         '''
        video_features, embedded_input, label_tensor = self.data_preprocessing(np.arange(self.size));
        model = self.build_model(video_features, embedded_input)
        filepath="Data/model_results/word-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        #history = LossHistory()
        #callbacks_list.append(history)
        hist = model.fit(x = [embedded_input,video_features], y = label_tensor, validation_split = 0.3, batch_size = 50, epochs= self.train_epochs, callbacks = callbacks_list)
        with open("Data/histfile.pkl", 'wb') as f:
            pickle.dump(hist.history, f)
        self.save_model(model,1)
        #print(history.history.keys())
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
    def test(self):   
        #######################################
        if(len(self.id2word) and self.max_sentence_length and self.vocabulary_size):
            print("Use Previous values")
        else:
            with open(self.dic_file,'rb') as f:
                save_dic = pickle.load(f)
                self.id2word = save_dic['dic']
                self.max_sentence_length = save_dic['max_len']
                self.vocabulary_size = save_dic['voca']
        #######################################
        #print("word : ",self.id2word[0])
        test_captions = []
        files = []
        captions_in_each_video = []
        
        for i in range(1500,2000):
            try:
                files.append(self.vid_ids[i])
                for j in range(len(self.training_labels[self.vid_ids[i]]['sentences'])):
                    ######training_labels[i]['caption'][j]#####
                    test_captions.append(self.training_labels[self.vid_ids[i]]['sentences'][j].lower().split(' '))
                captions_in_each_video.append(j)
            except KeyError:
                print("\tError Caption: %s"%self.vid_ids[i])
        
        #reading video features
        encoded_tensor = np.zeros((len(test_captions), self.max_sentence_length, self.vocabulary_size), dtype=np.float16)
        label_tensor = np.zeros((len(test_captions), self.max_sentence_length, self.vocabulary_size), dtype=np.float16)
        for i in range(len(test_captions)):
            for j in range(len(test_captions[i])):
                if(test_captions[i][j] not in self.word2id):
                    continue
                else:
                    print('excaped')
                encoded_tensor[i, j, self.word2id[test_captions[i][j]]] = 1 #convert each vector into to index
                if j<len(self.captions[i])-1:
                    label_tensor[i,j,self.word2id[test_captions[i][j+1]]] = 1
        encoded_tensor[:,0,0] = 1
                
        print("number of files : ",len(files))
        video_features = np.zeros((len(files),self.num_frames_out,self.num_features))
        new_features = np.zeros((len(test_captions),self.num_frames_out,self.num_features))
        
        
        for i in range(len(files)):
            obj = C3D(filename=self.feature_file, t_stride=1,t_size=5)
            obj.open_instance()
            video = obj.read_feat(video_name=files[i])
            m,n = video.shape
            #print("%s:%dx%d"%(self.vid_ids[n_batch[i]],m,n))
            if(m>=self.num_frames_out):
                video_features[i] = video[:self.num_frames_out,:]
            else:
                video_features[i,:m,:] = video
                video_features[i,m:,:].fill(0)
            obj.close_instance()
        
        ###################################################
        last = 0
        for i in range(len(captions_in_each_video)):
            num_caps = captions_in_each_video[i]
            for j in range(last,last+num_caps):
                new_features[j] = video_features[i]
            last = last+num_caps       
        ###################################################
        
        #new_features = np.reshape(new_features, (len(self.captions)*self.num_frames_out, 1, self.num_features))
        #new_features = self.transform_inputs(video_features)
        #new_features = np.reshape(new_features, (test_captions*self.num_frames_out, 1, self.num_features))
        #ncoded_tensor = np.repeat(encoded_tensor, self.num_frames_out, axis=0)
        embedded_input = self.embedding_layer(encoded_tensor)
        print("embedded_input : ", embedded_input.shape)
        print("video_features : ", new_features.shape)
        model  = self.build_model(new_features, embedded_input)
        model = self.load_model(model,1)
        print(embedded_input.shape,new_features.shape,label_tensor.shape)
        #output = model.evaluate(x = [embedded_input, new_features], y = label_tensor, batch_size=10, verbose = 1)
        output2 = model.predict(x = [embedded_input, new_features], batch_size=10, verbose = 1)
        #print(output.shape,output2.shape)
        '''
        new_features, embedded_input, label_tensor = self.data_preprocessing(np.arange(1500,2000));
        model  = self.build_model(new_features, embedded_input)
        model = self.load_model(model,1)
        #output = model.evaluate(x = [embedded_input, new_features], y = label_tensor, batch_size=10, verbose = 1)
        output = model.predict(x = [embedded_input, new_features], batch_size=10, verbose = 0)
        return output
        '''
        return output2
##############################################################################################################################################        