import sys
import numpy as np
sys.path.insert(0,'/mnt/data/Proposals/Activity_proposals/sparseprop/sparseprop') #This path contains the C3D feature reading code
from feature import C3D
class Utilities:
    
    def save_no_keys(self):#Which creates a list of ids which is only presented in id file not in csv file
        no_keys = []
        cap_labs = []
        frame_dic = {}
        save_data = {}
        with open(self.csv_file) as csvfile:
            captions = csv.reader(csvfile, delimiter=',')
            stat = 1
            for row in captions:
                if(stat):
                    stat = 0
                else:
                    cap_labs.append(row[0])
                    frame_dic[row[0]] = [int(row[1]), int(row[2]), int(row[3])] 
        cap_labs = list(set(cap_labs))
                
        for key in self.vid_ids:
            if(key not in self.training_labels):
                print(key,"1")
                no_keys.append(key)
            elif(key not in cap_labs):
                print(key,"2")
                no_keys.append(key)
        
        save_data['no_keys'] = no_keys
        save_data['frame_dic'] = frame_dic

        with open(self.pkl_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        return no_keys, frame_dic
    
    def extract_video_features(self, file, init, num_frames, vid_frames, frame_limit):
        video_features = np.zeros((self.num_frames_out,self.num_features))
        obj = C3D(filename=self.feature_file, t_stride=1, t_size=5)
        obj.open_instance()
        video = obj.read_feat(video_name=file)
        m = video.shape[0]
        ratio = 1.0*m/vid_frames
        init_n = int(ratio*init)
        if(num_frames>=vid_frames):
            nums_n = m - init_n
        else:
            nums_n = int(ratio*num_frames)
        features = obj.read_feat(video_name=file, f_init=init_n, duration=nums_n)
        obj.close_instance()
        if(m>=frame_limit):
            video_features = video[:frame_limit,:]
        else:
            video_features[:m,:] = video
            video_features[m:,:].fill(0)
        
        return video_features
 