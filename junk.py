with open(self.csv_file) as csvfile:
            captions = csv.reader(csvfile, delimiter=',')
            stat = 1
            for row in captions:
                if(stat):
                    stat = 0
                else:
                    sentence = row[4]
                    vid_id = row[0]
                    n_frames = int(row[2])
                    init = row[1]
                    video_features = np.zeros((len(n_batch),self.num_frames_out,self.num_features))    