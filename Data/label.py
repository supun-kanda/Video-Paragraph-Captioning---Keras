import json
import os
import csv
import cv2

vid_path = ''
vid_file_names = ''
data_json = ''
data_ids_json = ''
csv_file_name = ''

with open(data_json) as f:
	data = json.load(f)
with open(data_ids_json) as f:
	keys = json.load(f)
file_list = os.listdir(vid_path)
csv_file = open(csv_file_name,"w")
keys_present = []

fieldnames = ['video-name','f-init','n-frames','video-frames','sentences']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()
os.chdir(vid_path)
for line in file_list:
	key = line.split(".")[0]
	count = 0
	if(key in data):
		keys_present.append(key)
		vid = cv2.VideoCapture(line.split("\n")[0])
		vid_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
		FPS = vid.get(cv2.CAP_PROP_FPS)
		time_list = data[key]['timestamps'] 
		for segment in time_list:
			start_tim = segment[0]
			end_tim = segment[1]
			f_init = int(FPS * start_tim)
			n_frames = int(FPS * (end_tim -start_tim))
			writer.writerow({'video-name': "v_%s"%key, 'f-init':f_init, 'n-frames':n_frames, 'video-frames': vid_frames, 'sentences':data[key]['sentences'][count]})
			count+=1 
	else:
		print('No key:%s'%key)
print("Also")
for key in keys:
	if(key not in keys_present):
		print("No key:%s"key)
File.close()
csv_file.close()

