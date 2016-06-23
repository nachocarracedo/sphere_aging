#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import math
import os
import collections
#import missingno as msno

if __name__ == "__main__":

	# function that gets time elapsed in row
	def total_time(row):
		return row['end'] - row['start']

	z=0 # counts number of sequences
	train_sequences = os.listdir(".\\public_data\\train")#get all sequences

	#for each sequence    
	for seq in train_sequences:
		#load csv
		targets_df = pd.read_csv('.\\public_data\\train\\'+seq+'\\targets.csv')
		pirsensor_df = pd.read_csv('.\\public_data\\train\\'+seq+'\\pir.csv')
		acceleration_df = pd.read_csv('.\\public_data\\train\\'+seq+'\\acceleration.csv')
		vhallway_df = pd.read_csv('.\\public_data\\train\\'+seq+'\\video_hallway.csv')
		vkitchen_df = pd.read_csv('.\\public_data\\train\\'+seq+'\\video_kitchen.csv')
		vlivingroom_df = pd.read_csv('.\\public_data\\train\\'+seq+'\\video_living_room.csv')
		#drop NaN rows in target (we need them for supervised learning)
		targets_df.dropna(axis=0, inplace=True)
		
		
		#MOVEMENT ROOM SENSORS
		pirsensor_df['total_time'] = pirsensor_df.apply(total_time, axis=1)
		pirsensor_df['start_new_int'] = pirsensor_df['start'].map(lambda x: math.modf(x)[1])
		pirsensor_df['end_new_int'] = pirsensor_df['end'].map(lambda x: math.modf(x)[1])
		pirsensor_df['start_partial'] = pirsensor_df['start'].map(lambda x: 0 if math.modf(x)[0]<0.1 else 1)
		pirsensor_df['end_partial'] = pirsensor_df['end'].map(lambda x: 1 if math.modf(x)[0]<0.9 else 0)
		
		#initialize columns on target 
		targets_df['pir_sensor'] = np.nan
		targets_df['pir_sensor'] = pd.to_numeric(targets_df['pir_sensor'], errors='coerce')
		targets_df['pir_total_time'] = np.nan
		targets_df['pir_partial'] = np.nan # 0 if it was less that 0.2 seconds, 1 otherwise
		targets_df['pir_movement'] = 0
		targets_df['pir_several'] = 0
		
		#check for sensor on at the same time (only one should be on) and mark that second as 1 (0 when there is only 1 on) 
		all_segs = []
		for row in pirsensor_df.iterrows():
			start = row[1].start_new_int
			end = row[1].end_new_int
			# Calculate all the seconds
			for sec in range(int(start), int(end)+1):
				all_segs.append(sec)
		dups = [item for item, count in collections.Counter(all_segs).items() if count > 1]
		for i in dups:
			targets_df.loc[i, 'pir_several'] = 1
		
		'''
		Add sensor info to target_df:
		a. pir_sensor: sensor number
		b. pir_total_time: total time the sensor was on 
		c. pir_sensor_room: sensor room. str.
		d. pir_movement: 1 if movement detected, 0 otherwise.
		e. pir_partial: 1 if started/stoped movement on that second. 0 otherwise
		'''
		for row in pirsensor_df.iterrows():
			start = row[1].start_new_int
			end = row[1].end_new_int
			# Calculate all the seconds
			for sec in range(int(start), int(end)+1):
				targets_df.loc[targets_df.start==sec, 'pir_sensor'] = row[1]['index']
				targets_df.loc[targets_df.start==sec, 'pir_total_time'] = row[1].total_time
				targets_df.loc[targets_df.start==sec, 'pir_sensor_room'] = row[1]['name']
				targets_df.loc[targets_df.start==sec, 'pir_movement'] = 1
				if sec == start:
					targets_df.loc[targets_df.start==sec, 'pir_partial'] = row[1].start_partial
				elif sec == end:
					targets_df.loc[targets_df.start==sec, 'pir_partial'] = row[1].end_partial
				else:
					targets_df.loc[targets_df.start==sec, 'pir_partial'] = 0
		
		#fill up missing values. People might be iddle when in a room so we fill up NaN values with previous value
		for row in targets_df.iterrows():
			if pd.notnull(row[1].pir_sensor):
				last_sensor = row[1].pir_sensor
				last_sensor_room = row[1].pir_sensor_room
			else:
				targets_df.loc[row[0], 'pir_sensor'] = last_sensor
				targets_df.loc[row[0], 'pir_sensor_room'] = last_sensor_room
			
			
		#ACCELERATION

		# group seconds by aggregation mean and std of all columns
		acceleration_df["start"] = acceleration_df.t.map(lambda x: math.modf(x)[1])    
		gb_acc = acceleration_df.groupby('start')
		acc_mean = gb_acc['x','y','z','Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP'].aggregate(np.mean)
		acc_std = gb_acc['x','y','z','Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP'].aggregate(np.std)

		# add info to new data frame with seconds    
		acceleration_sec_df = pd.DataFrame()
		acceleration_sec_df['start'] = acc_mean.index
		acceleration_sec_df['acc_x_mean'] = acc_mean['x'].values
		acceleration_sec_df['acc_y_mean'] = acc_mean['y'].values
		acceleration_sec_df['acc_z_mean'] = acc_mean['z'].values
		acceleration_sec_df['acc_x_std'] = acc_std['x'].values
		acceleration_sec_df['acc_y_std'] = acc_std['y'].values
		acceleration_sec_df['acc_z_std'] = acc_std['z'].values
		acceleration_sec_df['acc_kitchen_mean'] = acc_mean['Kitchen_AP'].values
		acceleration_sec_df['acc_lounge_mean'] = acc_mean['Lounge_AP'].values
		acceleration_sec_df['acc_upstairs_mean'] = acc_mean['Upstairs_AP'].values
		acceleration_sec_df['acc_study_mean'] = acc_mean['Study_AP'].values
		acceleration_sec_df['acc_kitchen_std'] = acc_std['Kitchen_AP'].values
		acceleration_sec_df['acc_lounge_std'] = acc_std['Lounge_AP'].values
		acceleration_sec_df['acc_upstairs_std'] = acc_std['Upstairs_AP'].values
		acceleration_sec_df['acc_study_std'] = acc_std['Study_AP'].values
		
		#merge targets_df with new info into new data frame all_df
		all_df = pd.merge(targets_df, acceleration_sec_df, on='start', how='left')
		
		
		#VIDEO

		video_columns = list(vhallway_df.columns)
		video_columns.remove('t')
				#loop all 3 video csv dfs. Group by aggregating mean and std and merge wiht all_df    
		for df in (vhallway_df, vkitchen_df, vlivingroom_df):
			df["start"] = df.t.map(lambda x: math.modf(x)[1])

			gb_vid = df.groupby('start')
			v_mean = gb_vid[video_columns].aggregate(np.mean)
			v_std = gb_vid[video_columns].aggregate(np.std)

			video_df = pd.DataFrame()
			video_df['start'] = v_mean.index
			for col in video_columns:
				video_df['vid_hall_'+col+'_mean'] = v_mean[col].values
				video_df['vid_hall_'+col+'_std'] = v_std[col].values
				
			all_df = pd.merge(all_df, video_df, on='start', how='left')
				
		all_df.drop(['start','end'], axis=1, inplace= True)
		z=z+1
		print(z)
		#msno.matrix(all_df)
		
		#save file for sequence
		all_df.to_csv('.\\data\\all_df_'+str(z)+'.csv',index=False)
    
    
    