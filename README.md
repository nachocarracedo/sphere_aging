# sphere_aging

The task for this challenge is to recognize activities from the sensor data collected from participants. Here, "activity recognition" is the task of recognizing the posture and movements of the participants whose data was recorded, and our definition most closely aligns with the definition given by the accelerometer community. Three sensor modalities are provided for the prediction task:
Accelerometer - Sampled at 20 Hz;
RGB-D - Bounding box information given to preserve anonymity of participants;
Environmental - The values of passive infrared (PIR) sensors are given.
Lables of the dataset:
a_ascend - ascend stairs;
a_descend - descend stairs;
a_jump - jump;
a_loadwalk - walk with load;
a_walk - walk;
p_bent - bending;
p_kneel - kneeling;
p_lie - lying;
p_sit - sitting;
p_squat - squatting;
p_stand - standing;
t_bend - stand-to-bend;
t_kneel_stand - kneel-to-stand;
t_lie_sit - lie-to-sit;
t_sit_lie - sit-to-lie;
t_sit_stand - sit-to-stand;
t_stand_kneel - stand-to-kneel;
t_stand_sit - stand-to-sit;
t_straighten - bend-to-stand; and
t_turn - turn.

The prefix 'a' on a label indicates an ambulation activity (i.e. an activity requiring of continuing movement), the prefix 'p' indicate static postures (i.e. times when the participants are stationary), and the prefix 't' indicate posture-to-posture transitions. These labels are the target variables that are to be predicted in the challenge
Each target is a vector of length 20 and aggregates the annotations of all annotators over one second non-overlapping windows. To understand the elements of the target vector, if, for a particular target vector, the activity a_walk is given a value of 0.05, this should be interpreted as meaning that on average the annotators marked 5% of the window as arising from walking.
