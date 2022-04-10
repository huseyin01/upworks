########################################################################################################
# Really helps to have anaconda installed for this: https://docs.anaconda.com/anaconda/install/linux/
#
# This is the simplest program len could think of to train a scikit-learn ML model to predict wait
# times at Walt Disney World. 
#
# Future ideas:
#   See https://facebook.github.io/prophet/docs/quick_start.html
########################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor


########################################################
# a function that converts strings representing HH:MM,
# to number of minutes elapsed since midnight
#
# Returns -1 if str doesn't look like HH:MM
########################################################
def HHMM_to_mins_since_midnight(str):
	arr=str.split(":")
	
	if (len(arr)==2):
		return (60*int(arr[0]))+int(arr[1])
	return (-1)
# HHMM_to_mins_since_midnight




#############################################################
# rounds a number like 37 to the nearest multiple of 5 (35)
#############################################################
def round_to_nearest_5(f):
	return (int(f/5)*5.0)
# round_to_nearest_5	




###################################################################
# For a given predicted wait time, find the most likely
# posted wait time it corresponds to, based on the frequency
# with which posted wait times appear.
#
# For example, suppose the training data shows that posted
# wait times are distributed like this:
#
#	10 minutes appears 15,432 times
#	15 minutes appears      0 times
#	20 minutes appears 10,321 times
#
# If our model predicts 15 minutes, we know it's going to be
# wrong. So at that point, what's the next-best option? We
# should choose 10, because it's close to 15 and it is more
# likely than 20.
#
# input parameters:
# p is the predicted value, freq_arr is the frequency array
# returned by df.value_counts().sort_index() (it's a pandas Series)
#
# Remember that indexes in freq_arr are the posted wait times
#
# output:
#   the most likely posted wait time
#################################################################
def get_most_likely_posted(p, freq_arr):
	# this prints the index value
	#print(freq_arr.index[0])
	
	
	count=0		# the number of items we've seen in freq_arr
	last_i = p	# the value of the last index we saw in freq_arr
	last_v = 0
	for i, v in freq_arr.items():
		
		######################################################################
		# handles the case where p is less than the lowest value in freq_arr
		######################################################################
		if (count==0 and p <= i):
			return i
		
		#######################################################################
		# if this isn't the first row, then we have a "before" and "current"
		# value for i. Call them last_i and i. These are posted wait times
		#######################################################################
		if (count > 0):
			##############################################################
			# if p is between the the last value we saw, and this value
			# then figure out which of these two values to return
			##############################################################
			if (p >= last_i and p<= i):
				##########################################################
				# calculate the ratio of how many times the last value
				# has appeared, as a percent of (last_v + v)
				##########################################################
				last_v_ratio = last_v / (last_v + v)
				
				##############################################################
				# if that ratio is lopsided, go with the most likely one
				# For example, if one choice has appeared 90% of the time and 
				# the other has appeared 10%, just go with the 
				# one that's appeared 90% times
				##############################################################
				if (last_v_ratio > .70 or last_v_ratio < 0.30):
					###########################################
					# return the value seen most frequently
					###########################################
					if (last_v >= v):
						return last_i
					else:
						return i
				else:
					##########################################################
					# the ratios aren't lopsided, so look for the i that's
					# closest to p, and go with that one.
					#
					# if p is equidistant from last_i and i, go with
					# whatever value is seen more frequently, like above
					##########################################################
					last_i_diff = p - last_i  # the difference between p and last_i
					i_diff = i - p			  # the difference between p and i
					if (last_i_diff < i_diff):
						return last_i
					elif (last_i_diff > i_diff):
						return i
					else:
						########################################################
						# it's a tie in terms of distance from p. Go with the
						# most frequently seen posted wait time
						########################################################
						if (last_v >= v):
							return last_i
						else:
							return i
		
		########################################
		# save the last index and value seen
		########################################
		last_i = i
		last_v = v
		count += 1
	
	# if we get to this point, then p is larger than any posted wait time
	# we've seen. Return p.	
	return p
# get_most_likely_posted
	
	
	
##########################################################	
# Prepare to read in data using pandas
##########################################################
# Tell pandas that TimeOfDay should be a datetime 
parse_dates=['TimeOfDay']  
		
# training filename we want to use
train_filename="./Train2015-2018.csv"

# test filename we want to use
test_filename="./Train2019JAN.csv"

# The attraction we want to test
attr='AK07';

# set the columns we want to use. Note that one of them must be
# the attraction we want to test (attr).
# "MinsSunsetWDW" - didn't help with AK07
# "WDWTICKETSEASON" - didn't help with AK07
# "ss_10" makes March 2019 predictions worse, doesn't change April for AK07 
usecols=["TimeOfDay", attr, "time", "AKHourSegment", "WDWSEASON", "WDWINSESSION"] 


#############################################################################################
# Read in the training and test files, and use TimeOfDay as the index in a pandas dataframe
#############################################################################################
print("\nReading in training file")
train = pd.read_csv(train_filename, usecols=usecols, index_col="TimeOfDay", parse_dates=parse_dates)

print("\nReading in testing file")
test = pd.read_csv(test_filename, usecols=usecols, index_col="TimeOfDay", parse_dates=parse_dates)



###################################################################################
# Delete any rows from the dataframes where we don't have a target value for attr
###################################################################################
print("\nDeleting training rows where target isn't available")
train.dropna(subset=[attr], inplace=True)
test.dropna(subset=[attr], inplace=True)



#############################################################################################
# If we wanted to, here we could drop any training data that doesn't correspond to the same
# time of year as the data we're going to test on. (For example, if we're )
# testing on January, we could delete all the training data that isn't from 
# January.
#
# To do this, we'd get the earliest and latest dates (MM-DD) from the test dataframe, and delete
# anything in the training dataframe that isn't in that range
#############################################################################################



###################################################
# Show the distribution of the posted wait times
# for the training set and test set. sort_index()
# sorts the value_counts() array by the posted
# wait time.
###################################################
train_posted_wait_freqs = train[attr].value_counts().sort_index()
print("\nTraining values & frequencies:")
print(train_posted_wait_freqs)

test_posted_wait_freqs = test[attr].value_counts().sort_index()
print("\nTesting values & frequencies:")
print(test_posted_wait_freqs)




#########################################################################################
# append the test dataset to the train dataset, to make column manipulations like
# one-hot encoding consistent.
# We can use the index to split them back later
#########################################################################################
train = train.append(test)




#################################################################################
# Convert time strings that appear as HH:MM, to minutes elapsed since midnight
# because this probably makes more sense than a string like HH:MM.
#################################################################################
print("\nConverting time to minutes since midnight")
train['time_msm'] = train.apply(lambda row: HHMM_to_mins_since_midnight(row.time), axis=1)




#####################################################################################
# One-hot encode MK/AKHourSegment and WDWSeason. This can also be done in scikit-learn.
# The drop_first option drops the column from the dataframe
#####################################################################################
print("\nOne-hot encoding MK/AKHourSegment")
akhs_onehot = pd.get_dummies(train['AKHourSegment'], prefix='akhs', drop_first=True)
for col in akhs_onehot.columns: 
    train[col] = akhs_onehot[col] 


print("\nOne-hot encoding WDWSEASON")
wdws_onehot = pd.get_dummies(train['WDWSEASON'], prefix='wdws', drop_first=True)
for col in wdws_onehot.columns: 
	train[col] = wdws_onehot[col] 


#########################################################################
# Drop the columns that aren't numbers, because we just converted them
#########################################################################
train = train.drop("time", axis=1)
train = train.drop("AKHourSegment", axis=1)
train = train.drop("WDWSEASON", axis=1)




#################################################################################################
# get the names of all of the columns to train on (which is everything except the target, attr)
#################################################################################################
cols_to_train_on=[]
for col in train.columns:
	if (col != attr):
		cols_to_train_on.append(col)
#print(cols_to_train_on) 



#########################################
# Split into train and test data again
# Here we're doing it by date
#########################################
print("\nSplit into train and test")
df_train = train.loc['2017-01-01':'2018-12-31']
df_test = train.loc['2019-02-01':]


	
###################################################
# Show the distribution of the posted wait times
# for the training set and test set. sort_index()
# sorts the value_counts() array by the posted
# wait time.
###################################################
train_posted_wait_freqs = df_train[attr].value_counts().sort_index()
print("\nDF Training values & frequencies:")
print(train_posted_wait_freqs)

test_posted_wait_freqs = df_test[attr].value_counts().sort_index()
print("\nDF Training values & frequencies:")
print(test_posted_wait_freqs)	
	
	
	
##########################################
# show the columns, if we wanted to
##########################################
#print(df_train.tail(10).to_string() )
#print(df_test.head(10).to_string() )	



################################################################################
# Create and train the model.
# Note that changing criterion from mse to mae significantly increase runtime
# and may not give better results.
################################################################################
print("\nCreating the model")
model = RandomForestRegressor(n_estimators=30, criterion="mse")
print("Fitting...")
model.fit(df_train[cols_to_train_on],df_train[attr])
print("\nFeature importances:")
print(model.feature_importances_)


#############################
# make predictions
#############################
print("\nMaking predictions....")
df_test['preds'] = model.predict(df_test[cols_to_train_on])


############################################################
# Our predictions are most likely numbers like 24.39849747.
# Disney's posted wait time signs display numbers that are
# multiples of 5 (i.e., 0, 5, 10, 15, 20, etc.)
#
# Here we want to change our prediction to the most likely
# nearest posted wait time, based on the posted wait
# times we saw in the training data.
#
# There are 3 methods below, shown for teaching purposes.
# Len thinks the 'convert to most likely posted wait' is
# probably the best to go with.
############################################################
#####################################################################################
# round predictions to the nearst minute (just to make sure they look reasonable)
#####################################################################################
#print("\nRounding predictions to the nearest minute.")
#df_test['preds'] = df_test.apply(lambda row: round(row.preds,0), axis=1)
#print("\nHere's the distribution of the predictions after rounding to nearest minute:")
#pred_posted_wait_freqs = df_test['preds'].value_counts().sort_index()
#print(pred_posted_wait_freqs)


######################################################################
# convert to the nearest 5-minute number
######################################################################
#print("\nRounding predictions to the nearest 5 posted wait time.")
#df_test['preds'] = df_test.apply(lambda row: round_to_nearest_5(row.preds), axis=1)
#print("\nHere's the distribution of the predictions after nearest 5 adjustment:")
#pred_posted_wait_freqs = df_test['preds'].value_counts().sort_index()
#print(pred_posted_wait_freqs)


######################################################################
# convert to the most likely posted wait time
######################################################################
print("\nConverting predictions to the most likely posted wait time.")
df_test['preds'] = df_test.apply(lambda row: get_most_likely_posted(row.preds, train_posted_wait_freqs), axis=1)
print("\nHere's the distribution of the predictions after that adjustment:")
pred_posted_wait_freqs = df_test['preds'].value_counts().sort_index()
print(pred_posted_wait_freqs)



# Uncomment this line to satisfy yourself that the predictions look like they're supposed to
#print(df_test['preds'].tail(10).to_string())





##################################################################
# calculate the difference between the actual and the prediction
##################################################################
print("\nCalculating the different between actual and prediction")
df_test['diff']=df_test[attr] - df_test['preds']

	
	
###########################################################################################
# get all the rows between 11 a.m. and 5 pm, because that's the time window we care about
###########################################################################################
print("\nGetting rows between 11 a.m and 5 p.m.")
in_window = df_test.between_time('11:00', '17:00')



############################################################################################
# Now go through in_window and calculate the average error between 11 a.m. and 5 p.m. for
# each day we're testing. Print the average error for that day, as well as the overall
# average error.
############################################################################################
period_sum=0	# the aggregate error for the entire period (e.g., a month)
period_count=0	# the aggregate number of days in the period
day_sum=0		# the aggregate error for a day
day_count=0		# the number of samples we have for a day
row_count=0		# how many samples we've read in
curr_date=""	# the current day we're processing
print("\nDate,", "Average Error")
for index, row in in_window.iterrows():
	
	# get just the YYYYMMDD part of the date
	fmt_date=index.strftime("%Y-%m-%d")
	
	# if this is the first date we've seen, make it the current date
	if (row_count==0):
		curr_date = fmt_date
	
	# add one to the row count
	row_count += 1
	
	# if we're still in the same date, add in the error
	if (fmt_date == curr_date):
		day_sum += abs(row['diff'])
		day_count += 1
		period_sum += abs(row['diff'])
		period_count += 1
	else:
		if (day_count > 0):
			print(curr_date, ",", round(day_sum/day_count, 2))
		else:
			print(curr_date, ", 0.00")
		curr_date = fmt_date
		day_sum=0
		day_count=0

#################################################
# print out the last date at the end of the loop
#################################################
if (day_count > 0):
	print(curr_date, ",", round(day_sum/day_count, 2))
else:
	print(curr_date, ",0.00")

	
######################################
# print out the period average error
######################################
if (period_count > 0):
	print("OVERALL", round(period_sum/period_count,2))
else:
	print("OVERALL", "0.00")
