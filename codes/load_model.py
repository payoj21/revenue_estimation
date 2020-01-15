import sys
import os.path
from os import path

import math
import pandas as pd
import numpy as np
import datetime

# from random import shuffle
from random import seed
seed(20)

from catboost import CatBoostRegressor, Pool, cv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns


class load_data():
	def __init__(self, output, country):
		self.output = output
		self.country = country
		self.features = ["week","year","availability","ACCOMMODATION_TYPE","NUMBER_OF_PERSONS",'BEDROOM_COUNT', 'BATHROOM_COUNT', 'HAS_WIFI',
                         'HAS_POOL', 'LATITUDE', 'LONGITUDE', 'ELEVATION']

	@staticmethod
	def read_data_files():
		bookings = pd.read_csv(r"../data/Booking_level_data.csv")
		date_to_week = pd.read_csv(r"../data/weeks_start_date.csv")
		accommodations = pd.read_csv(r"../data/X_data.csv")
		# bookings = pd.read_csv(r"../data/Austria_2020_mom_gbv.csv")
		# availability = pd.read_csv(r"../data/Austria_availability.csv")
		weekly_availability = pd.read_csv(r"../data/week_wise_availability.csv")
		return bookings, date_to_week, accommodations, weekly_availability

		# return accommodations, bookings, availability
	def read_data(self):
		return pd.read_csv(r"../data/"+self.country+"/test.csv")
	
	def test_X(self, data):
		return data[self.features]
	
	def test_Y(self, data):
		return data[self.output]

class model():
	def __init__(self, output, country, lr, iter, l2_lambda, depth):
		self.output = output
		self.country = country
		self.model = CatBoostRegressor()
		self.lr = lr
		self.iter = iter
		self.l2_lambda = l2_lambda
		self.depth = depth
	
	def load_model(self):
		self.model.load_model(fname="../models/"+self.country+"/"+self.output+"/"+str(self.lr)+"_"+str(self.iter)+"_"+str(self.l2_lambda)+"_"+str(self.depth)+"_model.json", format="json")

	def predict(self,test_X):
		return self.model.predict(test_X)

class evaluate():
	def error_analysis(self, predicted, test_Y):
		error = predicted - test_Y
		error_percent = np.array([])
		predicted_y_ratio = np.array([])
		for i, ele in enumerate(predicted):
			y = test_Y[i]
			if y != 0:
				error_percent = np.append(error_percent, (ele-y)/y)
				predicted_y_ratio = np.append(predicted_y_ratio, ele/y)
			else:
				error_percent = np.append(error_percent, -1)
				predicted_y_ratio = np.append(predicted_y_ratio, -1)
		
		error_rmse = math.sqrt((error**2).mean())
		# error_percent = error_percent_arr
		# predicted_y_ratio = predicted_y_ratio_arr
		return error, error_rmse, error_percent, predicted_y_ratio


class plots():

	def __init__(self, output, country):
		self.output = output
		self.country = country

	def plot_output_rmse_histogram(self, error):
		plt.figure(figsize=(20,8))
		plt.hist(error, bins=1000)
		plt.xlim(-1000,1000)
		plt.ylabel("count")
		plt.xlabel("error")
		plt.legend()
		plt.savefig("../images/AT.png")

	def plot_output_ratio_histogram(self, ratio):
		plt.figure(figsize=(20,8))
		plt.hist(ratio, bins=1000)
		plt.xlim(0,5)
		plt.ylim(0,500)
		plt.ylabel("count")
		plt.xlabel("prediction/revenue_per_week")
		plt.legend()
		plt.savefig("../images/AT_ratio.png")

	def plot_weekwise_output_curve(self, weekwise_output):
		plt.figure(figsize=(20,5))
		plt.plot(weekwise_output.index, weekwise_output.predicted_revenue, label="Predicted")
		plt.plot(weekwise_output.index, weekwise_output.revenue, label="weekly revenue")
		plt.xlabel("Week")
		plt.ylabel("Revenue")
		plt.legend()
		plt.savefig("../images/"+self.country+"_weekly_"+self.output+"_prediction.png")

	def plot_annual_predicted_curve(self, annual_output):
		plt.figure(figsize=(20,5))
		x = [i for i in range(len(annual_output.index))]
		ticks = annual_output.index
		plt.plot(x, annual_output.predicted_revenue, label="Predicted")
		plt.xticks(x, ticks)
		plt.xlabel("Property id")
		plt.ylabel("Predicted Revenue")
		plt.legend()
		plt.savefig("../images/"+self.country+"_annual_"+self.output+"_prediction.png")

	def plot_annual_revenue_curve(self, annual_output):
		plt.figure(figsize=(20,5))
		x = [i for i in range(len(annual_output.index))]
		ticks = annual_output.index
		plt.plot(x, annual_output.revenue, label="annual revenue")
		plt.xticks(x, ticks)
		plt.xlabel("Property id")
		plt.ylabel("Actual Revenue")
		plt.legend()
		plt.savefig("../images/"+self.country+"_annual_"+self.output+"_revenue.png")
 

if __name__ == '__main__':
    output = sys.argv[1]
    country = sys.argv[2]
    lr = float(sys.argv[3])
    iterations = int(sys.argv[4])
    l2_lambda = float(sys.argv[5])
    depth = int(sys.argv[6])

    data_obj = load_data(output, country)
    data = data_obj.read_data()

    # # bookings, date_to_week, accommodations, weekly_availability = data_obj.read_data_files()

    # acco_details = pd.merge(weekly_availability, accommodations, on="ACCOMMODATION_CODE", how="inner")
    # # bookings = pd.merge(bookings, acco, on="ACCOMMODATION_CODE", how="inner")
    
    test_X = data_obj.test_X(data)
    test_Y = data_obj.test_Y(data)

    model_obj = model(output, country, lr, iterations, l2_lambda, depth)
    model_obj.load_model()
    predictions = model_obj.predict(test_X)
    data["predicted_revenue"] = predictions

    evaluation_obj = evaluate()
    error, error_rmse, error_percent, predicted_y_ratio = evaluation_obj.error_analysis(predictions, test_Y)

    print(error_rmse)

    weekwise = data.groupby(["week"]).sum()
    # weekwise = weekwise.reset_index()

    acco_weekwise = data.groupby(["ACCOMMODATION_CODE","week"]).sum()
    acco_weekwise = acco_weekwise.reset_index()
    acco_weekwise = acco_weekwise[["ACCOMMODATION_CODE", "week", "predicted_revenue", "revenue"]]
    annual = acco_weekwise.groupby("ACCOMMODATION_CODE").sum()
    # first_two_months = weekwise.drop(weekwise.index[weekwise.week > 10])
    # first_two_months = first_two_months.groupby("ACCOMMODATION_CODE").sum()
    # print(bookings.columns)
    # acco_revenue = bookings.drop(bookings.index[bookings.month > 2])
    
    # acco_revenue = acco_revenue.groupby(["ACCOMMODATION_CODE"]).sum()

    # acco_revenue.to_csv(r"../data/"+country+"/first_two_months_actual_gbv.csv")

    # print(weekwise.columns)

    # first_two_months.to_csv(r"../data/"+country+"/first_two_months_predicted_gbv.csv")

    annual.to_csv(r"../data/"+country+"/annual_gbv_and_predicted.csv")
    weekwise.to_csv(r"../data/"+country+"/weekly_gbv_and_predicted.csv")
    
    plot_obj = plots(output, country)
    plot_obj.plot_output_rmse_histogram(error)
    plot_obj.plot_output_ratio_histogram(predicted_y_ratio)
    plot_obj.plot_weekwise_output_curve(weekwise)
    plot_obj.plot_annual_predicted_curve(annual)
    plot_obj.plot_annual_revenue_curve(annual)
    

