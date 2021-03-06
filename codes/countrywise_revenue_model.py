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

class read_data():
    @staticmethod
    def read_data_files():
        bookings = pd.read_csv(r"../data/Booking_level_data.csv")
        date_to_week = pd.read_csv(r"../data/weeks_start_date.csv")
        accommodations = pd.read_csv(r"../data/X_data.csv")
        weekly_availability = pd.read_csv(r"../data/week_wise_availability.csv")
        return bookings, date_to_week, accommodations, weekly_availability
        
    @staticmethod
    def read_model_data(country):
        return pd.read_csv(r"../data/"+country+"/model_data.csv")
    
    @staticmethod
    def read_corrupt_data(country):
        return pd.read_csv(r"../data/"+country+"/corrupt_data.csv")
    
    @staticmethod
    def read_cleaned_data(country):
        return pd.read_csv(r"../data/"+country+"/cleaned_data.csv")
    
    @staticmethod
    def read_daywise_bookings(country):
        return pd.read_csv(r"../data/"+country+"/daywise_bookings.csv")


class write_data():
    @staticmethod
    def write_model_data(model_data, country):
        model_data.to_csv(r"../data/"+country+"/model_data.csv")
    
    @staticmethod
    def write_corrupt_data(corrupted_data, country):
        corrupted_data.to_csv(r"../data/"+country+"/corrupt_data.csv")
        
    @staticmethod
    def write_cleaned_data(cleaned_data, country):
        cleaned_data.to_csv(r"../data/"+country+"/cleaned_data.csv")
        
    @staticmethod
    def write_daywise_bookings(dataframe, country):
        dataframe.to_csv(r"../data/"+country+"/daywise_bookings.csv")

class prepare_data_functions():
    def __init__(self, country, bookings, date_to_week, accommodations, weekly_availability):
        self.country = country
        self.bookings = bookings
        self.date_to_week = date_to_week
        self.weekly_availability = weekly_availability
        self.accommodations = accommodations
        
        self.country_daywise_bookings = pd.DataFrame(columns=["ACCOMMODATION_CODE", "BOOKING_ID", "arrivaldate", "departuredate","bookingdate", "date", "revenue"])
        self.country_daywise_week = pd.DataFrame(columns=["year", "week", "week_start", "week_end","date"])
       
    
    def convert_to_datetime(self):
        self.bookings['arrivaldate'] = pd.to_datetime(self.bookings['arrivaldate'])
        self.bookings['departuredate'] = pd.to_datetime(self.bookings['departuredate'])
        self.bookings['bookingdate'] = pd.to_datetime(self.bookings['bookingdate'])
        self.date_to_week["WK_START"] = pd.to_datetime(self.date_to_week["WK_START"])
        return self
    
    def add_week_end(self):
        self.date_to_week["WK_END"] = self.date_to_week["WK_START"] + datetime.timedelta(7)
        return self
    
    def get_country_bookings(self):
        self.country_bookings = self.bookings[(self.bookings["ACCOMMODATION_CODE"].str.contains(self.country))&(self.bookings["BOOKING_STATUS"]=="BOOKING")]
        return self
    
    def prepare_country_daywise_bookings(self):
        print("Preparing daywise bookings")
        count = 0
        for booking in self.country_bookings.values:
            count+=1
            if(count%1000==0):
                print(str(count)+" bookings processed")
            days_of_bookings = int((booking[5]-booking[4])/np.timedelta64(1,'D'))
            rev_per_day = booking[7]/days_of_bookings
            for i in range(days_of_bookings):
                date = booking[4] + datetime.timedelta(i)
                day_booking = pd.DataFrame([(booking[0], booking[1], booking[4], booking[5], booking[6], date, rev_per_day)], columns=["ACCOMMODATION_CODE", "BOOKING_ID", "arrivaldate", "departuredate","bookingdate", "date", "revenue"])
                self.country_daywise_bookings = self.country_daywise_bookings.append(day_booking)
        self.country_daywise_bookings = self.country_daywise_bookings.reset_index().drop(["index"], axis=1)
        print("country daywise bookings dataframe created")
        return self
    
    def prepare_country_daywise_week(self):
        for week in self.date_to_week.values:
            days_of_booking = int(((week[3]-week[2]) / np.timedelta64(1, 'D')))
        #     rev_per_day = booking[7]/days_of_booking
            for i in range(days_of_booking):
                date = week[2] + datetime.timedelta(i)
                week_df = pd.DataFrame([(week[0], week[1], week[2], week[3], date)], columns=["year", "week", "week_start", "week_end","date"])
                self.country_daywise_week = self.country_daywise_week.append(week_df)
        self.country_daywise_week = self.country_daywise_week.reset_index().drop(["index"], axis=1)
        return self
    
    def merge_country_daywise_bookings_with_week(self):
        self.country_daywise_bookings = pd.merge(self.country_daywise_bookings, self.country_daywise_week, on="date", how="left")
        return self
        
    def get_country_weekly_revenue(self):
        country_weekly_bookings = self.country_daywise_bookings.groupby(["year","week","ACCOMMODATION_CODE"]).sum()
        country_weekly_bookings = country_weekly_bookings.reset_index()
        return country_weekly_bookings
    
    def get_country_weekly_occupancy(self):
        country_weekly_occupancy = self.country_daywise_bookings.groupby(["year","week","ACCOMMODATION_CODE"]).count()["date"]
        country_weekly_occupancy = country_weekly_occupancy.reset_index()
        country_weekly_occupancy = country_weekly_occupancy.rename(columns={'date':'#_days_booked'})
        country_weekly_occupancy["occupancy"] = country_weekly_occupancy["#_days_booked"]/7
        return country_weekly_occupancy
    
    @staticmethod
    def merge_country_weekly_revenue_and_occupancy(country_weekly_bookings, country_weekly_occupancy):
        country_weekly_revenue_occupancy = pd.merge(country_weekly_bookings, country_weekly_occupancy, on=["year","week","ACCOMMODATION_CODE"], how="left")
        return country_weekly_revenue_occupancy
    
    def get_country_availability(self):
        country_weekly_availability = self.weekly_availability[self.weekly_availability["ACCOMMODATION_CODE"].str.contains(self.country)]
        return country_weekly_availability
    
    @staticmethod
    def merge_country_bookings_with_weekly_availability(country_weekly_revenue_occupancy, country_weekly_availability):
        country_bookings_with_availability = pd.merge(country_weekly_availability, country_weekly_revenue_occupancy, on=["year", "week", "ACCOMMODATION_CODE"], how="outer")
        country_bookings_with_availability = country_bookings_with_availability.fillna(0)
        return country_bookings_with_availability
    
    def get_country_accommodations(self):
        return self.accommodations[self.accommodations["COUNTRY"] == self.country]
    
    @staticmethod
    def merge_accommodations_with_their_weekly_revenue(country_accommodations, country_bookings_with_availability):
        data = pd.merge(country_bookings_with_availability, country_accommodations, on="ACCOMMODATION_CODE", how="left")
        return data
    
    

class prepare_data:
    def __init__(self, country):
        self.country = country
        self.read_data_obj = read_data()
        self.write_data_obj = None
        self.prep_data_obj = None
        
    def write_model_data(self):
        self.write_data_obj = write_data()
        bookings, date_to_week, accommodations, weekly_availability = self.read_data_obj.read_data_files()
        self.prep_data_obj = prepare_data_functions(country, bookings, date_to_week, accommodations, weekly_availability)
        self.prep_data_obj.convert_to_datetime()
        self.prep_data_obj.add_week_end()
        self.prep_data_obj.get_country_bookings()
        self.prep_data_obj.prepare_country_daywise_bookings()
        self.prep_data_obj.prepare_country_daywise_week()
        self.write_data_obj.write_daywise_bookings(self.prep_data_obj.country_daywise_bookings, self.country)
        self.prep_data_obj.merge_country_daywise_bookings_with_week()
        country_weekly_revenue = self.prep_data_obj.get_country_weekly_revenue()
        country_weekly_occupancy = self.prep_data_obj.get_country_weekly_occupancy()
        country_weekly_revenue_occupancy = self.prep_data_obj.merge_country_weekly_revenue_and_occupancy(country_weekly_revenue, country_weekly_occupancy)
        country_weekly_availability = self.prep_data_obj.get_country_availability()
        country_bookings_with_availability = self.prep_data_obj.merge_country_bookings_with_weekly_availability(country_weekly_revenue_occupancy, country_weekly_availability)
        country_accommodations = self.prep_data_obj.get_country_accommodations()
        model_data = self.prep_data_obj.merge_accommodations_with_their_weekly_revenue(country_accommodations, country_bookings_with_availability)
        self.write_data_obj.write_model_data(model_data, self.prep_data_obj.country)
        return model_data
    
    def read_model_data(self):
        return self.read_data_obj.read_model_data(self.country)
        
class prepare_corrupted_data:
    def __init__(self, country, data):
        self.country = country
        self.read_data_obj = read_data()
        self.write_data_obj = write_data()
        self.data = data
    
    def get_corrupt_data(self):
        corrupted_data = self.data[(self.data["availability"] == 0) & ((self.data["occupancy"] > 0) | (self.data["revenue"] > 0))]
        corrupted_data = pd.concat([corrupted_data,self.data[(self.data["revenue"] > self.data["total2019"] + self.data["total2018"]) & (self.data["year"] < 2020)]], axis=0)
        corrupted_data = pd.concat([corrupted_data, self.data[self.data["availability"].isnull()]], axis=0)
        corrupted_data = pd.concat([corrupted_data, self.data[self.data["ACCOMMODATION_TYPE"].isnull()]], axis=0)
        corrupted_data = corrupted_data.reset_index()
        return corrupted_data

    def write_corrupt_data(self, corrupt_data):
        self.write_data_obj.write_corrupt_data(corrupt_data, self.country)
        # return corrupt_data
    
    def read_corrupt_data(self):
        return self.read_data_obj.read_corrupt_data(self.country)
    
class prepare_cleaned_data:
    def __init__(self, country, corrupt_data, model_data):
        self.country = country
        self.write_data_obj = write_data()
        self.read_data_obj = read_data()
        self.model_data = model_data
        self.corrupt_data = corrupt_data

    def get_cleaned_data(self):
        cleaned_data = self.model_data.drop(self.model_data.index[self.corrupt_data.index])
        cleaned_data = cleaned_data.reset_index().drop(["Unnamed: 0", "index"], axis=1)
        cleaned_data["distance_from_coast"] = cleaned_data["distance_from_coast"].fillna(-999)
        return cleaned_data
   
    def write_cleaned_data(self, cleaned_data):
        self.write_data_obj.write_cleaned_data(cleaned_data, self.country)
        # return cleaned_data
    
    def read_cleaned_data(self):
        return self.read_data_obj.read_cleaned_data(self.country)

class DataHandler:
    def __init__(self, cleaned_data):
        self.data = cleaned_data
        self.threshold = 0.8
        self.features = ["week","year","availability","ACCOMMODATION_TYPE","NUMBER_OF_PERSONS",'BEDROOM_COUNT', 'BATHROOM_COUNT', 'HAS_WIFI',
                         'HAS_POOL', 'LATITUDE', 'LONGITUDE', 'ELEVATION']
        
        self.outputs = ["revenue", "occupancy"]
        
        self.features_datatype = {'week': int, 
                                  'year': int,
                                  'ACCOMMODATION_TYPE': int,
                                  'NUMBER_OF_PERSONS': int,
                                  'BEDROOM_COUNT': int,
                                  'BATHROOM_COUNT': int,
                                  'HAS_WIFI': int,
                                  'HAS_POOL': int,
                                  'if_near_coast':int
                                 }
        
    def get_model_data(self):
        self.data = self.data.drop(self.data.index[self.data["availability"]==0])
        self.data = self.data[self.data["year"]<2020]
        return self.data
        
    def get_features(self):
#         return [feature for feature in feature_arr if feature not in self.features]
        return self.features

    def get_output(self):
#         outputs = ["revenue", "occupancy"]
        return self.outputs
    
    @staticmethod
    def get_unique_accommodations(model_data):
        return model_data["ACCOMMODATION_CODE"].unique()
    
    def get_features_datatype(self):
        return self.features_datatype
    
    def split_train_test(self, unique_accommodations):
        # print(len(unique_accommodations))
        indices = np.random.rand(len(unique_accommodations)) < self.threshold
        train_accommodations = unique_accommodations[indices]
        test_accommodations = unique_accommodations[~indices]
        train = self.data.merge(pd.DataFrame(set(train_accommodations).intersection(self.data["ACCOMMODATION_CODE"].tolist()), columns=["ACCOMMODATION_CODE"]), on=["ACCOMMODATION_CODE"], how="inner")
        test = self.data.merge(pd.DataFrame(set(test_accommodations).intersection(self.data["ACCOMMODATION_CODE"].tolist()), columns=["ACCOMMODATION_CODE"]), on=["ACCOMMODATION_CODE"], how="inner")        
        return train, test

    
class prepare_model_data:
    def __init__(self, cleaned_data, country, features, output):
        self.cleaned_data = cleaned_data
        self.country = country

        self.data_handler_obj = DataHandler(self.cleaned_data)

    def get_model_data(self):
        model_data = self.data_handler_obj.get_model_data()
        unique_accommodation = self.data_handler_obj.get_unique_accommodations(model_data)
        train, test = self.data_handler_obj.split_train_test(unique_accommodation)
        return train, test

class Model():
    def __init__(self, country=None, tree_depth = 10, loss_func="RMSE", iteration=1000, lr=0.01, reg_lambda=0.01, train=None, test=None, features=None, output=None):
        self.country = country
        self.iter = iteration
        self.reg_lambda = reg_lambda
        self.train = train
        self.test = test
        self.features = features
        self.output = output
        self.loss = loss_func
        self.depth = tree_depth
        self.learning_rate = lr
        
        self.seed = 2019
        self.threshold = 0.8
        
        self.model = CatBoostRegressor(
            loss_function=self.loss,
            iterations=self.iter,
            depth=self.depth,
            random_seed=self.seed,
            logging_level="Silent",
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.reg_lambda,
        )
        
    
    def separate_X_Y(self):
        return self.train[self.features], self.test[self.features], self.train[self.output], self.test[self.output]
    
    @staticmethod
    def get_categorical_feature_indices(train_X):
        return np.where(train_X.dtypes != np.float64)[0]
    
    def split_train_validation_set(self, train_X, train_Y):
        X_train, X_validation, y_train, y_validation = train_test_split(train_X, train_Y, train_size=self.threshold, random_state=42)
        return X_train, X_validation, y_train, y_validation
    
    def fit(self, X_train, X_validation, y_train, y_validation,categorical_features_indices):
        self.model.fit(
            X_train,
            y_train,
            cat_features=categorical_features_indices,
            eval_set=(X_validation, y_validation)
        )
        return self
    
    def predict(self, test_X):
        return self.model.predict(test_X)
    
    def evaluate(self, test_Y, predicted_Y):
        error = predicted_Y - test_Y
        error_percent_arr = np.array([])
        predicted_y_ratio_arr = np.array([])
        for i, ele in enumerate(predicted_Y):
            y = test_Y[i]
            if y != 0:
                error_percent_arr = np.append(error_percent_arr, (ele-y)/y)
                predicted_y_ratio_arr = np.append(predicted_y_ratio_arr, ele/y)
            # else:
            #     error_percent = np.append(error_percent, -1)
            #     predicted_y_ratio = np.append(predicted_y_ratio, -1)

        error_rmse = math.sqrt((error**2).mean())
        error_percent = error_percent_arr.mean()
        predicted_y_ratio = predicted_y_ratio_arr.mean()
        return error, error_rmse, error_percent, predicted_y_ratio

    def cross_validation(self, model, train_X, train_Y, categorical_features_indices):
        cv_params = model.get_params()
        cv_data = cv(
            Pool(train_X, train_Y, cat_features=categorical_features_indices),
            cv_params,
            plot=True
        )
        return cv_data
    def save_model(self, train_X):
        self.model.save_model(fname=r"../models/"+self.country+"/"+self.output+"/"+str(self.learning_rate)+"_"+str(self.iter)+"_"+str(self.reg_lambda)+"_"+str(self.depth)+"_model.json", format="json", export_parameters=None, pool=train_X)

if __name__ == '__main__':
    countries = sys.argv[1]            
    # output = sys.argv[2]
    countries = countries.replace(" ", "")
    countries_list = countries.split(",")

    # output_list = output.replace(" ", "").split(",")
    print("Countries the model is running for : ", countries_list)

    # country = "FR"
    learning_rates = [0.01, 0.05, 0.1]
    l2_lambdas = [0.0001, 0.0005, 0.01, 0.05, 0.1]
    iterations = [1500, 2000, 2500, 5000]
    depths = [10, 16]


    for country in countries_list:
        if_country = path.exists(r"../data/"+country+"/model_data.csv")
        prepare_data_object = prepare_data(country)
        if(if_country):
            print ("Model data for "+country+" exists")
            data = prepare_data_object.read_model_data()
        else:
            print ("Model data doesn't exists. Creating model data for "+country)
            data = prepare_data_object.write_model_data()

        prepare_corrupt_data_object = prepare_corrupted_data(country, data)
        corrupt_data = prepare_corrupt_data_object.get_corrupt_data()
        prepare_corrupt_data_object.write_corrupt_data(corrupt_data)

        prepare_cleaned_data_object = prepare_cleaned_data(country, corrupt_data, data)
        cleaned_data = prepare_cleaned_data_object.get_cleaned_data()
        prepare_cleaned_data_object.write_cleaned_data(cleaned_data)

        datahandle_object = DataHandler(cleaned_data)
        features = datahandle_object.get_features()
        outputs = datahandle_object.get_output()

        prepare_model_data_object = prepare_model_data(cleaned_data, country, features, outputs)
        train, test = prepare_model_data_object.get_model_data()

        test.to_csv(r"../data/"+country+"/test.csv")
        train.to_csv(r"../data/"+country+"/train.csv")

        print("Training model")
        
        for output in outputs:

            for learning_rate in learning_rates:
                for l2_lambda in l2_lambdas:
                    for iteration in iterations:
                        for depth in depths:
                            print("iterations : ", iteration, "\tlearning rate : ", learning_rate, "\tlambda : ", l2_lambda, "\tdepth : ", depth)
                            
                            model_obj = Model(country=country, 
                                iteration=iteration, 
                                train=train, 
                                test=test, 
                                lr=learning_rate, 
                                reg_lambda=l2_lambda, 
                                tree_depth=depth, 
                                features=features, 
                                output=output
                                )

                            train_X, test_X, train_Y, test_Y = model_obj.separate_X_Y()

                            categorical_features_indices = model_obj.get_categorical_feature_indices(train_X)

                            X_train, X_validation, y_train, y_validation = model_obj.split_train_validation_set(train_X, train_Y)

                            # print(len(X_train), len(y_train), len(X_validation), len(y_validation))
                            print("Length of training set : ", len(X_train), "\tLength of training labels : ", len(y_train))
                            print("Length of validation set : ", len(X_validation), "\tLength of validation labels : ", len(y_validation))
                            print("Length of test set : ", len(test_X), "\tLength of test labels : ", len(test_Y))
                            
                            model_obj.fit(X_train, X_validation, y_train, y_validation, categorical_features_indices)
                            
                            print("Done with the "+output+" model training. Now, saving the model.")
                            model_obj.save_model(X_train)

                            test_predictions = model_obj.model.predict(test_X)
                            output_field = "predicted_"+output
                            test[output_field] = test_predictions
                            # test_score = model_obj.evaluate(test_Y, test_predictions)
                            test_error, test_error_rmse, test_error_percent, test_predicted_y_ratio = model_obj.evaluate(test_Y, test_predictions)
                            
                            train_predictions = model_obj.model.predict(train_X)
                            train[output_field] = train_predictions
                            # train_score = model_obj.evaluate(train_Y, train_predictions)
                            train_error, train_error_rmse, train_error_percent, train_predicted_y_ratio = model_obj.evaluate(train_Y, train_predictions)
                            
                            print("training RMSE : ", train_error_rmse, " \ttest RMSE : ", test_error_rmse, "\t training prediction/revenue ratio: ", train_predicted_y_ratio, "\ttest prediction/revenue ratio: ", test_predicted_y_ratio,"\n\n")
                            
                            with open(r"../models/"+country+"/error_statements.txt", "a+") as file:
                                file.write("Country : "+country+"\t"+output+" model\n")
                                file.write("iterations : "+str(iteration)+"\tlearning rate : "+str(learning_rate)+"\tlambda : "+str(l2_lambda)+"\tdepth : "+str(depth)+"\n") 
                                file.write("training RMSE : "+str(train_error_rmse)+" \t test RMSE : "+str(test_error_rmse)+"\t training prediction/revenue ratio: "+str(train_predicted_y_ratio)+"\t test prediction/revenue ratio: "+str(test_predicted_y_ratio)+"\n\n")
                                file.close()