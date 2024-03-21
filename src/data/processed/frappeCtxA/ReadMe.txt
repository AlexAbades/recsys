Preprossed Frappe Data

Following the work from Moshe Unger et al. "Context-Aware Recommendations Based on Deep Learning Frameworks" in ACM 2020. 
The processed data file is Contextual.py

The contextual features used are: 
  - cnt, daytime, weather, isweekend and homework.
The user and items are: 
  - user, item
The goal is to predict reatings:
  - rating

From which all users and items with less than 5 interactions have been removed.

Number of users = 651 
Reduction of 31.98 % 

Number of items = 1127
Reduction of 72.39 % 

The contextual Nominal features (daytime, weather, isweekend and homework) have been binarized using one-hot enconding, the 
numerical features (cnt) has been tranformed using log base 10 to scale it from 0 to 4.46 and then normalized using min max. 

The final data consist on: 
0 - user
1 - item
2 - rating
3 - cnt
4 - daytime_afternoon
5 - daytime_evening
6 - daytime_morning
7 - daytime_night
8 - daytime_noon
9 - daytime_sunrise
10 - daytime_sunset
11 - weather_sunny
12 - weather_cloudy
13 - weather_unknown
14 - weather_foggy
15 - weather_rainy
16 - weather_stormy
17 - weather_drizzle
18 - weather_snowy
19 - isweekend_weekend
20 - isweekend_workday
21 - homework_home
22 - homework_unknown
23 - homework_work

