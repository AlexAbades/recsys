Preprossed Frappe Data

Following the work from Moshe Unger et al. "Context-Aware Recommendations Based on Deep Learning Frameworks" in ACM 2020. 
The processed data file is Contextual.py


The contextual features used are: 
  - cnt, daytime, weather, isweekend and homework.
The user and items are: 
  - user, item
The goal is to predict reatings:
  - rating

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

17 - weather_sunny
11 - weather_cloudy
18 - weather_unknown
13 - weather_foggy
14 - weather_rainy
16 - weather_stormy
12 - weather_drizzle
15 - weather_snowy

19 - isweekend_weekend
20 - isweekend_workday

21 - homework_home
22 - homework_unknown
23 - homework_work