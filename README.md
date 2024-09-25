# Automatic-Geolocation-Model
 Code for a NN model for determining geolocation data from only RGB information



I'll keep adding to this one as we go along.


Here's the Introduction from my Proposal (or, rather the start of it, enough to get the idea across!)

---



### What are we doing here?
 Most image formats contain geolocation information in their metadata or headers; however, in instances where this is not the case, we propose that that geolocation information can be somewhat accurately determined from the RGB data in the image alone, i.e. local image features.

These features, referred to herein as Geo Informative features cumulatively inform a machine learning model which determines LAT & LONG coordinates.  Our data is almost entirely sourced from the Google Steet View API.

The popular online video game GeoGuessr, in essence gamifies this process for humans. Players are given some randomized google Streetview location and their performance is scored as some function of the topological distance between the location that they choose by placing a pin onto a map and the true location of capture. Naturally, the process of narrowing down an answer includes recursively shrinking a pool of candidate locations by assessing decreasingly verbose image features. Such an approach translates well to a ML model; wherein some members of our Geo Informer Pattern are the most meaningful, that is, they shrink the candidate pool in the greatest way, or they are so unique or identifying (i.e. a sign reading ‘Carter’s Road Grocer, Carterton’s Top Grocer! or a particularly recognizable landmark like the Eiffel Tower).



### What's in the repo?

#### Tools:
 1. A handy GUI tool for downloading google street view data (BYO api key) using QT5
 2. A handy tool to generate evenly distributed geolocation data for streetview images (as even as we can be given the obvious topologicial challenges





#### Models:
 1. Todo

#### src:
 1. Todo

#### docs:
 1. Todo





#### frontend:
 1. React frontend for uploading images to the model (rather to a rest API hosted on code you'll find in /server/)
 2. Some fun and fancy css!

