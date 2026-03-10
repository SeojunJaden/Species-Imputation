# Species-Imputation

## Overview
Species Imputation is a research project that analyzes iNaturalist observations in the four San Diego reserves
  - Scripps Coastal Reserve
  -  Dawson Los Monos Canyon Reserve
  -  Elliot Chaparral Reserve
  -  Kendall Frost Reserve.
    
The project imputes missing or underrepresented species in the dataset using machine learning models trained from Google Earth Data and iNaturalist to predict the probability that a species might exist.

Live Website:
**https://speciesimputation.vercel.app/**

## How it Works

### 1.) Data Collection
  - iNaturalist UCSD species observation data
  - Environmental data from Google Earth Engine

### 2.) Data Processing 
 - Cleaning observation dataset
 - Generating environmental features for model training 

### 3.) Machine Learning models 
Multiple models were tested to predict species 
  - Random Forest
  - XGBoost
  - LightGBM
The final prediction ranks species by probability of presence.

### 4.) Visualization/Website 
Findings are displayed on a website that shows all four reserves 
  - Explore each reserve and the different predicted species categorized into their families
  - Compares observed vs predicted species
  - Views the probability of each species in different reserves 

## Dependencies

### Data Sources
* iNaturalist observation dataset
* Google Earth Engine environmental data

### Libraries
* Scikit-learn
* LightGBM
* XGBoost
* RandomForest
* pandas
* numpy

## Limitations and Future Improvements
* As of now, the dataset used for training is solely based in San Diego and could be improved with data from reserves outside of it.
* Incorporating real-time observation updates
* Expanding the number of represented reserves beyond four
* Adding more website features

## Research & Presentation
SayStroop Media Research Paper: *Species Imputation — Predicting species off Reserve data points*
* https://docs.google.com/document/d/1fDnpI5mcAqans1eGCgeqNHCKhrHuwZxJkooVFL5e1Jw/edit?usp=sharing

Slideshow Presentation:
* https://www.canva.com/design/DAHC8TAs_Ks/v5QbPyFEIqj3iTT1F0lBSA/edit

## Team
* Carsten Petersen
* Kylan Huynh
* Max Ha
* Jaden Lee
