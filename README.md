# PaintORA, The Oracle of Paintings

This project is for art makers and art lovers (like me!). 
It has two main goals:
- to help artists understand how different features of a painting affect its price on the market, so that they can better price their artwork
- to help art lovers find paintings on the market that match their taste and requirements

To achieve these goals, I analyzed thousands of paintings for sale on Etsy.com, the largest online marketplace for art in the US,
and leveraged machine learning techniques to derive a 'price oracle' algorithm and a 'comparable paintings' finder. 
Both algorithms exploit multi-modal information provided on Etsy.com listings, ranging from numerical (e.g. the dimensions) and categorical 
(e.g. the type of painting) to a variety of textual tags related to style, materials and colors, and more.

The results of this project can be freely accessed at:
https://paintora.herokuapp.com

## Overview of the project and repository files 

## Data Ingestion
- I used Etsy's Open API v3 (https://www.etsy.com/developers) to collect data on all paintings currently for sale on Etsy within the $100-$750 price range.
- I performed data cleaning, preprocessing and feature-engineering to have a reliable and balanced dataset to use for machine learning.
- The final dataset consists of ~70000 listings with full numerical and categorical information and at least one style-related textual tag.
- Four painting types (acrylic, oil, watercolor and mixed-type) and prices in the range $100-$650 are included in the final dataset.  
- Main tools: Pandas, Requests, Seaborn, some Scikit-Learn NLP features, and Python pickling.  
See the '*_data_downloading_*.ipynb' notebook for details. 

## Machine Learning 
- I tried a variety of linear and tree-based models to predict the price of a painting from individual feature types (numerical, categorical, textual tags of different types) or combinations of feature types. 
- The best results were obtained with a model blending four random-forests regressors using respectively: numerical+categorical features, style tags, materials and color tags, 'other' tags.
  Most tag types were processed with Tf-Idf and dimensionality reduction before feeding to the model.
- I trained a separate K Nearest-Neighbors model (working on dimensionality-reduced features) for finding comparable paintings to a given listing.
- Main tools: Scikit-Learn, especially model selection and pipeline tools, custom transformers, NLP feature-extraction tools, Linear Models, Tree-based models and KNN.  
See the '*_ml_models_*.ipynb' notebook for details.  

## Interative visualization of the results 
- I used Bokeh (https://bokeh.org) to create an app with multiple interactive widgets.
- The app is initialized by loading one of the 'test' listings (not used for training the machine learning models) and showing: its image, its listed and predicted price, its three closest neighbors in the 'train' set.
- A button is provided to allow the user to re-initialize the app with a different test listing (pseudo-random generator)
- These test listings are just provided as a starting point. The user can modify many of the features of the painting with multiple widgets, and receive price predictions and new comparable paintings for their custom requirements.
- When custom features are specified by the user (thus creating a new 'test' listing, likely never seen before by the models), then the painting showed in the main window is the closest one among those in the training set.
  In that case the 'comparable listings' are the 2nd, 3rd and 4th closest paintings in the training set.
- Main tools: Bokeh, especially widgets and image rendering from URL.   
See the '*_ml_models_*.ipynb' notebook for details.   

##App deployment
- I embedded the Bokeh app within a Flask app with a configuration that would work with Heroku.
  (https://github.com/bokeh/bokeh/blob/2.4.0/examples/howto/server_embed/flask_gunicorn_embed.py).
- I defined appropriate setting files (requirements.txt, Procfile and runtime.txt), created a Heroku app, and deployed to Heroku via Git.
See the '*_flask_gunicorn_embed_*.py' script for details.
There's also a '*_flask_embed_*.py' script used for local testing.   
(as in https://github.com/bokeh/bokeh/blob/2.4.0/examples/howto/server_embed/flask_embed.py)


_____________________________________________________________
Copyright &copy; 2021 Matteo Mischiati All Rights Reserved.