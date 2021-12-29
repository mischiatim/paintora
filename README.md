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

See the '_data_downloading_' notebook for details. 

## Machine Learning 
- I tried a variety of linear and tree-based models to predict the price of a painting from individual feature types (numerical, categorical, textual tags of different types) or combinations of feature types. 
- The best results were obtained with a model blending four random-forests regressors using respectively: numerical+categorical features, style tags, materials and color tags, 'other' tags.
  Most tag types were processed with Tf-Idf and dimensionality reduction before feeding to the model.
- I trained a separate K Nearest-Neighbors model (working on dimensionality-reduced features) for finding comparable paintings to a given listing.
- Main tools: Scikit-Learn, especially model selection and pipeline tools, custom transformers, NLP feature-extraction tools, Linear Models, Tree-based models and KNN.  

See the '_ml_models_' notebook for details.  

## Interactive visualization of the results 
- I used Bokeh (https://bokeh.org) to create an app with multiple interactive widgets.
- The app is initialized by loading one of the 'test' listings (not used for training the machine learning models) and showing: its image, its listed and predicted price, its three closest neighbors in the 'train' set.
- A button is provided to allow the user to re-initialize the app with a different test listing (pseudo-random generator)
- These test listings are just provided as a starting point. The user can modify many of the features of the painting with multiple widgets, and receive price predictions and new comparable paintings for their custom requirements.
- When custom features are specified by the user (thus creating a new 'test' listing, likely never seen before by the models), then the painting showed in the main window is the closest one among those in the training set.
  In that case the 'comparable listings' are the 2nd, 3rd and 4th closest paintings in the training set.
- Main tools: Bokeh, especially widgets and image rendering from URL.   

See the '_local_demo_.ipynb' notebook for details.   

## App deployment
- I  made a Bokeh server app with full Python callbacks (paintora_bokehapp_v1.py) and a Flask app that adds title and Github link to the Bokeh app (paintora_flaskapp_v1.py). I used Heroku cloud application services for hosting the Paintora app. 
- Combining Bokeh server app + Flask app is tricky on Heroku. I adopted the solution from the following repository:
https://github.com/hmanuel1/covid/tree/master/app/utest
(See also the discussion at https://discourse.bokeh.org/t/hosting-a-flask-bokeh-server-app-in-heroku/5490/8.)
This solution implements a HTTP reverse-proxy using Flask and a Web Socket reverse-proxy using Tornado.
- The combined Bokeh server app + Flask app is launched by the script run.py (see Procfile executed by Heroku).
The app can be tested locally by changing 'heroku' to 'local' in the first line of config.yaml, running 'python run.py' in a terminal and opening http://127.0.0.1:8000/ in an internet browser.
- I also created a version in which the Bokeh app is standalone without any Flask app: paintora_app_bokeh_standalone_v1,
following the example in the Bokeh documentation: https://github.com/bokeh/bokeh/blob/2.4.0/examples/howto/server_embed/standalone_embed.py.
This version can be run by calling 'bokeh serve --show paintora_app_bokeh_standalone_v1' for local testing, and replacing the content of 'Procfile' with the content of 'Procfile_bokeh_standalone' for execution in Heroku.
- Other versions of the code (paintora_app_flask_embed_v1.py, paintora_app_flask_gunicorn_embed_v1.py) work locally but not on Heroku.
- Main tools: Bokeh, Flask, Heroku cloud application services.   

See the 'paintora_bokehapp_v1.py', 'paintora_flaskapp_v1.py', 'run.py', 'config.py', 'wsproxy.py' scripts for details.
_____________________________________________________________
Copyright &copy; 2021 Matteo Mischiati All Rights Reserved.