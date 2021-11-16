import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline

#import PIL
#from PIL import Image

import warnings
warnings.filterwarnings('ignore')

from joblib import load

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

from bokeh.layouts import column, row
from bokeh.plotting import figure #, show #curdoc
from bokeh.io import output_notebook, output_file, reset_output 
from bokeh.models import Select, MultiChoice, Toggle, Div, Slider, CheckboxGroup, RadioButtonGroup 
from bokeh.themes import Theme
from bokeh.embed import server_document
from bokeh.server.server import Server

from threading import Thread
from flask import Flask, render_template, request, redirect
from tornado.ioloop import IOLoop


app = Flask(__name__)


class ModelTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, model):
        self.predictor = model
            
    def fit(self, X, y):
        # Fit the stored predictor.
        self.predictor.fit(X, y)
        return self
    
    def transform(self, X):
        # Use predict on the stored predictor as a "transformation".
        # reshape(-1,1) is required to return a 2-D array which is expected by scikit-learn.
        return np.array(self.predictor.predict(X)).reshape(-1,1)
    

def stringlist_to_dict(stringlist):
        dict = {}
        for string in stringlist:
            dict[string]=1
        return dict
    
class DictEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X will be a pandas series of strings representing lists of strings. Return a pandas series of dictionaries
        return X.apply(eval).apply(stringlist_to_dict)
    

class TagsEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X will be a pandas series of strings representing lists of strings. Return a pandas series of single strings containing all individual strings joined by a space
        return X.apply(eval).apply(lambda stringlist:' '.join(stringlist))


def paintora_app(doc):
    
    min_price = 100.0
    
    #Loading data and models:
    
    #Let's start by loading the training and testing dataset (as computed last after removing duplicates) and precomputed neighbors indices for all test paintings 

    #Note: for deployment I needed to split the training data in 2 parts as it was going over the Github limits
    
    paintings_df_train_w_imageinfo_v1_filename_part1 = f'./App_data/paintings_from_USD{int(min_price)}_df_train_v1_part1.joblib'

    with open(paintings_df_train_w_imageinfo_v1_filename_part1, 'rb') as f:
        paintings_df_train_part1 = load(f)
        print(f'Loaded joblib file with PART 1 of dataframe of train paintings >USD{int(min_price)} used in the App v1.')
      
    paintings_df_train_w_imageinfo_v1_filename_part2 = f'./App_data/paintings_from_USD{int(min_price)}_df_train_v1_part2.joblib'

    with open(paintings_df_train_w_imageinfo_v1_filename_part2, 'rb') as f:
        paintings_df_train_part2 = load(f)
        print(f'Loaded joblib file with PART 2 of dataframe of train paintings >USD{int(min_price)} used in the App v1.')
    
    #To recombine the two parts of the training set:
    paintings_df_train = pd.concat([paintings_df_train_part1,paintings_df_train_part2])
       
#     #old version attempting to load all training data at once (works locally but not on Github)
    
#     paintings_df_train_w_imageinfo_v1_filename = f'./App_data/paintings_from_USD{int(min_price)}_df_train_v1.joblib'

#     with open(paintings_df_train_w_imageinfo_v1_filename, 'rb') as f:
#         paintings_df_train = load(f)
#         print(f'Loaded joblib file with dataframe of train paintings >USD{int(min_price)} used in the App v1.')
    
    
    paintings_df_test_w_imageinfo_v1_filename = f'./App_data/paintings_from_USD{int(min_price)}_df_test_v1.joblib'

    with open(paintings_df_test_w_imageinfo_v1_filename, 'rb') as f:
        paintings_df_test = load(f)
        print(f'Loaded joblib file with dataframe of test paintings >USD{int(min_price)} used in the App v1.')
        
    
    nearest_neighbors_indices_paintings_test_v1_filename = f'./App_data/nearestneighbors_paintings_from_USD{int(min_price)}_v1.joblib'
    
    with open(nearest_neighbors_indices_paintings_test_v1_filename, 'rb') as f:
        neigh_ind_test = load(f)
        print(f'Loaded joblib file with indices for 10 nearest neighbors for all test paintings >USD{int(min_price)} used in the App v1.')
    

    #For the price prediction, I will use the Blended linear model (random forest for numerical+categorical features, style tags, materials tags, other tags with linear blending). 
    #Let's load it (the required custom classes were defined above)

    blended_model_rforest_linear_filename = './App_data/blended_model_rforest_linear.joblib'

    with open(blended_model_rforest_linear_filename, 'rb') as f:
        blended_model_rforest_linear = load(f)
        print('Loaded joblib file with linear blended model of random forests for numerical+categorical variables, style tags, materials tags, other tags.')
    
    
    #Also load the KNN regressor model for finding nearest neighbors:
    knn_regressor_CVmodel_filename = './App_data/knn_CVmodel.joblib'

    with open(knn_regressor_CVmodel_filename, 'rb') as f:
        knn_CVmodel = load(f)
        print('Loaded joblib file with KNNregression model using numerical+categorical variables, style tags, materials tags, other tags.')
    
    
    #Here starts the actual PaintORA code:
    
    #starting_text = 'Starting PaintORA'
    #print(starting_text)
        
    num_listings_test = len(paintings_df_test)
    
    #initialize Numpy's random generator
    np.random.seed(seed=100)
    
    #I initialize with the seventh random painting which is a nice mountain picture
    for i in range(0,6):
        _ = np.random.randint(num_listings_test)
    index_to_show = np.random.randint(num_listings_test)
    
    new_painting_df = paintings_df_test.iloc[[index_to_show]]
    predicted_price = blended_model_rforest_linear.predict(new_painting_df)[0]
    
    #I already precomputed the nearest neighbors for each painting in the test set, so here I just need to recall the right row
    nneighbor_indices_to_show = neigh_ind_test[index_to_show,:]
    
    #Only if I decide to recompute the neighbors (after changing some parameters, and so on) I will have to make a new calculation 
    
    def create_neighbor_1_figure(nneighbor_index_to_show):
    
        title_1 = 'Comparable #1: $' + str(round(paintings_df_train.iloc[nneighbor_index_to_show]['price'],2))
    
        p1 = figure(title = title_1,width=150, height=200, x_range=(0,150),y_range=(0,200),min_border=0,toolbar_location = None, x_axis_type=None, y_axis_type=None)
        
        p1.image_url(url=[paintings_df_train.iloc[nneighbor_index_to_show]['image_url_fullxfull']], x=75, y=100, w=150, h=200, anchor='center') 
    
        return p1
    
    def create_neighbor_2_figure(nneighbor_index_to_show):

    
        title_2 = 'Comparable #2: $' + str(round(paintings_df_train.iloc[nneighbor_index_to_show]['price'],2))
    
        p2 = figure(title = title_2,width=150, height=200, x_range=(0,150),y_range=(0,200),min_border=0,toolbar_location = None, x_axis_type=None, y_axis_type=None)
        
        p2.image_url(url=[paintings_df_train.iloc[nneighbor_index_to_show]['image_url_fullxfull']], x=75, y=100, w=150, h=200, anchor='center') 
        
        return p2
    
    
    def create_neighbor_3_figure(nneighbor_index_to_show):
    
        title_3 = 'Comparable #3: $' + str(round(paintings_df_train.iloc[nneighbor_index_to_show]['price'],2))
    
        p3 = figure(title = title_3,width=150, height=200, x_range=(0,150),y_range=(0,200),min_border=0,toolbar_location = None, x_axis_type=None, y_axis_type=None)
        
        p3.image_url(url=[paintings_df_train.iloc[nneighbor_index_to_show]['image_url_fullxfull']], x=75, y=100, w=150, h=200, anchor='center') 
        
        return p3
    
    
    def create_listing_figure(index_to_show,from_test_set=True):
        
        if from_test_set:
            price_string = str(round(paintings_df_test.iloc[index_to_show]['price'],2))
        else:
            price_string = str(round(paintings_df_train.iloc[index_to_show]['price'],2))
        
        main_fig_title = 'Closest painting based on features (price range $100-$650):                 $' + price_string  #+ new_painting_df.iloc[0]['url']
    
        p = figure(title = main_fig_title, width=500, height=600, x_range=(0,500),y_range=(0,600),min_border=0,toolbar_location = None, x_axis_type=None, y_axis_type=None)
        
        if from_test_set:
            p.image_url(url=[paintings_df_test.iloc[index_to_show]['image_url_fullxfull']], x=250, y=300, w=500, h=600, anchor='center') 
        else:
            p.image_url(url=[paintings_df_train.iloc[index_to_show]['image_url_fullxfull']], x=250, y=300, w=500, h=600, anchor='center') 
  
        return p


    def update_predictions_and_images(new_painting_df):
        
        predicted_price = blended_model_rforest_linear.predict(new_painting_df)[0]

        div_prediction.text=("<b>Predicted price on Etsy.com for a painting with the features specified on the left: $</b>"+ str(round(predicted_price,2)))
        
        features_paintings_df_new = knn_CVmodel.best_estimator_['all scaled features'].transform(new_painting_df) 
    
        nneighbor_indices_to_show = knn_CVmodel.best_estimator_['knn'].kneighbors(features_paintings_df_new,4,return_distance=False)[0]
         
        layout.children[1] = column(create_listing_figure(nneighbor_indices_to_show[0],from_test_set=False), div_prediction, width=500, height=700)
        
        layout.children[2] = column(create_neighbor_1_figure(nneighbor_indices_to_show[1]), create_neighbor_2_figure(nneighbor_indices_to_show[2]), \
                                        create_neighbor_3_figure(nneighbor_indices_to_show[3]), toggle_reinitialize, width=200, height=700, margin=(0,0,0,50))

    
    def restart_with_new_index(status):
        if status==True:
  
            index_to_show = np.random.randint(num_listings_test)
    
            new_painting_df = paintings_df_test.iloc[[index_to_show]]
            predicted_price = blended_model_rforest_linear.predict(new_painting_df)[0]
            div_prediction.text=("<b>Predicted price on Etsy.com for a painting with the features specified on the left: $</b>"+ str(round(predicted_price,2)))
        
            #I already precomputed the nearest neighbors for each painting in the test set, so here I just need to recall the right row
            nneighbor_indices_to_show = neigh_ind_test[index_to_show,:]

            features_paintings_df_new = knn_CVmodel.best_estimator_['all scaled features'].transform(new_painting_df) 
    
            nneighbor_indices_to_show = knn_CVmodel.best_estimator_['knn'].kneighbors(features_paintings_df_new,3,return_distance=False)[0]
            
            when_made_select.value =new_painting_df.iloc[0]['when_made']
            types_select.value=new_painting_df.iloc[0]['painting_type']
            
            style_tags_multi_choice.value=eval(new_painting_df.iloc[0]['style_tags_new'])

            materials_tags_multi_choice.value=eval(new_painting_df.iloc[0]['materials_tags_new'])
            
            other_tags_multi_choice.value=eval(new_painting_df.iloc[0]['tags_new'])
    
            max_dim_slider.value=new_painting_df.iloc[0]['max_dimension']
    
            min_dim_slider.value=new_painting_df.iloc[0]['area']/new_painting_df.iloc[0]['max_dimension']
    
            if new_painting_df.iloc[0]['made_by_seller']:
                checkbox_group_madebyseller.active=[0]
            else:
                checkbox_group_madebyseller.active=[]           
            
            layout.children[1] = column(create_listing_figure(index_to_show,from_test_set=True), div_prediction, width=500, height=700)
            
            layout.children[2] = column(create_neighbor_1_figure(nneighbor_indices_to_show[0]), create_neighbor_2_figure(nneighbor_indices_to_show[1]), \
                                        create_neighbor_3_figure(nneighbor_indices_to_show[2]), toggle_reinitialize, width=200, height=700, margin=(0,0,0,50))
                 
 
    def update_max_dim(attr, old, new):
        min_dimension = new_painting_df.iloc[0]['area']/new_painting_df.iloc[0]['max_dimension']
        if new>=min_dimension:
            new_painting_df.loc[new_painting_df.index[0],'max_dimension']=new
            new_painting_df.loc[new_painting_df.index[0],'area']=new*min_dimension
            new_painting_df.loc[new_painting_df.index[0],'aspect_ratio']=new/min_dimension
        else:
            new_painting_df.loc[new_painting_df.index[0],'max_dimension']=min_dimension
            new_painting_df.loc[new_painting_df.index[0],'area']=new*min_dimension
            new_painting_df.loc[new_painting_df.index[0],'aspect_ratio']=min_dimension/new
        update_predictions_and_images(new_painting_df)

    def update_min_dim(attr, old, new):
        max_dimension = new_painting_df.iloc[0]['max_dimension']
        if new<=max_dimension:
            new_painting_df.loc[new_painting_df.index[0],'area']=new*max_dimension
            new_painting_df.loc[new_painting_df.index[0],'aspect_ratio']=max_dimension/new
        else:
            new_painting_df.loc[new_painting_df.index[0],'max_dimension']=new
            new_painting_df.loc[new_painting_df.index[0],'area']=new*max_dimension
            new_painting_df.loc[new_painting_df.index[0],'aspect_ratio']=new/max_dimension
        update_predictions_and_images(new_painting_df)
        
    def update_type(attr, old, new):
        avail_types = ['acrylic', 'oil', 'watercolor', 'more_than_one']
        new_painting_df.loc[new_painting_df.index[0],'painting_type']=new
        update_predictions_and_images(new_painting_df)
        
    def update_when_made(attr, old, new):
        new_painting_df.loc[new_painting_df.index[0],'when_made']=new
        predicted_price = blended_model_rforest_linear.predict(new_painting_df)[0]
        update_predictions_and_images(new_painting_df)
        
        
    def update_madebyseller(attr, old, new):
        if new==[0]:
            new_painting_df.loc[new_painting_df.index[0],'made_by_seller']=True
            predicted_price = blended_model_rforest_linear.predict(new_painting_df)[0]
            update_predictions_and_images(new_painting_df)
        elif new==[]:
            new_painting_df.loc[new_painting_df.index[0],'made_by_seller']=False
            predicted_price = blended_model_rforest_linear.predict(new_painting_df)[0]
            update_predictions_and_images(new_painting_df)
    
    def update_style_tags_list(attr, old, new):
        new_painting_df.loc[new_painting_df.index[0],'style_tags_new']=repr(new)
        predicted_price = blended_model_rforest_linear.predict(new_painting_df)[0]
        update_predictions_and_images(new_painting_df)
        
    def update_materials_tags_list(attr, old, new):
        new_painting_df.loc[new_painting_df.index[0],'materials_tags_new']=repr(new)
        predicted_price = blended_model_rforest_linear.predict(new_painting_df)[0]
        update_predictions_and_images(new_painting_df)
      
    def update_other_tags_list(attr, old, new):
        new_painting_df.loc[new_painting_df.index[0],'tags_new']=repr(new)
        predicted_price = blended_model_rforest_linear.predict(new_painting_df)[0]
        update_predictions_and_images(new_painting_df)
 
        
    #Initialize the control widgets
    
    avail_when_made = ['made_2020s', 'made_2010s', 'made_before_2010', 'made_to_order']
    
    when_made_select = Select(title='When was it made:', value=new_painting_df.iloc[0]['when_made'], options=avail_when_made, margin=(10,20,10,0)) 
    when_made_select.on_change('value',update_when_made)

    
    avail_types = ['acrylic', 'oil', 'watercolor', 'more_than_one']
    types_select = Select(title='Type:', value=new_painting_df.iloc[0]['painting_type'], options=avail_types, margin=(10,20,10,0)) 
    types_select.on_change('value',update_type)


    all_style_tags = ['abstract','african','american','asian','athletic','automobilia','avant','beach','boho','burlesque','century','chic','coastal',\
                      'contemporary','cottage','country','deco','edwardian','expressionism','fantasy','fashion','floral','folk','goth',\
                      'hippie','hipster','historical','hollywood','impressionism','industrial','kawaii','kitsch','landscape','mcm','mediterranean',\
                      'military','minimalism','mod','modern','modernism','nautical','neoclassical','nouveau','photorealism','pop','portrait','primitive',\
                      'realism','regency','renaissance','resort','retro','rocker','rustic','sci','southwestern','spooky','steampunk','traditional',\
                      'tribal','victorian','vintage','western','whimsical','woodland','zen'] #,'cовременный','неоклассический','традиционный']
                      #note: I removed three Russian styles and styles that occurred in less than 50 paintings 
        
    style_tags_multi_choice = MultiChoice(title='Style: ', options=all_style_tags, value=eval(new_painting_df.iloc[0]['style_tags_new']), margin=(10,20,30,0), height=120) #margin=(10,20,30,0), height=150
    style_tags_multi_choice.on_change("value", update_style_tags_list) 
        
    
    all_materials_tags = ['aluminum','canvas','cardboard','fabric','framed','hardboard','metal','glass','paper','wood','unframed',\
                         'collage','charcoal','gel','gesso','glitter','gloss','gouache','graphite','ink','matte','pastel','pen','pencil','print','spray',\
                         'black','blue','gold','green','metallic','purple','red','white','yellow']
                         
                         #words that were part of the vocabulary but did not include in the selection:
                         #'140','acid','acrylic','acrylics','arches','archival','birch','board','brush','brushes',\
                         #'clear','cold','cotton','epoxy','fine','finish','floetrol','frame',\
                         #'grade','hand','hang','hanger','hanging','hardware','heavy','knife',\
                         #'lb','leaf','linen','liquitex','love','marker','mat','media','mixed','newton','oil','oils','painted','palette',\
                         #'panel','paste','pigment','pigments','premium','press','pressed','protective','resin','satin',\
                         #,'stretched','stretcher','texture','thick','uv','varnish','water','watercolor','watercolors','winsor',\
                         # 'wire','wooden','wrap','wrapped']
                            
                         #This was the full list:
                         #['140','acid','acrylic','acrylics','aluminum','arches','archival','birch','black','blue','board','brush','brushes','canvas','cardboard',\
                         #'charcoal','clear','cold','collage','cotton','epoxy','fabric','fine','finish','floetrol','frame','framed','gel','gesso','glass','glitter',\
                         #'gloss','gold','golden','gouache','grade','graphite','green','hand','hang','hanger','hanging','hardboard','hardware','heavy','ink','knife',\
                         #'lb','leaf','linen','liquitex','love','marker','mat','matte','media','metal','metallic','mixed','newton','oil','oils','painted','palette',\
                         #'panel','paper','paste','pastel','pen','pencil','pigment','pigments','premium','press','pressed','print','protective','purple','red','resin',\
                         #'satin','spray','stretched','stretcher','texture','thick','unframed','uv','varnish','water','watercolor','watercolors','white','winsor',\
                         # 'wire','wood','wooden','wrap','wrapped','yellow']
                            
    materials_tags_multi_choice = MultiChoice(title='Materials and Colors: ', options=all_materials_tags, value=eval(new_painting_df.iloc[0]['materials_tags_new']), margin=(10,20,30,0), height=100) #150 
    materials_tags_multi_choice.on_change("value", update_materials_tags_list) 
    
     
#     #Here is the whole list of the most common other tags and bigrams, with the frequency they were found in the training set. From these I choice a few common/representative themes.
#     {'nature': 3514,'flowers': 3285,'ocean': 3256,'flower': 2572,'pink': 2570,'trees': 2299,'life': 2230,'signed': 2150,'woman': 2073,'house': 1998,'pour': 1787,'tree': 1720,\
#      'still': 1719,'orange': 1699,'seascape': 1674,'fluid': 1663,'sunset': 1634,'pet': 1588,'living': 1565,'kind': 1531,'animal': 1524,'office': 1517,'bright': 1425,'sky': 1399,\
#       'dog': 1385,'mountain': 1384,'rectangle': 1283,'sea': 1281,'air': 1263,'textured': 1232,'forest': 1220,'plein': 1197,'day': 1188,'mountains': 1175,'antique': 1155,\
#      'square': 1103,'girl': 1071,'scene': 1060,'decoration': 1054,'lake': 1037, 'christmas': 1006,'female': 979,'cat': 978,'housewarming': 959,'geometric': 945, 'bird': 942,\
#       'fall': 909, 'winter': 909, 'river': 905, 'farm': 899, 'beautiful': 887, 'her': 884, 'california': 882, 'garden': 846, 'bedroom': 835, 'brown': 833, 'gray': 830,\
#       'vibrant': 825, 'summer': 809, 'birthday': 807, 'dark': 807, 'photo': 788, 'tropical': 781, 'lover': 778, 'figure': 764, 'wedding': 756, 'nude': 749, 'autumn': 745,\
#       'clouds': 745, 'farmhouse': 740, 'spring': 731, 'waves': 721, 'impasto': 720, 'kitchen': 709, 'rainbow': 704, 'nursery': 701, 'personalized': 694, 'desert': 659,\
#      'family': 657, 'city': 643, 'southwest': 621, 'work': 616, 'anniversary': 609, 'snow': 607, 'scenery': 606, 'botanical': 603, 'bold': 592, 'horse': 592, 'pouring': 592,\
#      'minimal': 577, 'him': 576, 'ocean seascape': 569, 'man': 568, 'light': 560, 'surreal': 558, 'baby': 546, 'figurative': 546, 'barn': 544, 'surrealism': 544, \
#      'psychedelic': 542, 'cityscape': 535, 'teal': 529, 'fun': 524, 'mother': 523, 'music': 523, 'turquoise': 518, 'field': 517, 'moon': 512, 'street': 511, 'french': 508,\
#       'boat': 501, 'space': 497, 'women': 488, 'rose': 486, 'sunrise': 486, 'silver': 484, 'outsider': 483, '3d': 481, 'night': 468, 'mom': 467, 'face': 463, 'portraits': 463,\
#       'idea': 462, 'mexico': 461, 'grey': 455, 'birds': 452, 'coast': 448, 'roses': 447, 'fauvism': 439, 'commission': 437, 'prints': 435, 'wildlife': 432, 'cabin': 420,\
#      'animals': 413, 'kids': 407, 'child': 400, 'spiritual': 398, 'park': 397, 'neutral': 396, 'island': 394, 'ideas': 392, 'children': 389, 'peaceful': 389, 'decorative': 388,\
#       'rock': 388, 'shipping': 388, 'bouquet': 381, '16x20': 379, 'florida': 379, 'rectangular': 378, 'piece': 376, 'sand': 376, 'wave': 375, 'maine': 373, 'architecture': 370,\
#      'collectibles': 367, 'holiday': 366, 'cute': 365, 'indian': 365, 'happy': 364, 'colorado': 361, 'bohemian': 356, 'fish': 355, 'mothers': 354, 'midcentury': 352, 'gothic': 347,\
#       'leaves': 347, 'décor': 345, 'sun': 345, 'halloween': 344, 'vase': 344, 'handpainted': 337, 'memorial': 337, 'ship': 337, 'collectible': 334, 'horizontal': 331, 'lady': 329,\
#       'urban': 328, 'trippy': 326, 'fruit': 325, 'dutch': 321, 'rural': 320, 'wild': 319, 'england': 317, 'order': 315, 'palm': 315, 'inspired': 308, 'people': 307, 'woods': 307,\
#       'vertical': 303, 'native': 298, 'horror': 293, 'earth': 292, 'poster': 289, 'natural': 284, 'york': 280, 'heart': 279, 'rocks': 279, 'smith': 278, '20': 277, 'studio': 276,\
#       'eclectic': 273, 'pine': 273,'view': 272,'waterfall': 271,'cloud': 270,'neon': 270, 'expressive': 268,'fishing': 268,'sunset ocean': 265,'female woman': 264, 'graffiti': 263,\
#      'cool': 262, 'organic': 262,'pacific': 262,'cat pet': 260, 'expression': 260,'valley': 259,'accent': 258, 'plants': 258,'pond': 258,'lovers': 256,'butterfly': 255,\
#      'line': 255,'warming': 253,'16': 252,'set': 251,'shabby': 251,'texas': 251,'romantic': 249,'miniature': 244,'boats': 243,'oregon': 243,'sailboat': 239,'santa': 239,\
#      'inches': 238,'classic': 237,'galaxy': 237,'multi': 237,'west': 237,'shapes': 235,'24': 234,'food': 234,'flow': 233,'watercolour': 233,'hair': 232,'plant': 232,'bob': 230,\
#      'european': 229,'sketch': 229,'stars': 229,'beige': 228,'12': 227,'nature rectangle': 225,'signed plein': 225,'statement': 225,'japanese': 224,'magic': 224,'rocky': 224,\
#      'dog cat': 223,'beauty': 222,'eye': 222,'underwater': 221,'ross': 220,'grass': 219,'shore': 219,'road': 218,'couple': 217,'landscapes': 217,'male': 217,'village': 217,\
#      'italy': 216,'fire': 214,'eyes': 212,'pattern': 212,'11x14': 210,'north': 210,'sunflower': 210,'outdoor': 209,'south': 209,'bridge': 208,'skull': 208,'oversized': 207,\
#      'size': 207,'best': 206,'body': 205,'bathroom': 204,'hawaii': 203,'head': 203,'northwest': 203,'religious': 203,'soft': 203,'storm': 203,'stream': 203,'study': 203,\
#      'feminine': 202,'hills': 202,'1960s': 201,'countryside': 201,'lighthouse': 201,'young': 201,'inspirational': 200,'sailing': 200,'wax': 200,'flower garden': 198,\
#      'self': 198,'weird': 198,'3d prints': 197,'calm': 197,'southern': 197,'cartoon': 196,'copper': 196,'scenic': 196,'surf': 196,'graphic': 195,'warm': 195,'aspen': 194,\
#      'national': 194,'van': 194,'geode': 193,'harbor': 193,'pallet': 193,'cactus': 192,'lavender': 191,'bay': 190,'chinese': 190,'girls': 190,'20th': 189,'star': 189,\
#      'blues': 188,'dad': 186,'life still': 186,'elegant': 185,'lines': 185,'meditation': 185,'seaside': 185'dining': 184,'local': 184,'oilpainting': 184,'vivid': 184,\
#      'realtor': 183,'cheap': 182,'mexican': 182,'creative': 181,'sculpture': 181,'vacation': 181,'funky': 180,'guitar': 180,'india': 180,'abstraction': 179,'carolina': 179,\
#      'father': 179,'flowers vase': 179,'building': 178,'hand': 178,'arizona': 177,'aqua': 176,'commissioned': 176,'marsh': 176,'woman nude': 176,'hangings': 175,'paris': 175,\
#      'plein rectangle': 175,'works': 175,'simple': 174,'cow': 173,'creepy': 173,'diego': 173,'fairy': 173,'fan': 173,'above': 172,'italian': 172, 'sexy': 172,'outdoors': 171,\
#      '18x24': 170,'boy': 170,'poppies': 170,'lily': 169,'wine': 169,'jackson': 168,'travel': 167,'christian': 166,'gogh': 165,'pets': 165,'8x10': 164,'cherry': 164,'cubism': 164,\
#      'rain': 164,'triptych': 164,'valentine': 164,'cave': 163,'france': 163,'intuitive': 163,'mermaid': 162,'ooak': 162,'pollock': 162,'theme': 162,'moody': 161,'1970s': 160,\
#      'culture': 160,'gay': 160,'pastels': 160,'reflection': 160,'cowboy': 159,'hot': 159,'creek': 158,'loss': 158,'drip': 157,'naive': 157,'11': 156,'meadow': 156, 'men': 156,\
#      'recycled': 156,'wildflowers': 156,'horses': 155,'prima': 154,'liquid': 153,'matted': 153,'ralph': 153,'silhouette': 153,'dog pet': 152,'john': 152,'lgbtq': 151,\
#      'musician': 151,'round': 151,'wave ocean': 151,'deep': 150,'stone': 150,'wilderness': 150,'alla': 149,'angel': 149,'erotic': 149,'fields': 149,'sunflowers': 149,\
#      'time': 149,'plein nature': 148,'collector': 147,'jill': 147,'sign': 147,'splatter': 147,'tones': 147,'10': 146,'14': 146,'loose': 146,'circle': 145,'movie': 145,\
#      'museum': 145,'navy': 145,'nursery baby': 145,'valentines': 145,'wallart': 145,'deer': 144,'energy': 144,'owl': 144,'cape': 143,'listed': 143,'magenta': 143,\
#      'serene': 143,'snowy': 143,'bed': 142,'feminism': 142,'houses': 142,'housewarming house': 142,'marine': 142,'violet': 142,'celestial': 141,'chicago': 141,'church': 141,\
#      'magical': 141,'peace': 141,'calming': 140,'shop': 140,'under': 140,'alcohol': 139,'aquarelle': 139,'blossom': 139,'path': 139,'pet lover': 139,'collection': 138,\
#      'dorm': 138,'dress': 138,'reflections': 138,'book': 137,'bright flowers': 137,'goddess': 137,'puppy': 137,'world': 137,'equestrian': 136,'monet': 136,'rectangle fauvism': 136,\
#      'unstretched': 136,'county': 135,'dream': 135,'flowers flower': 135,'history': 135,'mark': 135,'mountains mountain': 135,'multicolor': 135,'owned': 135,'bouquet flowers': 134,\
#      'handprint': 134,'special': 134,'dye': 133,'nature mountains': 133,'poppy': 133,'coral': 132,'girl woman': 132,'graphic handprint': 132,'hiking': 132,'michigan': 132,\
#      'prints graphic': 132,'1980s': 131,'blossoms': 131,'architectural': 130,'human': 130,'mystical': 130,'sea ocean': 130,'cats': 129,'circles': 129,'decorations': 129,\
#      'flower still': 129,'funny': 129,'music guitar': 129,'peony': 129,'full': 128,'movement': 128,'portraiture': 128,'peonies': 127,'tie': 127,'friend': 126,'hill': 126,\
#      'luxury': 126,'morning': 126,'queen': 126,'usa': 126,'famous': 125,'little': 125,'series': 125,'bear': 124,'spirit': 124,'celebrity': 123,'century': 123,'masonite': 123,\
#      'ranch': 123,'real': 123,'buy': 122,'canyon': 122,'fathers': 122,'town': 122,'1950s': 121,'prairie': 121,'chicken': 120,'fluid pour': 120,'historic': 120,'painterly': 120,\
#      '140lb': 119,'africa': 119,'oriental': 119,'equine': 118,'flower pink': 118,'pride': 118,'state': 118,'crystal': 117,'emerging': 117,'flowers spring': 117,'foliage': 117,\
#      'glow': 117,'healing': 117,'scape': 117,'german': 116'rooster': 116,'wife': 116,'jillkrutickfineart': 115,'universe': 115,'bronze': 114,'cosmic': 114,'dancer': 114,\
#      'dogs': 114,'dreamy': 114,'first': 114,'jillkrutickfineart underwater': 114,'king': 114,'outer': 114,'yupo': 114,'caribbean': 113,'vibrant jillkrutickfineart': 113,\
#      'washington': 113,'bar': 112,'ice': 112,'kid': 112,'knives': 112,'virginia': 112,'apartment': 111,'cream': 111,'item': 111,'krutick': 111,'picasso': 111,'18': 110,\
#      'aesthetic': 110,'nyc': 110,'bars': 109,'buildings': 109,'dye krutick': 109,'iris': 109,'strange': 109,'car': 108,'jill dye': 108,'materials': 108,'tan': 108,\
#      'underwater tie': 108,'window': 108,'america': 107,'giclee': 107,'huge': 107,'or': 107,'post': 107,'alien': 106,'diptych': 106,'dirty': 106,'lee': 106,'mandala': 106,\
#      'utah': 106,'background': 105,'business': 105,'english': 105,'inspiration': 105,'mini': 105,'ocean waves': 105,'peach': 105,'point': 105,'studios': 105,'closing': 104,\
#      'crystals': 104,'dots': 104,'hat': 104,'pets pet': 104,'seashore': 104,'visionary': 104,'winter scene': 104,'daisy': 103,'entryway': 103,'non': 103,'photo commission': 103,\
#      'russian': 103,'skyline': 103,'9x12': 102,'calligraphy': 102,'land': 102,'lowbrow': 102,'nature mountain': 102,'sacred': 102,'wash': 102,'classical': 101,'coffee': 101,\
#      'evening': 101,'flying': 101,'grand': 101,'tree forest': 101,'your': 101,'30': 100,'apple': 100,'bloom': 100,'dawn': 100,'elephant': 100,'flag': 100,'shower': 100,\
#      'sunny': 100,'tattoo': 100}
    
      
    all_other_tags = ['nature','flowers','ocean','trees','mountains','sky','river','farm',\
                      'woman','man','children','dog','cat','bird','animal','wildlife',\
                       'spring','summer','fall','winter','christmas','tropical','cityscape',\
                       'geometric','scene','decoration','photo','minimal','figurative','portraits','architecture',\
                      'pour','fluid','antique','textured','collectibles','urban','bright','dark',\
                       'office','nursery','kitchen','bedroom']
    
    other_tags_multi_choice = MultiChoice(title='Other Tags: ', options=all_other_tags, value=eval(new_painting_df.iloc[0]['tags_new']), margin=(10,20,30,0), height=120) #margin=(10,20,30,0), height=150 
    other_tags_multi_choice.on_change("value", update_other_tags_list) 
    
    
    max_dim_slider = Slider(start=10, end=70, value=new_painting_df.iloc[0]['max_dimension'], step=1, title="Dimension 1 [in]")
    max_dim_slider.on_change("value", update_max_dim) 

    
    min_dim_slider = Slider(start=10, end=70, value=new_painting_df.iloc[0]['area']/new_painting_df.iloc[0]['max_dimension'], step=1, title="Dimension 2 [in]")
    min_dim_slider.on_change("value", update_min_dim) 
    
    labels_checkboxes = ['Made by seller']
    if new_painting_df.iloc[0]['made_by_seller']:
        active_checkboxes=[0]
    else:
        active_checkboxes = []
    checkbox_group_madebyseller = CheckboxGroup(labels=labels_checkboxes, active=active_checkboxes)
    checkbox_group_madebyseller.on_change('active',update_madebyseller)
    
    
    toggle_reinitialize = Toggle(label='Restart with new painting',active=False, margin=(10,20,20,0), width=150, height=75, width_policy='fixed', background='yellow') #margin=(50,20,30,0), background='black')
    toggle_reinitialize.on_click(restart_with_new_index)

    
    #This is the important text box showing the price prediction 
    div_prediction = Div(text=("<b>Predicted price on Etsy.com for a painting with the features specified on the left:  $</b>"+ str(round(predicted_price,2))), width=500, height=100,  width_policy='fixed', margin=(0,0,0,0), background='yellow') #height=50,
    
    #controls = column(max_dim_slider,min_dim_slider, types_select, when_made_select, toggle_madebyseller, style_tags_multi_choice, width=200, height=600, margin=(50,0,0,0))
    #controls = column(max_dim_slider,min_dim_slider, types_select, when_made_select, checkbox_group_madebyseller, style_tags_multi_choice, width=200, height=600, margin=(50,0,0,0))
    controls = column(max_dim_slider,min_dim_slider, types_select, when_made_select, checkbox_group_madebyseller, style_tags_multi_choice,materials_tags_multi_choice,other_tags_multi_choice,width=200, height=600, margin=(50,0,0,0))
        

    neighbors = column(create_neighbor_1_figure(nneighbor_indices_to_show[0]), create_neighbor_2_figure(nneighbor_indices_to_show[1]), \
                       create_neighbor_3_figure(nneighbor_indices_to_show[2]), toggle_reinitialize, width=200, height=700, margin=(0,0,0,50))
    
    painting_and_pred = column(create_listing_figure(index_to_show,from_test_set=True), div_prediction, width=500, height=700)
    
    layout = row(controls, painting_and_pred, neighbors) #row(controls, create_listing_figure(), neighbors) #create_neighbors_figure()) #, neighbors)
    doc.add_root(layout)
    #curdoc().add_root(layout)


@app.route('/', methods=['GET'])
def paintora_app_page():
    script = server_document('http://localhost:5006/paintora_app')
    return render_template('about_paintora_with_Github.html', script=script, template="Flask")
    #return render_template('about_paintora_basic.html', script=script, template="Flask")

def bk_worker():
    # Can't pass num_procs > 1 in this configuration. If you need to run multiple
    # processes, see e.g. flask_gunicorn_embed.py
    server = Server({'/paintora_app': paintora_app}, io_loop=IOLoop(), allow_websocket_origin=['127.0.0.1:8000']) #["localhost:8000"])
    server.start()
    server.io_loop.start()

Thread(target=bk_worker).start()

if __name__ == '__main__':
    print('Opening single process Flask app with embedded Bokeh application on http://localhost:8000/')
    print()
    print('Multiple connections may block the Bokeh app in this configuration!')
    print('See "flask_gunicorn_embed.py" for one way to run multi-process')
    app.run(port=8000)