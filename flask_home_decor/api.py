import numpy as np
import pandas as pd
import tables
import dill
import pickle
import keras
import itertools
import re

# image processing
from skimage.io import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# keras libs
from keras.models import Model
from keras.applications import VGG16
from keras import models, layers, optimizers
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

c_b = {}
example = 'rustic wood wooden distressed pine oak reclaimed coffee table natural'
# example = {
#   'Excerpt': '',  # no excerpt given from user
#   }

# Helper Functions
def get_input(img_path):
    size = 224
    img = imread(img_path)
    img = resize(img, (size, size,3), preserve_range=True).astype(np.float32)
    img = preprocess_input(img)
    return img

def find_product(text):
    #text=features['description']
    if (text =='Enter description(optional)'):
        #print('#######nothing here')
        text =""

    #Load image recommender data and image_model and extract
    df_image = pd.read_csv('/Users/user/Documents/Data_Science/Projects/flask_home_decor/data/recommender_matrix.csv')
    df_image = df_image.drop(columns = ['Unnamed: 0'])
    my_model=models.load_model('/Users/user/Documents/Data_Science/Projects/flask_home_decor/data/final_image_model.hdf5')
    layer_to_extract = 'fc1'
    neurons = 128
    my_model_extract = Model(inputs=my_model.input, outputs=my_model.get_layer(layer_to_extract).output)

    #Load NLP and LDA models and data
    nlp_data = pd.read_csv('/Users/user/Documents/Data_Science/Projects/flask_home_decor/data/nlp_matrix.csv')
    nlp_data = nlp_data.drop(columns = 'Unnamed: 0')

    vectorizer = dill.load(open('/Users/user/Documents/Data_Science/Projects/flask_home_decor/data/nlp_tf_vectorizer', 'rb'))
    lda_model = dill.load(open('/Users/user/Documents/Data_Science/Projects/flask_home_decor/data/nlp_lda','rb'))

    # Process user excerpt if it exists and get feature vector
    if(text!="" ):
        user_text = pd.DataFrame(columns=['excerpt'])
        user_text=user_text.append(pd.DataFrame([text], columns=['excerpt']),ignore_index=True)
        vector = vectorizer.transform(user_text['excerpt'])
        topic_vector = lda_model.transform(vector)
        user_excerpt = pd.DataFrame(topic_vector, columns=['topic_'+ str(i)for i in range(1,8)])
        user_excerpt=user_excerpt.apply(np.log)

    # Process image and get feature vector
    # Assumed location where uploaded image gets stored
    pic_link = '/Users/user/Documents/Data_Science/Projects/flask_home_decor/user_image'
    user_img_matrix=get_input(pic_link)
    user_img_matrix = np.expand_dims(user_img_matrix, axis=0)
    user_data = my_model_extract.predict(user_img_matrix)
    search_in = np.array(df_image.loc[:,'ifeature1':])

    # Image similarity
    metric = 'euclid'
    if(metric == 'euclid'):
        from sklearn.metrics.pairwise import euclidean_distances
        results= euclidean_distances(search_in, user_data)
        show_me_image = pd.DataFrame(results).sort_values(0, ascending=True).head(15)

    if(metric == 'cosine'):
        from sklearn.metrics.pairwise import cosine_similarity
        results = cosine_similarity(search_in, user_data)
        show_me_image = pd.DataFrame(results).sort_values(0, ascending=False).head(15)

    if(metric == 'manhat'):
        from sklearn.metrics.pairwise import manhattan_distances
        results = manhattan_distances(search_in, user_data)
        show_me_image = pd.DataFrame(results).sort_values(0, ascending=True).head(15)

    # text similarity
    if(text==""): # no text by user
        recommend_image = df_image.iloc[list(show_me_image.index),0:6].values
        #print('#################')
    else:
        given_excerpt = np.array(user_excerpt)
        search_in_nlp = np.array(nlp_data.iloc[list(show_me_image.index),4:])
        from sklearn.metrics.pairwise import cosine_similarity
        results2 = cosine_similarity(search_in_nlp, given_excerpt)
        show_me_nlp = pd.DataFrame(results2).sort_values(0, ascending=False).head(6)
        final_list = list(show_me_image.iloc[list(show_me_nlp.index),:].index)
        recommend_image = df_image.iloc[final_list,0:6].values
        #print('$$$$$$$$$$$$$$$$')

    # construct the response dictionary image_link-pagelink
    send_result = {
        'image1': recommend_image[0][0],
        'page1': recommend_image[0][2],
        'image2': recommend_image[1][0],
        'page2': recommend_image[1][2],
        'image3': recommend_image[2][0],
        'page3': recommend_image[2][2],
        'image4': recommend_image[3][0],
        'page4': recommend_image[3][2],
        'image5': recommend_image[4][0],
        'page5': recommend_image[4][2],
        'image6': recommend_image[5][0],
        'page6': recommend_image[5][2],
    }

    return send_result


if __name__ == '__main__':
    print(find_product(example))
