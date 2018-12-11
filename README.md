# Home Decor Helper 

### Project Kojak

> #### Data
> **36,000** product images ans text descriptions from overstock.com
>
> **Selenium** used to webscrape 



> #### Tools
>
> 1. Selenium
> 2. Numpy
> 3. Pandas
> 4. Matplotlib
> 5. Seaborn
> 6. Flask
> 7. Javascript
> 8. HTML
> 9. CSS
> 10. NLTK
> 11. PyLDAvis
> 12. Sklearn
> 13. Scipy
> 14. Keras
> 15. Tensorflow
> 16. Powerpoint
> 17. Typora
> 18. Jupyter Notebooks



> #### Algorithms
>
> 1. Porter Stemmer, Snowball Stemmer , Wordnet Lemmatizer
> 2. CountVectorizer - Bag-of-words
> 3. Latent Dirichlet Allocation
> 4. Convolutional Neural Networks
> 5. VGG16, ResNet50, VGG 19
> 6. Euclidean distance
> 7. Cosine Similarity



> #### Modules 
>
> 1. **data_collection** - contains notebooks for web scraping from overstock.com
>
> 2. **code** - contains all main notebooks for modelling and runnin gthe application 
>
>    1. **data_clean_join.ipynb** 
>
>       Opens all the the scraped .csv files for each category cleans and then joins to create the model_data.csv and recommender_data.csv
>
>    2. **model_cnn.ipynb**
>
>       This notebook pre-processes all images used to model and sets up the CNN and trains the CNN. Output is the best image classification model.
>
>    3. **model_nlp.ipynb**
>
>       Creates a bag-of-words model from the text description and used LDA topic modelling to create text analysis features. 
>
>    4. **recommender_image_preprocess.ipynb**
>
>       It is not possible pre-process all 36,000 images so this notebook breaks it down into 4 parts and generates the image matrices. 
>
>    5. **recommender_feature_extraction.ipynb**
>
>       Load all 4 images matrices and extracts features for all product images using the convlutional neural network.
>
> 3. **data** - all stored data including dilled/hdf5 pipelines
>
>    1. **/model**
>
>       All core models, .csv used to run the recommender. 
>
>    2. **/samples**
>
>       samples of photos I have taken and images from crate and barrel used to test the recommender 
>
>    3. **/scrape**
>
>       file saved during the scrape, cleaning process 
>
> 4. **flask_home_decor**
>
>    1. This is a self contained module that runs my flask web application. has everything in one place so you can pretty much just download this and run things.
>
> 5. **docs** - documents 
>
>    1. Proposal - proposal_project_fletcher.pdf
>    2. Presentation -PPT_book_recommender .pdf
>    3. Summary - Summary_Book_recommender.pdf
>



> #### How to run ?
>
> 1. Run the following code notebooks - code/application.ipynb 
>
> 2. Run flask app -> 
>
>    cd flask_home_decor 
>
>    python app.py 
>
>    paste http://127.0.0.1:5000/static/index.html in browser.

