### KICKOFF - CODING AN APP IN STREAMLIT

### import libraries
import pandas as pd
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# image scraping
from bs4 import BeautifulSoup
import requests
from IPython.display import Image

st.subheader('BrainStation Data Science Diploma Capstone Project')
st.write('Catherine Hocknell')


#######################################################################################################################################
### LAUNCHING THE APP ON THE LOCAL MACHINE
### 1. Save your *.py file (the file and the dataset should be in the same folder)
### 2. Open git bash (Windows) or Terminal (MAC) and navigate (cd) to the folder containing the *.py and *.csv files
### 3. Execute... streamlit run <name_of_file.py>
### 4. The app will launch in your browser. A 'Rerun' button will appear every time you SAVE an update in the *.py file


#######################################################################################################################################
### Create a title

st.title("Movie Recommendations for New Users")


#######################################################################################################################################
### SENTIMENT ANALYSIS

st.subheader("Predicting a Movie Rating:")

# A. Load and cache the model using joblib
@st.cache_resource
def load_model():
    model =  joblib.load('sentiment_pipe.pkl')
    return model

#define the model
model = load_model()

# B. Set up input field
text = st.text_input('Enter your review text below', 'Thanks, I hate it!')

# C. Use the model to predict sentiment & write result
prediction = model.predict({text})

# output predictions
if prediction == 1:
    st.markdown('This review is **:red[Fresh]**! :tomato:')
else:
    st.markdown('This review is **:orange[Rotten]**! :worried:')

#######################################################################################################################################
### CONTENT RECOMMENDER FUNCTION

# load and cache the dataframes
@st.cache_data
def load_data(link):
    df = pd.read_csv(link, index_col = 0)
    return df

# movie info
movies_df = load_data('movie_with_url.csv')
# vectorised keywords
count_matrix = load_data('content_vectoriser.csv')

# define and cache the movie recommendation function
@st.cache_data
def content_recommender(title, filt = 'Maximum Rating', vote_threshold=50, rotten = True) :
    
    """
    This function is used to run the content based recommender system.
    The purpose of this system is for the user to input the name of a movie, and the function will output the top 10 movies
    which have the highest similarity to the input, based on the keywords related to the given movie.
    
    Parameters:
    -----------
    
    title:  Movie Title
            Can be exact or just a small fraction of the movie title, which will be selected based on the filter provided
    filt:   Selection filter
            Either 'rating' or 'reviews'
            Defines the method used to select the film title if there are multiple results for the given input
            'ratings' = outputs the movie with the highest average rating
            'reviews' = outputs the movie with the highest number of reviews
    vote_threshold: Minimum number of reviews required for the movie to be counted
            Default = 50
            Only movies which have been reviewed more times that this input will be included in the recommendations
    rotten: Boolean input on whether Rotten movies should be included in the recommendations
            Default = True (Rotten movies will be included)
            
    Returns:
    -------
    
    top_movies: Top 10 movies most similar to selected input movie, based on keywords
    
    Other Outputs:
    --------
    
    Movie Title: Full title of movie defined as the input to the function
    
    Movie Description: Full description of the input movie
    
    Graph:  Graphical output of the similarity score for each of the predicted recommendations.
            Each output is coloured based on the overall classification score of the movie.
    
    """
        
    
    # Get the movie by the contents of the title
    # first find if there are any exact matches
    movie_exact = movies_df[(movies_df['movie_title'] == title) & (movies_df['tomatometer_count'] > vote_threshold)].index
    # then also find if there are options that contain the text from the input
    movie_options = movies_df[(movies_df['movie_title'].str.contains(title)) & (movies_df['tomatometer_count'] > vote_threshold)].index

    if filt == 'Maximum Rating':
        multi_filter = 'tomatometer_rating'
    else:
        multi_filter = 'tomatometer_count'

    
    # there can sometimes be movies with the same title, which needs to be assessed
    
    # need to check that the movie index is able to be found with the added vote_threshold added - if not, output error
    if (len(movie_exact) == 0) & (len(movie_options) == 0):
        return print('Movie not found - please check spelling or decrease the vote_threshold')
    # now check for an exact match, as this should be used over all others
    elif len(movie_exact) == 1:
        movie_index = movie_exact[0]
    # then check if there is only one option containing the input string
    elif len(movie_exact) > 1:
        exact_index = movies_df.iloc[movie_exact,:][[multi_filter]].values.argmax()
        movie_index = movie_exact[exact_index]    
    elif len(movie_options) == 1:
        movie_index = movie_options[0]        
    else:
        # if more than one movie, pick the one with the highest rating to compare against
        option_index = movies_df.iloc[movie_options,:][[multi_filter]].values.argmax()
        movie_index = movie_options[option_index]
    
    # define the full title of the move that is being compared
    movie_title = movies_df.iloc[movie_index,1]
    
    # define all ratings for given movie
    ratings = count_matrix.loc[movie_index,:]
    #reshape to 2D array
    ratings = ratings.values.reshape(1, -1)
    # calculate similarities
    similarities = cosine_similarity(count_matrix, ratings, dense_output=False)
    
    # Create a dataframe with the movie titles
    sim_df = pd.DataFrame(
        {'Movie Title': movies_df['movie_title'],
         'Description': movies_df['movie_info'],
         'Release Date': movies_df['original_release_date'],
         'Similarity': similarities.squeeze(),
         'Count': movies_df['tomatometer_count'],
         'Rating': movies_df['tomatometer_status'],
         'url' : movies_df['url']
        })
    
    # include filter option to not include movies rated rotten within the output
    if rotten == True:
         # Get the top 10 movies above the threshold, not including the movie title that is being compared
        top_movies = sim_df[(sim_df['Count'] > vote_threshold) & (sim_df['Movie Title'] != movie_title)].sort_values(by='Similarity', ascending=False).head(10)
    else:
        top_movies = sim_df[(sim_df['Count'] > vote_threshold) & (sim_df['Movie Title'] != movie_title) & (sim_df['Rating'] != 'Rotten')].sort_values(by='Similarity', ascending=False).head(10)
    
    
    
    # printing selected movie name, rating and description
    print(f'Movie Name: \033[1m{movie_title}\033[0;0m')
    print(f'Movie Rating: \033[1m{sim_df.iloc[movie_index,5]}\033[0;0m')
    print(f'Movie Description: \n{sim_df.iloc[movie_index,1]}')
    
  
    return top_movies, movie_index

#######################################################################################################################################
### USER RECOMMENDER FUNCTION




###############################################################
#################### image import function ####################
def movie_image(movie_id):

    try:
        url = movies_df.loc[movie_id,'url']
        
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features = 'lxml')
        
        img = soup.find("div", {"class": "movie-thumbnail-wrap"}).find("img")
        
        icon = img['src']
    
    except ValueError:
        icon = 'https://resizing.flixster.com/WJWgxyXNpxeQ2uYNrQq8tjKWvRc=/206x305/v2/https://flxt.tmsimg.com/assets/p9384_p_v8_bk.jpg'
    
    return icon

########################################################
############# CONTENT RECOMMENDER OUTPUT ###############

st.subheader("Provide Movie Recommendations:")

# define function input
mov_title = st.text_input('Enter the name of a movie you like below', 'Thor')

# choose filter
filter_names = ['Maximum Rating', 'Maximum Number of Reviews']
filt = st.radio('If multiple options, choose movie based on:', filter_names)

# run recommendation
top_movies, movie_index = content_recommender(mov_title, filt)

movie_title = movies_df.iloc[movie_index,1]
# top four ids
top_index = list(top_movies.head(4).index)

# define input
st.markdown(f'Movie Name: **{movie_title}**')
st.markdown(f'Movie Rating: **{movies_df.iloc[movie_index,13]}**')
st.markdown(f'Movie Description: \n{movies_df.iloc[movie_index,2]}')

# display output
st.markdown('### Top 4 Similar Movies by Content:')

# display top four by content images
col1, col2, col3, col4 = st.columns(4)

with col1:
    mov1 = top_movies.loc[top_index[0],'Movie Title']
    st.image(movie_image(top_index[0]))
    st.markdown(f'**{mov1}**')
    

with col2:
    mov2 = top_movies.loc[top_index[1],'Movie Title']
    st.image(movie_image(top_index[1]))
    st.markdown(f'**{mov2}**')
    

with col3:
    mov3 = top_movies.loc[top_index[2],'Movie Title']
    st.image(movie_image(top_index[2]))
    st.markdown(f'**{mov3}**')
    

with col4:
    mov4 = top_movies.loc[top_index[3],'Movie Title']
    st.image(movie_image(top_index[3]))
    st.markdown(f'**{mov4}**')
    

# plot graph

#define the colours based on rating
colour_map={"Rotten":"thistle","Fresh":"mediumpurple", "Certified-Fresh": "rebeccapurple"}
colours = [colour_map[top_movies.loc[v,"Rating"]] for v in top_movies.index]

# define ylabels to prevent duplicating
ylabel = list(top_movies['Movie Title'])
    
fig = plt.figure()
sns.set_theme(style="whitegrid")
# plot barplot with numeric y coordinates 1-10
plt.barh(range(1,11), top_movies['Similarity'], color = colours) # assign colours based on rating
# re-define ylabels based on ylabel list
plt.yticks(range(1,11), ylabel)
      
#plotting the legend
markers = [plt.Line2D([0,0],[0,0],color=colour, marker='o', linestyle='') for colour in colour_map.values()]
plt.legend(markers, colour_map.keys(), numpoints=1, loc = 'lower right')
    
#labels
plt.xlabel('Similarity')
plt.ylabel('Movie Title')
plt.title(f'Movies Similar to {movie_title} based on Movie Content')
plt.xlim((0,max(top_movies['Similarity'])+0.1))
    
# have highest similarity movies at the top
plt.gca().invert_yaxis()
st.pyplot(fig)

# print table
st.dataframe(top_movies)