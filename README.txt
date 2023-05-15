READ ME.txt

Catherine Hocknell
catherine.hocknell@gmail.com
https://github.com/chocknell


New User Movie Recommendations - Sentiment Analysis and Three Different Types of Recommender System
This project involves a sentiment analysis of Rotten Tomatoes movie reviews, then follows up with 
a number of possible Recommender Systems to be used by a new user to a streaming service.

Data Source:
https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

data file location:
https://drive.google.com/drive/folders/16K1OpmfV5Jw-7RBIgmgQFkmvL-op9Ll2?usp=sharing


Documents in this project folder include:

Project Summary Document:
1 - Catherine_Hocknell_MovieRecommendations_FinalReport.pdf - The final summary report for the project

Jupyter Notebooks (in chronalogical order):
2 - Review Data Cleaning (1) - EDA and cleaning of Reviews dataset - includes Reviews data dictionary
3 - Data Wrangling (2) - NLP Processing and User relationships - includes Movie data dictionary
4 - Sentiment Analysis (3) - Classification model evaluation for sentiment analysis of reviews
5 - Content Based Recommender System (4) - Rec. System based on Movie dataset content
6 - User Based and Collaborative Recommender System (5) - Collaborative and User Similarity Rec. Systems

The raw data files included in the data folder are:
7 - rotten_tomatoes_critic_reviews.csv - original review data from kaggle
8 - rotten_tomatoes_movies.csv - original movie data from kaggle
9 - cleaned_reviews_full.csv - reviews df once new 'final_score' column has been assigned
10 - languages_complete.csv - subset of 9 with languages assigned of each review
11 - english_reviews_full.csv - previous csv with all non-english reviews removed
12 - reviews_for_vectorizing.csv - combined 9 and 11 with NLP pre-processing of review text

Other files within the data folder are:
13 - positive.png - output from Positive sentiment wordcloud
14 - negative.png - output from Negative sentiment wordcloud
15 - comment_l.png - mask image 1 used in generation of wordclouds
16 - comment_r.png - mask image 2 used in generation of wordclouds
17 - environment.yml - 'capstone' environment file
18 - model_results.csv - sentiment analysis model result scores to be used to generate visualisations
19 - Visuals.twb - Sentiment analysis model accuracy visualisations (input = model_results.csv)
20 - Target Variables.twb - Target variable distribution visualisation (input = reviews_for_vectorizing.csv)
