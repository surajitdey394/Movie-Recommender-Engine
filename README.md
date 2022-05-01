# Movie-Recommender-Engine
<!-- ![](https://media.giphy.com/media/3ohhwDMC187JqL69DG/giphy.gif) -->
<p align="center" width="100%">
    <img width="25%" src="https://media.giphy.com/media/3ohhwDMC187JqL69DG/giphy.gif"> 

# **ABOUT**
I have always define Cinema as an art in motion, formed by layers and layers of other arts which keeps you telling a lot of subtle litle things- when set in motion,     creates something like a music to the eyes. 
For every person cinema has it own charisma and this charisma has drawn nearly every person of this world. Present time sees the epitome of cinema making and also in 
the number of people intersted in this art form, early 2019 saw a peak in ticket sales nearing $12 Billion ticket sale worldwide. 
<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/24209142/166158842-3b146da8-2985-4df5-b51c-541c0853f542.png" width="55%">

This project aims to help this community who like to admire this very art form - "MOVIES".


    
# **INTRODUCTION**

This project is build from the logics of content based recommendations, helps a user by suggesting simmilar kind of movies based on user's preference. The recommendation engine generates three types of recommendations for the user:

1. Top-10 similar movies for the user, which the user may like to watch.
2. Top-10 movies from the same director.
3. Movies from same franchise.

The models uses TF-IDF Vectorization and Word-Embeddings from Glove-840B word-embeddings on 20M Movielens dataset. 


    
# **EDA on 20M Movielens Dataset**
1. Wordcloud on movie titles- Shows the most common words used for movie titles which gives an idea about the most used concepts in dataset's films.
<p align="center" width="100%">
    <img src = "https://user-images.githubusercontent.com/24209142/166154052-caabe46e-d4cc-4cf6-85a5-2c6ab509f163.png" width = 55%>

2. Bar chart of genre in the dataset: Shows the distribution of genres in the dataset.
<p align="center" width="100%">
    <img src = "https://user-images.githubusercontent.com/24209142/166164494-4201adc9-5547-4d5d-91ee-5752cde820b9.jpg" width = 75%>

3. Heatmap of Genres- The heatmap graph shows the genres that are related to each other and can share the list of a movie genre. 

 <p align="center" width="100%">   
    <img src="https://user-images.githubusercontent.com/24209142/166165008-4fe4566b-0f64-42c8-a2ad-032e0eddaf81.png" width="65%">
