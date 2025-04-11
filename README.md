# YouTube Analysis and Recommender System

In this project, we perform a detailed analysis of trending YouTube data to build a recommender system. YouTube, being one of the largest video-sharing platforms, holds a wealth of information that can be leveraged to understand user preferences, predict trends, and suggest relevant content.

## Overview

This project focuses on analyzing YouTube videos, their metadata, user interactions, and building a system that can recommend videos based on different factors such as view history, user preferences, and similar content.


## Objectives

- **Data Collection**: Retrieve YouTube video data, including video titles, descriptions, view counts, likes, and comments.
- **Data Preprocessing**: Clean and prepare the dataset for analysis.
- **Exploratory Data Analysis (EDA)**: Investigate patterns, trends, and insights from the dataset.
- **Recommender System**: Build a content-based or collaborative filtering recommender system to suggest videos to users.
- **Evaluation**: Evaluate the recommender system using appropriate metrics like accuracy, precision, and recall.


## Methodology

We will employ machine learning algorithms to build our recommender system. Depending on the approach, we will use algorithms such as:
- **Content-Based Filtering**: Recommending videos based on similarities between the content of videos.
- **Channel-Based Filtering**: Recommending channels with a variety of videos that might suit your preferences.


The model will be evaluated based on the ability to suggest relevant and engaging videos to users.

## Conclusion

By the end of this project, we aim to have a variety of fully functional recommender systems that are capable of providing personalized video recommendations, enhancing user engagement, and improving their overall experience on YouTube.

# REQUIREMENTS FOR PROJECT

Before importing the libraries, it is recommended to download all the requirements in a virtual environment.  
We have used conda to work on this project, so to ensure everything works as intended try to create a conda environment with the following command:

```bash
conda env create -f environment.yml
```

If you cannot access conda, feel free to use the requirements.txt for python environments.

# CONTENTS OF THE PROJECT

    1. Data
        - Here you will find all the information regarding the dataset, with the link to the original webpage.
    
    2. Scripts
        - A series of python scripts that explore different methods of recommending videos, with their respective pros and cons.

    3. Src
        - A python script that collects all the necessary functions for the scripts explained above.
    
    4. Miscellaneous files
        - Files such as requirements and objectives of the project can be found below the folders explained above.
