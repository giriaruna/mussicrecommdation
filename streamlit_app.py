# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load datasets
df = pd.read_csv("data.csv")  # Primary dataset
data_by_genres = pd.read_csv("data_by_genres.csv")  # Genres dataset
data_by_year = pd.read_csv("data_by_year.csv")  # Year dataset

# Set up Streamlit app
st.title("Music Recommendation System 🎶")

# Initial data cleaning
cleaned_df = df.dropna()  # Drop rows with missing values

# Sidebar for navigation
app_page = st.sidebar.selectbox('Select Page', ['Overview', 'Visualization', 'Collaborative Filtering', 'Content-Based Filtering', 'Conclusion'])
df_cleaned = df.dropna()

# Overview Page
if app_page == 'Overview':
    # Display Spotify logo
    image_path = Image.open("spotify-image.jpg")  
    st.image(image_path, width=400)
   
    # Project Introduction
    st.header("Project Introduction")
    st.write("""This project focuses on building a music recommendation system that utilizes user listening history and song characteristics from the Spotify dataset. The aim is to analyze user preferences and music trends to create personalized
    recommendations that enhance the overall listening experience.
    """)
    # Questions we aim to answer
    st.subheader('Questions we aim to answer:')
    st.write("1. What factors contribute to song popularity?")
    st.write("2. How do audio features correlate with user engagement?")
    st.write("3. What are the trends in music preferences over time?")

    st.subheader("Let's explore the dataset!")
    st.write("The dataset we will be analyzing contains information about various songs:")
    st.dataframe(df.head())

    st.write("Statistics on the dataset:")
    st.dataframe(df.describe())

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    stats = df_cleaned.describe()
    st.write(stats)
    
    st.write("Source: [Spotify Dataset](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset)") 
    st.subheader("Our Goals")
    st.write("The goals of our project are to analyze music features that influence song popularity and to discover patterns in listener preferences.")

    st.success("Let's embark on this musical journey!")

# Visualization Page
elif app_page == 'Visualization':
    st.header("Dataset Exploration")

    # Familiarize with the dataset
    st.subheader("Dataset Overview")
    st.write("This dataset contains information about various songs, including their features and user interactions.")
    
    # Check for missing values
    missing_values = df.isnull().sum()

    # Analyze audio features
    st.subheader("Audio Feature Analysis")
    st.write("Analyzing audio features like tempo, energy, and danceability...")
    
    # Create plots for audio features
    feature_columns = ['tempo', 'energy', 'danceability']
    for feature in feature_columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(cleaned_df[feature], bins=30, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        st.pyplot(plt)

    # Explore user engagement metrics
    st.subheader("User Engagement Metrics")
    st.write("Exploring user engagement metrics such as play counts and likes...")
    # Example plot for user engagement
    plt.figure(figsize=(10, 5))
    sns.barplot(x='popularity', y='name', data=cleaned_df.sort_values('popularity', ascending=False).head(10))
    plt.title('Top 10 Most Popular Songs')
    plt.xlabel('Popularity')
    plt.ylabel('Song Name')
    st.pyplot(plt)

# Content-Based Filtering Page
elif app_page == 'Content-Based Filtering':
    st.header("Content-Based Filtering")
    st.write("Using basic content-based filtering techniques based on audio features...")
    
    # Example: Recommend songs based on audio features of a selected song
    song_to_filter = st.selectbox("Select a song for content-based recommendations:", cleaned_df['name'])
    
    if song_to_filter:
        # Get the features of the selected song
        selected_song = cleaned_df[cleaned_df['name'] == song_to_filter]
        
        if not selected_song.empty:
            energy = selected_song['energy'].values[0]
            danceability = selected_song['danceability'].values[0]

            # Filter songs based on similar energy and danceability
            recommendations = cleaned_df[
                (cleaned_df['energy'] >= energy - 0.1) & (cleaned_df['energy'] <= energy + 0.1) &
                (cleaned_df['danceability'] >= danceability - 0.1) & (cleaned_df['danceability'] <= danceability + 0.1)
            ]

            recommendations = recommendations[recommendations['name'] != song_to_filter]  # Exclude the selected song
            st.write("Recommended songs based on features similar to **{}**:".format(song_to_filter))
            st.dataframe(recommendations[['name', 'energy', 'danceability']].head(10))  # Display top 10 recommendations

# Conclusion Page
elif app_page == 'Conclusion':
    st.header("Conclusion")
    st.write("Summarize the findings and insights from the project.")

