# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# readign three datasets
df = pd.read_csv("data.csv")  # primary dataset
data_by_genres = pd.read_csv("data_by_genres.csv")  # genres dataset
data_by_year = pd.read_csv("data_by_year.csv")  # year dataset
data_w_genres = pd.read_csv('data_w_genres.csv')
data_by_artist = pd.read_csv('data_by_artist.csv')


# app title
st.title("Music Recommendation System 🎶")


# Sidebar for navigation
app_page = st.sidebar.selectbox('Select Page', ['Overview', 'Visualization', 'AI Curation', 'Conclusion'])


# dropping rows with missing values
df_cleaned = df.dropna()




# Overview Page
if app_page == 'Overview':
   # displaying Spotify logo
   image_path = Image.open("spotify-image.jpg") 
   st.image(image_path, width=400) 
  
   # Project Introduction
   st.header("Project Introduction")
   st.write("""
   Our goal is to leverage advanced AI techniques to enhance music discovery by analyzing user-generated playlists and providing personalized song recommendations. We aim to identify the key characteristics that influence song popularity and user preferences, ultimately creating a dynamic platform that helps users uncover new music that aligns with their tastes. By refining our AI algorithms, we strive to deliver a tailored listening experience that not only meets but anticipates user needs.
   """)

   # Questions we aims to answer
   st.header("Key Questions We'll Explore:")
   st.write("""
   - How can song characteristics (e.g., tempo, danceability, energy) inform personalized recommendations?
   - What patterns exist in the data that help us understand user preferences and music trends?
   - How can we effectively combine user-created playlists with AI-generated recommendations?
   """)


   # Overview
   st.header("Dataset Overview")
   st.write("### Primary Dataset")
   st.write(df_cleaned.head())  # Display the first few rows of the cleaned dataset
   st.write(df_cleaned.describe())  # Display summary statistics


   st.write("### Genres Dataset")
   st.write(data_by_genres.head())  # Display the first few rows of the genres dataset
   st.write(data_by_genres.describe())  # Display summary statistics


   st.write("### Year Dataset")
   st.write(data_by_year.head())  # Display the first few rows of the year dataset
   st.write(data_by_year.describe())  # Display summary statistics

  
   st.write("Source: [Spotify Dataset](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset)")
  
   # our Goals
   st.subheader("Our Goals")
   st.write("""
   The goals of our project are to analyze music features that influence song popularity and to discover patterns in listener preferences.
   """)
   
   st.subheader("Group Members")
   st.write("**Aruna Giri**, **Uriel Olayinka**, **Harini Vinu**, **Jani Jain**")

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
       # Create histogram with KDE for each feature
       sns.histplot(df_cleaned[feature], bins=30, kde=True)
       plt.title(f'Distribution of {feature}')
       plt.xlabel(feature)
       plt.ylabel('Frequency')
       st.pyplot(plt)
   # Explanation:
   # This loop iterates over the selected audio features ('tempo', 'energy', 'danceability'),
   # generating a histogram with a KDE (Kernel Density Estimate) for each feature.
   # This visualization helps us observe the distribution of these features across songs in the dataset.




   # Correlation Heatmap of Audio Features
   st.subheader("Correlation Heatmap of Audio Features")
   numeric_df = df_cleaned.select_dtypes(include=['float64', 'int64'])
   correlation = numeric_df.corr()
   plt.figure(figsize=(12, 8))
   sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
   st.pyplot(plt)




   # Trend Over Years: Count of Songs by Tempo
   st.subheader("Trend of Tempo Over the Years")


   # Create bins for tempo
   bins = [0, 60, 120, 180, 240]  # Adjust these bins according to your dataset
   bin_labels = ['0-60', '61-120', '121-180', '181-240']
   df_cleaned['tempo_bins'] = pd.cut(df_cleaned['tempo'], bins=bins, labels=bin_labels)


   # Count songs per year and tempo bin
   tempo_count_by_year = df_cleaned.groupby(['year', 'tempo_bins']).size().reset_index(name='count')


   # Create a pivot table for easier plotting
   tempo_pivot = tempo_count_by_year.pivot(index='year', columns='tempo_bins', values='count').fillna(0)


   # Plotting the trend of tempo over the years
   plt.figure(figsize=(12, 6))
   sns.lineplot(data=tempo_pivot, dashes=False, markers=True)
   plt.title("Trend of Song Counts by Tempo Over the Years")
   plt.xlabel("Year")
   plt.ylabel("Count of Songs")
   plt.xticks(rotation=45)
   plt.legend(title='Tempo Bins')
   st.pyplot(plt)


   st.write("""
   **Explanation:**
   This line plot illustrates the trend of song counts in different tempo bins over the years. Each line represents a specific tempo range, allowing us to see how the popularity of songs within those tempo categories has changed over time. This visualization helps us understand shifts in music trends and how they might influence listener preferences.
   """)




   # Decade Analysis
   st.subheader("Trend of Songs by Decade")
   # Function to get the decade
   def get_decade(year):
       return f"{year // 10 * 10}s"


   # Create a decade column in the cleaned DataFrame
   df_cleaned['decade'] = df_cleaned['year'].apply(get_decade)


   # Create a count plot for decades
   plt.figure(figsize=(11, 6))
   sns.countplot(data=df_cleaned, x='decade', palette='pastel')
   plt.title('Count of Songs by Decade')
   plt.xlabel('Decade')
   plt.ylabel('Count')
   st.pyplot(plt)


   st.write("""
   **Explanation:**
   This count plot illustrates the number of songs released in each decade. This helps visualize how music production has varied over the years and can indicate trends in music popularity over time.
   """)




# AI Curation Page
elif app_page == 'AI Curation':
   st.header("AI-Driven Recommendations")
   st.write("""
   Our AI curation feature enhances your music experience by suggesting songs that match the vibe of your
   created playlists. This involves analyzing the audio characteristics and overall mood of your selected songs
   to recommend additional tracks that complement your chosen theme.
   """)


   # Example User Interaction
   st.subheader("Create Your Playlist")
   user_playlist = st.text_area("Enter your playlist songs (comma-separated):", "Song 1, Song 2, Song 3")


   # Process User Input
   if st.button("Get Recommendations"):
       # Example function to simulate recommendation based on user input
       # In a real implementation, you would include your recommendation algorithm here
       recommendations = get_recommendations(user_playlist)  # Placeholder function
       st.write("### Recommended Songs:")
       st.write(recommendations)


   st.write("""
   ### How It Works
   - **Step 1**: You input your favorite songs.
   - **Step 2**: The AI analyzes the vibe and characteristics of your playlist.
   - **Step 3**: It suggests additional songs that fit the same vibe, helping you discover new music.
   """)


# Conclusion Page
elif app_page == 'Conclusion':
   st.header("Conclusion and Insights")
   st.balloons()
   image_path = Image.open("fun.jpg") 
   st.image(image_path, width=400) 
   st.write("""
   Our analysis has provided valuable insights into music trends and user preferences, directly addressing our initial goals.
   """)

   st.write("""
   - **Key Findings**:
     - **Audio Feature Analysis**: We identified how different audio features influence song popularity and user engagement.
     - **Enhanced Recommendation System**: By understanding the impact of these features, we can improve our recommendation system, suggesting songs that are likely to resonate with users' preferences.
   """)

   st.write("### Future Work")
   st.write("""
   Moving forward, we aim to enhance our music recommendation system by:
   - **Data Visualization**: Implementing visualizations that display insights on music genres and user preferences.
   - **User Experience**: Designing features that support continuous play and personalized music recommendations based on user input and preferences.
   - **AI Recommendations**: Further improving the AI's ability to provide highly personalized and accurate song suggestions based on user vibes.

   By pursuing these goals, we can create a more engaging and personalized music experience that not only recommends songs but also helps users discover new favorites based on predicted popularity.
   """)

