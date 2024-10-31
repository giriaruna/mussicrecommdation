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
    This app leverages AI to enhance your music discovery experience by analyzing user-created playlists 
    and suggesting 'vibe-matching' songs. Our goal is to help you find new music that aligns with the 
    overall mood of your favorite tracks.
    """)

    # Questions we aims to answer
    st.header("Questions We Aim to Answer")
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

    # descriptive Statistics
    st.subheader("Descriptive Statistics")
    stats = df_cleaned.describe()
    st.write(stats)
    
    st.write("Source: [Spotify Dataset](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset)") 
    
    # our Goals
    st.subheader("Our Goals")
    st.write("""
    The goals of our project are to analyze music features that influence song popularity and to discover patterns in listener preferences.
    """)

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

    # Top 10 Most Popular Songs
    # Explore user engagement metrics
    st.subheader("User Engagement Metrics")
    st.write("Exploring user engagement metrics such as play counts and likes...")
    # Example plot for user engagement
    plt.figure(figsize=(10, 5))
    sns.barplot(x='popularity', y='name', data=df_cleaned.sort_values('popularity', ascending=False).head(10))
    plt.title('Top 10 Most Popular Songs')
    plt.xlabel('Popularity')
    plt.ylabel('Song Name')
    st.pyplot(plt)

    st.write("Explanation: \n This bar plot showcases the 10 most popular songs in the dataset by popularity score. By sorting the data and displaying only the top 10, we gain insights into which songs have achieved the highest user engagement, giving clues about popular music trends.")
    
    avg_energy_by_genre = data_by_genres.groupby('genres')['energy'].mean().reset_index()
    top_avg_energy_by_genre = avg_energy_by_genre.sort_values(by='energy', ascending=False).head(10)
    # Bar plot for average energy by genre
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_avg_energy_by_genre, x='energy', y='genres', palette='viridis')
    plt.title("Average Energy of Top 10 Genres")
    plt.xlabel("Average Energy")
    plt.ylabel("Genre")

    # Rotate y-axis labels for better readability
    plt.yticks(rotation=45)
    st.pyplot(plt)

    # Explanation
    st.write("""
    **Explanation:** 
    This bar plot showcases the average energy levels of the top 10 genres based on their mean energy. By limiting the number of genres displayed, we can better observe and understand the variations in energy across different music genres.
    """)

    # Box plot for energy distribution by genre
    st.subheader("Energy Distribution Across Genres")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data_by_genres, x='genres', y='energy', palette='viridis')
    plt.title("Energy Distribution Across Genres")
    plt.xlabel("Genre")
    plt.ylabel("Energy")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Explanation
    st.write("""
    **Explanation:** 
    This box plot illustrates the distribution of energy levels across various genres. Each box represents the interquartile range (IQR) of energy values, which contains the middle 50% of the data. The horizontal line within the box indicates the median energy level for each genre. 
    - **Whiskers** extend to show the range of the data, excluding outliers, which are plotted as individual points beyond the whiskers.
    - By examining this plot, we can identify which genres have higher energy levels and the variability within each genre. 
    - For instance, genres like EDM may show higher energy levels with a wider range, while classical music may cluster around lower energy values.
    - This visualization helps us understand how energy levels differ not only between genres but also within them, revealing unique trends and listener preferences.
    """)


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

    # Show correlation heatmap
    st.subheader("Correlation Heatmap of Audio Features")
    plt.figure(figsize=(12, 8))
    correlation = df_cleaned.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap of Audio Features")
    st.pyplot(plt)

    # User Engagement Metrics
    st.subheader("User Engagement Metrics")
    st.write("Exploring user engagement metrics such as play counts and likes...")

    # Example plot for user engagement
    plt.figure(figsize=(10, 5))
    sns.barplot(x='popularity', y='name', data=df_cleaned.sort_values('popularity', ascending=False).head(10))
    plt.title('Top 10 Most Popular Songs')
    plt.xlabel('Popularity')
    plt.ylabel('Song Name')
    st.pyplot(plt)

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
    st.write("""
    Our analysis revealed significant insights into music trends and user preferences. By examining 
    audio features and their relationships with user engagement metrics, we can improve our 
    recommendation system.
    """)

    st.write("""
    - **Key Findings**: 
      - Certain genres tend to have higher energy levels, influencing recommendations.
      - Over the years, there has been a noticeable increase in the popularity of upbeat music.
      - Content-based filtering methods can effectively enhance user playlist experiences by providing 
      vibe-matching recommendations.
    """)

    st.write("### Future Work")
    st.write("""
    We aim to refine our recommendation model further by integrating collaborative filtering 
    methods and enhancing the AI's ability to understand user preferences through more complex 
    data analysis.
    """)

# Placeholder function for recommendations (you will implement this logic)
def get_recommendations(user_playlist):
    # Simulated recommendation logic - replace this with actual recommendation code
    # For example, based on your AI model's outputs.
    return ["Suggested Song 1", "Suggested Song 2", "Suggested Song 3"]
