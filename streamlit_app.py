# Required Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set the title for the app
st.title("Spotify Music Recommendation System 🎶")

# Sidebar for Navigation
app_page = st.sidebar.selectbox('Select Page', ['Overview', 'Visualization', 'Collaborative Filtering', 
                                                  'Content-Based Filtering', 'Conclusion'])

# Load the dataset
df = pd.read_csv("data.csv")

# Page: Overview
if app_page == 'Overview':
    image_path = Image.open("spotify-image.jpg")  # Replace with an appropriate image path
    st.image(image_path, width=400)

    st.subheader('Questions we aim to answer:')
    st.write("What factors contribute to song popularity?")
    st.write("How do audio features correlate with user engagement?")
    st.write("What are the trends in music preferences over time?")

    st.subheader("Let's explore the dataset!")
    st.write("The dataset we will be analyzing contains information about various songs:")
    st.dataframe(df.head())

    st.write("Information about the DataFrame:")
    st.text(df.info())  # Display DataFrame info directly

    st.write("Statistics on the dataset:")
    st.dataframe(df.describe())

    st.write("Source: [Spotify Dataset](https://www.kaggle.com/datasets/...)")  # Update with the actual source link

    st.subheader("Our Goals")
    st.write("The goals of our project are to analyze music features that influence song popularity and to discover patterns in listener preferences.")

# Page: Visualization
elif app_page == 'Visualization':
    st.title("Data Visualization")

    list_columns = df.columns

    values = st.multiselect("Select two variables:", list_columns, ["popularity", "danceability"])

    # Creation of the line chart
    if len(values) >= 2:
        st.line_chart(df.set_index(values[0])[values[1]])

    # Create heatmap data
    data1 = df.drop(columns=list(df.select_dtypes(include=['object']).columns))
    heatmap_data = data1.corr()

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", ax=ax, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title("Correlation Matrix", fontsize=14)
    st.pyplot(fig)

    # Pairplot
    values_pairplot = st.multiselect("Select 4 variables:", list_columns, ["danceability", "energy", "tempo", "popularity"])
    if len(values_pairplot) == 4:
        df2 = df[values_pairplot]
        pair = sns.pairplot(df2)
        st.pyplot(pair)

# Page: Collaborative Filtering
elif app_page == 'Collaborative Filtering':
    st.title("Collaborative Filtering")
    st.write("This method recommends songs based on their popularity and features.")

    # Function to recommend songs based on similar features
    def get_collaborative_recommendations(song_id, n_recommendations=5):
        if song_id in df['id'].values:
            # Extract the song features
            song_features = df[df['id'] == song_id].drop(columns=['id', 'name', 'artists', 'release_date', 'year', 'popularity'])
            # Calculate similarity based on features (using simple Euclidean distance)
            distances = np.linalg.norm(df.drop(columns=['id', 'name', 'artists', 'release_date', 'year', 'popularity']).values - song_features.values, axis=1)
            # Get indices of the closest songs
            recommended_indices = np.argsort(distances)[:n_recommendations]
            recommended_songs = df.iloc[recommended_indices][['name', 'artists', 'popularity']]
            return recommended_songs
        else:
            return pd.DataFrame(columns=['name', 'artists', 'popularity'])

    # User Input for Collaborative Recommendations
    song_id = st.text_input('Enter Song ID (for collaborative recommendations):')
    if st.button('Get Collaborative Recommendations'):
        if song_id:
            recommendations = get_collaborative_recommendations(song_id)
            if not recommendations.empty:
                st.write('Recommended Songs:')
                st.dataframe(recommendations)
            else:
                st.write("No recommendations found for this song.")
        else:
            st.write("Please enter a valid Song ID.")

# Page: Content-Based Filtering
elif app_page == 'Content-Based Filtering':
    st.title("Content-Based Filtering")
    st.write("This method recommends songs based on the features of a selected song.")

    # Function for Content-Based Recommendations
    def get_content_based_recommendations(song_id, n_recommendations=5):
        if song_id in df['id'].values:
            # Extract the song features
            song_features = df[df['id'] == song_id].drop(columns=['id', 'name', 'artists', 'release_date', 'year', 'popularity'])
            # Calculate similarity based on features (using simple Euclidean distance)
            distances = np.linalg.norm(df.drop(columns=['id', 'name', 'artists', 'release_date', 'year', 'popularity']).values - song_features.values, axis=1)
            # Get indices of the closest songs
            recommended_indices = np.argsort(distances)[:n_recommendations]
            recommended_songs = df.iloc[recommended_indices][['name', 'artists', 'popularity']]
            return recommended_songs
        else:
            return pd.DataFrame(columns=['name', 'artists', 'popularity'])

    # User Input for Content-Based Recommendations
    song_id = st.text_input('Enter Song ID (for content-based recommendations):')
    if st.button('Get Content-Based Recommendations'):
        if song_id:
            recommendations = get_content_based_recommendations(song_id)
            if not recommendations.empty:
                st.write('Recommended Songs:')
                st.dataframe(recommendations)
            else:
                st.write("No recommendations found for this song.")
        else:
            st.write("Please enter a valid Song ID.")

# Page: Conclusion
elif app_page == 'Conclusion':
    st.title('Conclusion 🎈')
    st.balloons()

    st.subheader('1. Insights:')
    st.markdown('- **Music Features Analysis:** Analyzing audio features revealed significant trends in how these affect popularity and user engagement.')
    st.markdown('- **Collaborative Filtering Success:** Recommendations based on song features can effectively suggest similar music to users.')
    st.subheader('2. Future Improvements:')
    st.markdown("- **Enhanced Recommendations:** Incorporating additional features and user feedback mechanisms could improve the accuracy of recommendations.")
    st.subheader('3. Long-term Considerations:')
    st.markdown("- **Dynamic Updates:** Regularly updating the dataset with new songs and user interactions will keep the recommendations relevant.")
