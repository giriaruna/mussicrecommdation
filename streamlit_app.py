import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mt
from PIL import Image
import io

# Set the title for the Streamlit app
st.title("Spotify Music Recommendation System 🎶")

# Sidebar for page selection
app_page = st.sidebar.selectbox('Select Page', ['Overview', 'Visualization', 'Conclusion'])

# Load the Spotify dataset (ensure the correct path to your CSV file)
df = pd.read_csv("data.csv")

if app_page == 'Overview':
    #image_path = Image.open("spotify_logo.jpg")  # Replace with your image
    #st.image(image_path, width=400)
    
    st.subheader('Questions we aim to answer: ')
    st.write("What factors influence music recommendations? How can we personalize listening experiences?")
    
    st.subheader("Dataset Overview")
    st.write("We will analyze the Spotify dataset to provide music recommendations.")
    st.dataframe(df.head())
    
    st.write("Information about the dataset: ")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("Statistics on the dataset: ")
    st.dataframe(df.describe())

if app_page == 'Visualization':
    st.title("Data Visualization")

    # Display a line chart of two selected variables
    list_columns = df.columns
    values = st.multiselect("Select two variables: ", list_columns, ["danceability", "energy"])

    # Creation of the line chart
    if len(values) == 2:
        st.line_chart(df[values])

    # Create a heatmap for correlation analysis
    heatmap_data = df.corr()  # Calculate correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size as needed
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", ax=ax, cmap='RdBu_r', vmin=-1, vmax=1) 
    plt.title("Correlation Matrix of Spotify Features", fontsize=14)
    st.pyplot(fig)

    # Pairplot of selected features
    values_pairplot = st.multiselect("Select 4 variables for pairplot: ", list_columns, ["danceability", "energy", "loudness", "tempo"])
    if len(values_pairplot) == 4:
        df2 = df[values_pairplot]
        pair = sns.pairplot(df2)
        st.pyplot(pair)


if app_page == 'Conclusion':
    st.title('Conclusion')
    st.balloons()
