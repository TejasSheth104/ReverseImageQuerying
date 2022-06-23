import streamlit as st
import pickle
import matplotlib.image as mpimg
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def main():
  # Load initial input prompt
  st.title("Reverse Image Querying")
  image_file = st.file_uploader("Uplaod Image", type=['png', 'jpeg', 'jpg'])

  # Open, Load & close the pickle file 
  file = open('datax', 'rb')
  df = pickle.load(file)
  file.close()

  if image_file is not None:
    # getting query image  
    input_image = mpimg.imread(image_file)
    # computing rgb avg of the image
    red = np.average(input_image[:, :, 0])
    blue = np.average(input_image[:, :, 1])
    green = np.average(input_image[:, :, 2])
    # storing it as a feature in an array
    query_feature = [input_image,red, blue, green]

    # calulate cosine distance from query image to all images 
    cosine_distance=[]
    idx = 0
    for i in range(len(df)):
      temp_data = df.iloc[i, -3:]
      temp_data = np.array(temp_data).reshape(1, -1)
      dist=cosine(query_feature[-3:], temp_data)
      cosine_distance.append([dist, idx])
      idx += 1

    # sorting the cosine distances
    cosine_distance.sort()

    # storing the images and their respective cosine distances 
    result = []
    for dist,idx in cosine_distance:
      result.append(idx)

    # displaying output    
    x = 0
    i = 0
    cols = st.columns(3)
    for idx in result:
      plt.figure()
      plt.imshow(df.iloc[idx][0])
      cols[i].image(df.iloc[idx][0], use_column_width=True)
      i += 1
      if(i>2):
        i = 0
      x += 1
      if(x > 15):
        break

if __name__ == '__main__':
  main()