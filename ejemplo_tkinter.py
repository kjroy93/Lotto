"""Create a GUI window with a title."""

import streamlit as st
import numpy as np
import pandas as pd

st.sidebar.title('This is a sidebar')
valor = st.sidebar.selectbox('Choose a number', [1, 2, 3])
if valor ==1:
    map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])

    st.map(map_data)

if valor == 2:
    st.write('Mira un boton')
    if st.button('Click me'):
        st.write('marico el que lo lea')
    