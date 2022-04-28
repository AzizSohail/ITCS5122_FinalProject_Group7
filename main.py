import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
from vega_datasets import data
import pydeck as pdk
import plotly.express as px

option = st.sidebar.radio("Select the dashboard", (
'Total Number of Incidents Per Ward', 'Total Number of Incidents Per District',
'Total Number of Incidents Per Community Area',
'Selection Histogram - Maximum type of Crimes commited per community Area ',
'Layered Bar Chart - layered representation of Crimes per Community Area',
'Map Visualization - Number of crimes at a given time of a day',
'Map Visualization - Locations with most theft/burglaries'))


@st.cache()
def load_data():
    data = pd.read_csv("Crimes_dataset.csv", nrows=100000)
    # data =  pd.read_csv("drive/My Drive/Crimes_dataset.csv")
    data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    data['time'] = pd.to_datetime(data['Date'].astype(str)).dt.hour
    data['date'] = [d.date() for d in pd.to_datetime(data['Date'].astype(str))]
    data['day'] = [d.day_name() for d in pd.to_datetime(data['Date'].astype(str))]

    data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    data.drop(columns=['Date'], inplace=True)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    return data


if option == 'Total Number of Incidents Per Ward':
    data = pd.read_csv("Crimes_dataset.csv")
    data = data.dropna()
    data.head()
    data[['District', 'Ward', 'Community Area']] = data[['District', 'Ward', 'Community Area']].astype('int')
    data[['District', 'Ward', 'Community Area']] = data[['District', 'Ward', 'Community Area']].astype('str')
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    WardData = pd.DataFrame(data['ward'].value_counts(ascending=True).astype(float))
    WardData = WardData.reset_index()
    WardData.columns = ['ward', 'Crime_Count']

    bar_chart1 = alt.Chart(WardData, title='Total Number of Incidents Per Ward').mark_bar().encode(

        x=alt.X('ward:N', title='ward'),
        y=alt.Y('Crime_Count:Q', title='Crime_Count'),
    ).properties(
        width=800, height=600
    )
    st.write(bar_chart1)

elif option == 'Total Number of Incidents Per District':
    data = pd.read_csv("Crimes_dataset.csv")
    data = data.dropna()
    data.head()
    data[['District', 'Ward', 'Community Area']] = data[['District', 'Ward', 'Community Area']].astype('int')
    data[['District', 'Ward', 'Community Area']] = data[['District', 'Ward', 'Community Area']].astype('str')
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    DisData = pd.DataFrame(data['district'].value_counts(ascending=True).astype(float))
    DisData = DisData.reset_index()

    DisData.columns = ['district', 'Crime_Count']

    bar_chart2 = alt.Chart(DisData, title='Total Number of Incidents Per District').mark_bar().encode(

        x=alt.X('district:N', title='district'),
        y=alt.Y('Crime_Count:Q', title='Crime_Count'),
    ).properties(
        width=700, height=500
    )
    st.write(bar_chart2)

elif option == 'Total Number of Incidents Per Community Area':

    data = pd.read_csv("Crimes_dataset.csv")
    data = data.dropna()
    data.head()
    data[['District', 'Ward', 'Community Area']] = data[['District', 'Ward', 'Community Area']].astype('int')
    data[['District', 'Ward', 'Community Area']] = data[['District', 'Ward', 'Community Area']].astype('str')
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    ComData = pd.DataFrame(data['community_area'].value_counts().astype(float))
    ComData = ComData.reset_index()

    ComData.columns = ['community_area', 'Crime_Count']

    bar_chart3 = alt.Chart(ComData, title='Total Number of Incidents Per Community Area').mark_bar().encode(

        x=alt.X('community_area:N', title='community_area'),
        y=alt.Y('Crime_Count:Q', title='Crime_Count'),
    ).properties(
        width=700, height=500
    )

    st.write(bar_chart3)


elif option == 'Selection Histogram - Maximum type of Crimes commited per community Area ':

    data = pd.read_csv("Crimes_dataset.csv", nrows=100000)
    st.header("Maximum type of Crimes commited per Community Area")
    # alt.data_transformers.disable_max_rows()
    data2 = data[data["Primary Type"].isin(['THEFT', 'BATTERY', 'CRIMINAL DAMAGE'])]

    select = alt.selection(type='interval')
    values = alt.Chart(data2).mark_point().encode(
        x='Year:N',
        y='Community Area:Q',
        color=alt.condition(select, 'Primary Type:N', alt.value('lightgray'))
    ).properties(
        width=480,
        height=400

    ).add_selection(
        select
    )
    bars = alt.Chart(data2).mark_bar().encode(
        y='Primary Type:N',
        color='Primary Type:N',
        x='count(Primary Type):Q'
    ).properties(
        width=200,
    ).transform_filter(
        select
    )
    st.write(values | bars)

elif option == 'Layered Bar Chart - layered representation of Crimes per Community Area':

    data = pd.read_csv("Crimes_dataset.csv", nrows=100000)
    st.header("Layered representation of Crimes per Community Area")
    # alt.data_transformers.disable_max_rows()
    data2 = data[data["Primary Type"].isin(['THEFT', 'BATTERY', 'CRIMINAL DAMAGE'])]
    layered = alt.Chart(data2).mark_bar(opacity=0.7).encode(
        x='Year:O',
        y=alt.Y('Community Area:Q', stack=None),
        color="Primary Type",
    ).properties(
        width=700,
        height=500
    )

    st.write(layered)

elif option == 'Map Visualization - Number of crimes at a given time of a day':
    data = load_data()
    st.header("Number of crimes at a given time of a day")
    hour = st.slider("Hour to look at", 0, 23)
    hour
    original_data = data
    data = data[data["time"] == hour]
    midpoint = (np.average(data['latitude']), np.average(data['longitude']))
    st.markdown("All Crimes")

    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state={
            "latitude": midpoint[0],
            "longitude": midpoint[1],
            "zoom": 11,
            "pitch": 30,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data[['time', 'latitude', 'longitude']],
                get_position=["longitude", "latitude"],
                auto_highlight=True,
                radius=100,
                extruded=True,
                pickable=True,
                elevation_scale=4,
                elevation_range=[0, 1000],
            ),
        ],
    ))


elif option == 'Map Visualization - Locations with most theft/burglaries':
    data = load_data()
    st.header("Locations with most theft/burglaries")
    hour3 = st.slider("Select any time", 0, 23)
    hour3
    data4 = data[data["primary type"] == "THEFT"]
    data5 = data4[data4["time"] == hour3]
    midpoint = (np.average(data5['latitude']), np.average(data5['longitude']))
    st.markdown("Theft")

    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state={
            "latitude": midpoint[0],
            "longitude": midpoint[1],
            "zoom": 11,
            "pitch": 30,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data5[['time', 'latitude', 'longitude']],
                get_position=["longitude", "latitude"],
                auto_highlight=True,
                radius=100,
                extruded=True,
                pickable=True,
                elevation_scale=4,
                elevation_range=[0, 1000],
            ),
        ],
    ))

