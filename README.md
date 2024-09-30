# UrbanClusterStatistics
-------
A web application based on the [Momepy](https://docs.momepy.org/en/stable/index.html) library.

The application receives either GDB/SHP files containing buildings and streets of an urban area exported from a GIS software, or fetches the data from OpenStreetMaps.

Multiple morphological metrics are then computed. The metrics are visualized on top of a map of the area.

The metrics are then utilized for the detection of urban clusters within the area, once more plotted on top of a map of the area.

A statistical analysis of the clusters is then performed and visualized.

## Zip files for windows
The latest versions of the application can be found at: https://drive.google.com/drive/folders/1H_L4Z8-4u4spILFSY_iYLom680190FbZ?usp=drive_link

##  app
* Directory containing code for the app. subdivisions are preliminary for now.

### to run the app 
* first install the requirements by running the following command in the terminal:   
```pip install -r requirements.txt```

* then run the app by running the following command in the terminal:  
```streamlit run app/streamlit_app.py```
