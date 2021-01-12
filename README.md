# :seedling: Multimodal freight USA
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Goal: Generate and analyze a multimodal network in the USA to make freight transportation greener.

## Usage
To run the app on a docker image:
Clone the repo, change the work direction and run the following commands in your CLI:<br>
`$ docker build -t dockermfreigh .` To build the image with its own python environment and all the requirements for the project <br>
`$ docker run -p 5000:5000 --name green_freight dockermfreigh` To run the image and display the app on the local server. <br>
Then, open your browser and type localhost:5000. The app should be running.

## Description
This project intended to build a simplified model of a multimodal freight network in the USA.
### Data
The data used in this project was essentially pulled from publicly available databases:
The geographical data for roads and railroads are available on the [BTS](https://data-usdot.opendata.arcgis.com) atlas database. For the edges attributes other data sources were used:
* __CO2 emissions:__ The emissions for each mode were computed using the U.S. LifeCycle Inventory Database (USLCI) database. To estimate the CO2emissions of a shipment, a weight of about 40 tonnes per load was estimated assuming a 53 feet container filled at 70% of its max capacity.
* __Duration:__ For both, the rail and the road graphs, each edge had a distance attribute defined in the initial data set. To compute the duration, speed had to be assigned. For railroads, the track class of each edge was used. For roads, the state speed limit was used.
* __Pricing:__ An internal historical database was used for a rough estimations of state to state pricing per mode.

### Notebooks
Analysis of the different networks, the different path search algorithms and Noe4j performance can be found in the `Notebooks` directory.

### Disclaimer
This model is a PoC, the duration, pricing and CO2 results displayed in the app are only estimations.

