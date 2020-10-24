# :seedling: Multimodal freight USA
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Goal: Generate and analyze a multimodal network in the USA to make freight transportation greener.

## Usage
To run the app on a docker image:
Clone the repo, change the work direction and run the following commands in your CLI:<br>
`$ build -t dockermfreigh .` To build the image with it's own conda environment and all the requirements for the project <br>
`$ docker run -p 5000:5000 dockermfreigh` To run the image and display the app on the local server. <br>
Then, open you browser and type localhost:5000. The app should be running.

## Structure
### Jupyter Notebooks
| File | Description | Status |
| ----------- | ----------- |  ----------- | 
| Rail_EDA.ipynb | Exploration and cleaning of the [BTS](https://data-usdot.opendata.arcgis.com/datasets/north-american-rail-lines-1) rail network dataset | Done |
| Rail_EDA_florida.ipynb | Rail network generation for florida, computation speed assessment | Done |
| Rail_network.ipynb | Rail network analysis | Later |
| OSM_roads_florida.ipynb | Generate the road network of florida from OSM data | Done |
| BTS_roads_florida.ipynb | Generate the road network of florida from BTS data | Done |
| Florida_roads_compare.ipynb | Compare BTS with OSM network routing performance | Done |
| OSM_problems.ipynb | Analyzed why OSM data is not producing a correct graph| Done |
| CO2_EDA.ipynb | Compute co2 equivalent for rail and road | Done |
| merge_florida.ipynb | Merge Rail and road network through intermodal facilities | Done |
| verify_gen_network.ipynb | Generate the rail and road network, display them | Done |
| verify_merge.ipynb | Verify merge networks, display the resulting plot | Done |
| astar_heuristic.ipynb | Explore and plot the astar routing algorithm | WIP |


### Scripts
| Directory | File | Description | Status |
| ----------- | ----------- | ----------- |  ----------- | 
| mfreight.Rail| gen_rail_net.py | Generate rail network | Done|
| mfreight.Road| gen_road_net.py | Generate road network | Done|
| mfreight.Multimodal | merge_graphs.py | Merge rail and road networks | PR|
| mfreight.Multimodal | graph_utils.py | Display and adapt the multimodal graph | PR|
| mfreight.utils.| * | All the support functions for the main scripts above | PR|
| . | app.py | Plotly dash app | Could be improved|

## License

This project is licensed under Private.
