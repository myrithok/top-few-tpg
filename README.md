# Top-Few TPGs: Simultaneous Inputs for Atari Games Using TPGs
### Andrew Mitchell
### 001225727

This repository is forked from https://github.com/gegelati/gym-gegelati, and modified to receive the top 3 actions instead of the top 1.

This project depends on an installation of my modified GEGELATI, which can be found at https://github.com/myrithok/cas739_project_001225727_gegelati.

In order to run this project, start the python server component with:
```
$ cd server-python
$ python3 gym_http_server.py
```
Then build and execute cas739_project_001225727 using cmake. The ```top-few-tpg``` build target runs my top-few tpg implementation for simultaneous inputs, while the ```single-tpg``` build target will run the typical tpg for the standard action space.

If you encounter any installation issues, please see the original repository.
