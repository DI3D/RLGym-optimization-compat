# The Rocket League Gym But Optimized!?!?
***----NOTE: LIGHTLY TESTED, PROBABLY NOT BUGGY----***
## Description
This is a python API that can be used to treat the game [Rocket League](https://www.rocketleague.com) as though it were an [OpenAI Gym](https://gym.openai.com)-style environment for Reinforcement Learning projects. This API must be used with the accompanying Bakkesmod plugin.

## Requirements
* A Windows 10 PC
* Rocket League (Both Steam and Epic are supported)
* [Bakkesmod](https://www.bakkesmod.com)
* The RLGym plugin for Bakkesmod (It's installed automatically by pip)
* Python between versions 3.7 and 3.9 (3.10 not supported).

## Installation
Because of the .pyd for RLGym-Rust calculations, you must copy and paste everything in the /rlgym directory instead of installing it unfortunately. If someone knows of a way to install .pyd files with the package, please do let me know.

Once the API is installed, you will need to enable the RLGym plugin from inside the Bakkesmod plugin manager. To do this, first launch the game, then press F2 to open the Bakkesmod menu. Navigate to the `plugins` tab and open the `Plugin Manager`. From there, scroll down until you find the RLGym plugin, and enable it. Close the game when this is done.

RLGym is now installed! simply run ```example.py``` from our repo to ensure everything works.

## Usage
For tutorials and documentation, please visit our [Wiki](https://rlgym.github.io/).
