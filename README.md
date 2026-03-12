# EmblaV1
Extending DreamerV3 for robotic application and better Morphologieawareness by implementing a FFKSM
=======

## Projectplan

[miroboard](https://miro.com/welcomeonboard/cEVObU1TdjYyeTBRMjM0SEZubXFGaFFGdXB0WUwzYjRoc3pRRURXRnZ6T3ZmY3g3MUNiL3NTV2R5UktOMm1EcFJ1eGkzRlByQk16UFMxNUlBYUkvM0crSjRXTWUxclYzT3UxSXN4V1lDVjBIM2RxWnRFT1JGamNqUVdIK0lpQ3ZyVmtkMG5hNDA3dVlncnBvRVB2ZXBnPT0hdjE=?share_link_id=935135186188)

## Usage

- install requirements (you might need to install some additional libs, also note that you need torch with cuda)
- replace pusher_v5.xml in the gymnasium lib folder (yourPythonVenv/gymnasium/envs/mujoco/assets) with the file in this repo! (necessary for the colourfilter)
- run main.py to start
- modify config to change all relevant Dreamer and FFKSM parameters

## Current state

Core architecture done!
Needs benchmark. Maybe implement SM student model

## Acknowledgements

- [NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer) Helped me to understand the DreamerV3 Architecture. Used the Code as a base.
- [SelfModel](https://github.com/H-Y-H-Y-H/SelfSimRobot) Original Selfmodel-code, which I modified and fused with the DreamerV3 Architecture.