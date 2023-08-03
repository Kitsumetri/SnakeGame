# Simple snake game

## Build
#### Linux / MacOS
```
git clone https://github.com/Kitsumetri/SnakeGame.git
cd SnakeGame
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/main.py
```
#### Windows
```
git clone https://github.com/Kitsumetri/SnakeGame.git
cd SnakeGame
python3.11 -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python src/main.py
```
## Dependencies
* python 3.1.1.*
* pygame 2.5.0
  
## Control
* **W** - Up
* **A** - Left
* **S** - Down
* **D** - Right
* **Space** - start game in main window
  
## Features
| Sprite   | Feature |
|---|---|
| ![apple_sprite](https://github.com/Kitsumetri/SnakeGame/assets/100523204/877acfb7-b6f2-441b-880c-278d7510e694) | +1 point, +1 snake's length |
| ![blueberry_sprite](https://github.com/Kitsumetri/SnakeGame/assets/100523204/d3e83c65-64c0-4e8d-a953-aea6273eae8c) | +2 points, +4 snake's speed |
| ![lemon_sprite](https://github.com/Kitsumetri/SnakeGame/assets/100523204/2b702514-febe-4099-806d-16a73f2d790e) | -1 snake's length |

## Security
All sprites were generated using DALL-E
