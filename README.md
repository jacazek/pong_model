# pong_model
The dumbest game you might ever play
## Installation
Create a virtual environment  
`python -m venv .venv`

Activate the environment  
`source .venv/bin/active`

If installing for AMD GPU training/inference  
`pip install -r requirements-rocm.txt`

Install dependencies  
`pip install -r requirements.txt`

## Model configuration
To adjust model parmaters, update the model_configuration.py

If no GPU, be sure to set `device` to `cpu`

## Training
Run the training script to generate a model  
`python trainer.py`

## Test the model
Run the main script with desired generator
1. engine - generates states computed mathematically
2. fuzzy_engine - generates states using model trained on states generated from engine

Update main.py to use the desired generator  
![alt text](docs/image.png "Image")  

then run the main script  
`python main.py`

## Todo
- [ ] Capture metrics for model performance during training  
- [x] Include bounding box collisions in the input data  
- [ ] separate paddle control and scoring from ball engine
- [ ] enable user control of paddles
- [x] introduce variability in generator to paddle movements  
- [ ] limit ball vector to certain degrees  
- [ ] provide extreme negative feedback the further ball goes out of bounds during training  
- [ ] provide extreme negative feedback for ball moving slowly or not at all  
- [ ] try out a couple different model architectures to see which might start to provide usable results
- [ ] predict score (continous integer mse) and hits as well (binary state cross-entropy)