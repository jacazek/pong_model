# pong_model
The dumbest game you might ever play
## Installation
1. Create a virtual environment  
`python -m venv .venv`


2. Activate the environment  
`source .venv/bin/active`


3. If installing for AMD GPU training/inference
   1. `pip install -r requirements-rocm.txt`
   2. For MI100 gpu, 
      1. clone flash attention repo if using flash attention and install  
      ```git clone https://github.com/Dao-AILab/flash-attention.git  dependencies/flash-attention```
      2. Navigate to flash_attention direcotry  
      ```cd dependecies/flash_attention```
      3. Modify the setup.py to include gfx908 in supported archs
      2. Install using ROCm environment  
      ```export GPU_ARCHS=gfx908 && rocm-python setup.py install```

5. If installing for Nvidia GPU
   1. `pip install flash-attn --no-build-isolation`


5. Install all other dependencies  
`pip install -r requirements.txt`

## Model configuration
To adjust model parameters, update the model_configuration.py

If no GPU, be sure to set `device` to `cpu`

## Training
Run the training script to generate a model  
`python trainer.py`

By default, RNNModel is trained. Provide the `--model_type` CLI arg to train a different model type.

Run `python trainer.py -h` to see all options for training.

Some model types such as TransformerModel use multiple processes while training.  
To prevent consuming all CPU, you can specify OMP_NUM_THREADS=4 to limit the number of threads.

## Test the model
Run the main script with desired generator
1. exact - generates states computed mathematically
2. fuzzy - generates states using model trained on states generated from engine

e.g.  
`python main.py --generator_type exact`

Run `python main.py -h` to see all options for running main script.

Some model types such as TransformerModel use multiple processes while running.  
To prevent consuming all CPU, you can specify OMP_NUM_THREADS=2 to limit the number of threads.  
`OMP_NUM_THREADS=2 python main.py`

The `--model_path` argument must be provided. This path is either an mlflow runs path or a relative path to local file

1. mlflow path example: 'runs:/000fc0c95642447899b50e9104b7f6a0/model_e44'
2. local path example: 'artifacts/000fc0c95642447899b50e9104b7f6a0/model_e44'

Loading a model from mlflow will cache the model in artifacts directory.

## Todo
- [ ] Capture metrics for model performance during training  
- [x] incorporate MLFlow for tracking progress
- [x] parameterize the model variant via CLI (and other runtime args)
- [x] Include bounding box collisions in the input data  
- [x] separate paddle control and scoring from ball engine
- [x] enable user control of paddles
- [x] introduce variability in generator to paddle movements  
- [ ] consider resetting ALL states to zero when ball resets so states prior to scoring don't affect ball behavior
- [ ] limit ball vector to certain degrees  
- [ ] provide extreme negative feedback the further ball goes out of bounds during training  
- [ ] provide extreme negative feedback for ball moving slowly or not at all  
- [ ] try out a couple different model architectures to see which might start to provide usable results
- [x] predict score (continuous integer mse) and hits as well (binary state cross-entropy)
- [ ] make sure all inputs to the model are standardized
  - currently position information is between 0 and 1 whereas velocity is between -1 and 1
- [ ] create separate training configuration file
  - include options to adjust generated paddle velocities to control even data generation
- [ ] introduce variability configuration setting for models to produce more unpredictable output
- [ ] Let the model control scaling factor of the game for extra glitchy experience
- [ ] Consider how to make the model control arbitrary number of balls...
  - multiple model instances?
  - Another model to determine how many instances should be provided?
- [x] Create common loop for fuzzy and exact engines (mainly paddle control)
  - Not a common loop, but setup of paddles has been encapsulated by factories
- [ ] Train the model on resetting the game when best of X reached
