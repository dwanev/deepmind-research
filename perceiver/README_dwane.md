

# Perceiver IO

https://github.com/lucidrains/perceiver-pytorch  
code ~/projects/deepmind-research/perciever  

Repo was forked into dwanev, and checked out in projects


## Environment setup

python3 -m venv ~/.venv/perceiver  
source ~/.venv/perceiver/bin/activate  
pip install --upgrade pip  
pip install jaxlib==0.1.71
pip install "jax[cpu]"==0.2.20
pip install -r perceiver/requirements.txt  
AttributeError: module 'setuptools' has no attribute 'find_namespace_packages'  
pip install --upgrade setuptools  
pip install -r perceiver/requirements.txt




## Link virtual env with Jupyter notebook

pip install ipykernel  
python -m ipykernel install --user --name=perceiver  

## Running

cd ~/projects/deepmind-research  
source ~/.venv/perceiver/bin/activate

jupyter notebook

perceiver_masked_language_modelling.ipynb



## Training 

cd ~/projects/deepmind-research  
source ~/.venv/perceiver/bin/activate  
./perceiver/train/launch_local.sh

## Dummy Data : Custom Tensorflow Datasets

https://www.tensorflow.org/datasets/add_dataset

tfds --help
tfds new random_image_dataset
