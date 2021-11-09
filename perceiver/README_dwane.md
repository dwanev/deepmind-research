

# Perceiver IO

https://github.com/lucidrains/perceiver-pytorch  
code ~/projects/deepmind-research/perciever  

Repo was forked into dwanev, and checked out in projects


## Environment setup (ubuntu )

python3 -m venv ~/.venv/perceiver  
source ~/.venv/perceiver/bin/activate  
pip install --upgrade pip  
pip install jaxlib==0.1.71  
pip install "jax[cpu]"==0.2.20  
pip install -r perceiver/requirements.txt 

(on my ubuntu machine I got the following error, on my mac I did not:) 
AttributeError: module 'setuptools' has no attribute 'find_namespace_packages'  
pip install --upgrade setuptools    
pip install -r perceiver/requirements.txt  


## Link virtual env with Jupyter notebook (ubuntu)

python3 -m pip install ipykernel  
python3 -m ipykernel install --user --name=perceiver  

## Running (ubuntu)

cd ~/projects/deepmind-research  
source ~/.venv/perceiver/bin/activate
jupyter notebook
perceiver_masked_language_modelling.ipynb



## Environment setup ( mac )
python3 -m venv ~/.venv/perceiver  
source ~/.venv/perceiver/bin/activate  
python3 -m pip install --upgrade pip  
python3 -m pip install jaxlib==0.1.71  
python3 -m pip install "jax[cpu]"==0.2.20  
python3 -m pip install -r perceiver/requirements.txt 
python3 -m pip install ipykernel
python3 -m ipykernel install --user --name=perceiver2  

### running ( mac )

source ~/.venv/perceiver/bin/activate  
jupyter notebook
select kernel perceiver2




## Training 

cd ~/projects/deepmind-research  
source ~/.venv/perceiver/bin/activate  
./perceiver/train/launch_local.sh

## Dummy Data : Custom Tensorflow Datasets

https://www.tensorflow.org/datasets/add_dataset

tfds --help
tfds new random_image_dataset
