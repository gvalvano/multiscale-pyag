
#  Self-supervised Multi-scale Consistency for Weakly Supervised Segmentation Learning  
  
Code for the paper:  
  
> Valvano G., Leo A. and Tsaftaris S. A. (DART, 2021), *Self-supervised Multi-scale Consistency for Weakly Supervised Segmentation Learning*.  
  
The official project page is [here](https://vios-s.github.io/multiscale-pyag/).  
An online version of the paper can be found [here](https://arxiv.org/abs/2108.11900).  

## Citation:  
```  
@incollection{valvano2021self,
  title={Self-supervised Multi-scale Consistency for Weakly Supervised Segmentation Learning},
  author={Valvano, Gabriele and Leo, Andrea and Tsaftaris, Sotirios A},
  booktitle={Domain Adaptation and Representation Transfer, and Affordable Healthcare and AI for Resource Diverse Global Health},
  pages={14--24},
  year={2021},
  publisher={Springer}
}
```  
  
<img src="https://github.com/vios-s/multiscale-pyag/blob/main/images/banner.png" alt="mscale_pyags" width="600"/>

----------------------------------  
  
## Notes:  
  
For the experiments, refer to: `experiments/acdc/exp_unet_pyag.py`. This file contains the main class that is used to train on the ACDC dataset. Please, refer to the class method `define_model()` to see how to correctly build the CNN architecture. The structure of the segmentor can be found under the folder `architectures`.
  
Once you download the [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) and the [scribble annotations](https://gvalvano.github.io/wss-multiscale-adversarial-attention-gates/data), you can pre-process it using the code in the file `data_interface/utils_acdc/prepare_dataset.py`. 
You can also train with custom datasets, but you must adhere to the template required by `data_interface/interfaces/dataset_wrapper.py`, which assumes the access to the dataset is through a tensorflow dataset iterator.

Once preprocessed the data, you can start the training/test of the model using `run.sh`.


## Requirements
This code was implemented using TensorFlow 1.14 and the libraries detailed in [requirements.txt](https://github.com/gvalvano/multiscale-pyag/requirements.txt).
You can install these libraries as:
`pip install -r requirements.txt`
or using conda (see [this](https://stackoverflow.com/questions/51042589/conda-version-pip-install-r-requirements-txt-target-lib)).

We tested the code on a TITAN Xp GPU, and on a GeForce GTX 1080, using CUDA 10.2. 

