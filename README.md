


<div align="center">

# Project Name - Demo for CIFAR10 model

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>
## Description


Build demo inference docker image timm model with Pytorch Lightning using -

1.)Pytorch Lightning
<br>
2.) CIFAR image set
<br>
3.) TIMM pre trained model
<br>
4.) Hydra
<br>
5.) Docker
<br>
6.) Model serialized into  TorchScript Scripted Model
<br>
6.)Model Trainied on  parameters obtained from experiment run - 

<ul>Optimizer - torch.optim.SGD </ul>
<ul>Learning Rate :  0.0224 [timm.yaml] </ul>
<ul>Batch Size : 128[cifar10.yaml] </ul>


## How to run

code available in master branch


Build CIFAR10 docker image - 
```bash

make build 

or 

docker build --tag emlv2_session4 .


```

Run CIFAR10 container

```bash
make run 

or 

docker run -p 8080:8080 emlv2_session4:latest

```

<br>


