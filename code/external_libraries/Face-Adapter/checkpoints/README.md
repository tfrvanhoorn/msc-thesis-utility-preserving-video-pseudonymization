---
license: apache-2.0
---

# FaceAdapter Model Card

<div align="center">

[**Project Page**](https://faceadapter.github.io/face-adapter.github.io/) **|** [**Paper**](https://arxiv.org/pdf/2405.12970) **|** [**Code**](https://github.com/FaceAdapter/Face-Adapter) **|** [🤗 **Gradio demo**](https://huggingface.co/spaces/FaceAdapter/FaceAdapter)


</div>

## Introduction

Face-Adapter is an efficient and effective face editing adapter for pre-trained diffusion models, specifically targeting face reenactment and swapping tasks. 


<div  align="center">
<img src='__assets__/banner.gif'>
</div>


## Usage

You can directly download the model in this repository or download in python script:

```python
# Download a specific file
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="FaceAdapter/FaceAdapter", filename="controlnet/config.json", local_dir="./checkpoints")
# Download all files 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="FaceAdapter/FaceAdapter", local_dir="./checkpoints")
```
