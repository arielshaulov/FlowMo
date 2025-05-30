# FlowMo: Variance-Based Flow Guidance for Coherent Motion in Video Generation
<p align="center">
  <a href="https://arielshaulov.github.io/FlowMo/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
 <a href="https://arxiv.org/abs/2309.03884"><img src="https://img.shields.io/badge/arXiv-2306.00966-b31b1b.svg" height=20.5></a>
</p>

> **Ariel Shaulov\***, **Itay Hazan\***, **Lior Wolf**, **Hila Chefer**
> 
> Text-to-video diffusion models are notoriously limited in their ability to model temporal aspects such as motion, physics, and dynamic interactions. Existing approaches address this limitation by retraining the model or introducing external conditioning signals to enforce temporal consistency. In this work, we explore whether a meaningful temporal representation can be extracted directly from the predictions of a pre-trained model without any additional training or auxiliary inputs. We introduce FlowMo, a novel training-free guidance method that enhances motion coherence using only the model's own predictions in each diffusion step. FlowMo first derives an appearance-debiased temporal representation by measuring the distance between latents corresponding to consecutive frames. This highlights the implicit temporal structure predicted by the model. It then estimates motion coherence by measuring the patch-wise variance across the temporal dimension, and guides the model to reduce this variance dynamically during sampling. Extensive experiments across multiple text-to-video models demonstrate that FlowMo significantly improves motion coherence without sacrificing visual quality or prompt alignment, offering an effective plug-and-play solution for enhancing the temporal fidelity of pre-trained video diffusion models.


# Wan2.1

The official repository for **Wan2.1**, a high-quality text-to-video generation model.

## 📦 Getting Started

1. **Clone the repository**
   ````
   git clone https://github.com/arielshaulov/FlowMo.git
   ````
   
2. **Visit the repository**
   [https://github.com/Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
   and follow their instructions on order to run the Wan2.1 model

                    
## Run
````
python generate.py --task t2v-1.3B \
                    --size 832*480 \
                    --ckpt_dir path/to/model/weighs \
                    --prompts "A painter creating a landscape on canvas." \
                    --seeds 1024 \
                    --optimize "True" \
````

## Citation
If you find our code useful for your research, please consider citing our paper.
```
@article{flowmo2025,
  title={FlowMo: Variance-Based Flow Guidance for Coherent Motion in Video Generation},
  author={Shaulov, Ariel and Hazan, Itay and Wolf, Lior and Chefer, Hila},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
