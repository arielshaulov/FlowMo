# FlowMo: Variance-Based Flow Guidance for Coherent Motion in Video Generation
<p align="center">
  <a href="https://arielshaulov.github.io/zero-shot-audio-captioning/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
 <a href="https://arxiv.org/abs/2309.03884"><img src="https://img.shields.io/badge/arXiv-2306.00966-b31b1b.svg" height=20.5></a>

</p>

> **Ariel Shaulov\***, **Itay Hazan\***, **Lior Wolf**, **Hila Chefer**
> 
> Text-to-video diffusion models are notoriously limited in their ability to model temporal aspects such as motion, physics, and dynamic interactions. Existing approaches address this limitation by retraining the model or introducing external conditioning signals to enforce temporal consistency. In this work, we explore whether a meaningful temporal representation can be extracted directly from the predictions of a pre-trained model without any additional training or auxiliary inputs. We introduce FlowMo, a novel training-free guidance method that enhances motion coherence using only the model's own predictions in each diffusion step. FlowMo first derives an appearance-debiased temporal representation by measuring the distance between latents corresponding to consecutive frames. This highlights the implicit temporal structure predicted by the model. It then estimates motion coherence by measuring the patch-wise variance across the temporal dimension, and guides the model to reduce this variance dynamically during sampling. Extensive experiments across multiple text-to-video models demonstrate that FlowMo significantly improves motion coherence without sacrificing visual quality or prompt alignment, offering an effective plug-and-play solution for enhancing the temporal fidelity of pre-trained video diffusion models.


## Requirements
````
conda env create -f environment.yml
````

## Run
````
# LanguageBind
python main.py --video_dir_path "" --audio_dir_path "" --gpu_id 0 --backbone language_bind --candidates_file_path "" --alpha 0.5 --filter_threshold 0.55 --threshold_stage1 0.75 --threshold_stage2 0.75 --gamma 2.5 --dataset LLP/AVE --method bbse-cosine --fusion early

# CLIP & CLAP
python main.py --video_dir_path "" --audio_dir_path "" --gpu_id 0 --backbone clip_clap --candidates_file_path "" --alpha 0.45 --filter_threshold 0.5 --threshold_stage1 0.75 --threshold_stage2 0.75 --gamma 1 --dataset LLP/AVE --method bbse-cosine --fusion early
````

## Citation
If you find our code useful for your research, please consider citing our paper.
```
@misc{shaar2025adaptingunknowntrainingfreeaudiovisual,
      title={Adapting to the Unknown: Training-Free Audio-Visual Event Perception with Dynamic Thresholds}, 
      author={Eitan Shaar and Ariel Shaulov and Gal Chechik and Lior Wolf},
      year={2025},
      eprint={2503.13693},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.13693}, 
}
```
