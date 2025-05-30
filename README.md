# 🎬 FlowMo: Variance-Based Flow Guidance for Coherent Motion in Video Generation

<div align="center">
<a href="https://arielshaulov.github.io/FlowMo/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
 <a href="https://github.com/arielshaulov/video-motion"><img src="https://img.shields.io/badge/arXiv-2306.00966-b31b1b.svg" height=20.5></a>
</div>

## 📝 Abstract

Text-to-video diffusion models are notoriously limited in their ability to model temporal aspects such as motion, physics, and dynamic interactions. Existing approaches address this limitation by retraining the model or introducing external conditioning signals to enforce temporal consistency. 

**FlowMo** explores whether a meaningful temporal representation can be extracted directly from the predictions of a pre-trained model without any additional training or auxiliary inputs. Our novel training-free guidance method enhances motion coherence using only the model's own predictions in each diffusion step.

### 🔬 Key Innovations

- **Appearance-debiased temporal representation** by measuring distances between consecutive frame latents
- **Motion coherence estimation** through patch-wise variance measurement across temporal dimensions  
- **Dynamic variance reduction** guidance during sampling
- **Plug-and-play solution** requiring no additional training

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-compatible GPU (2 x H100)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/arielshaulov/video-motion.git
   cd video-motion
   ```

2. **Set up Wan2.1 model**
   
   Visit the official [Wan2.1 repository](https://github.com/Wan-Video/Wan2.1) and follow their setup instructions to obtain the model weights.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎯 Usage

### Basic Text-to-Video Generation

```bash
python generate.py --task t2v-1.3B \
                   --size 832*480 \
                   --ckpt_dir path/to/model/weights \
                   --prompts "A painter creating a landscape on canvas." \
                   --seeds 1024 \
                   --optimize "True"
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--task` | Model task specification | `t2v-1.3B` |
| `--size` | Output video resolution | `832*480` |
| `--ckpt_dir` | Path to model weights | Required |
| `--prompts` | Text prompt for generation | Required |
| `--seeds` | Random seed for reproducibility | `1024` |
| `--optimize` | Enable FlowMo optimization | `True` |

---

## 📊 Results

FlowMo demonstrates significant improvements in:
- ✅ **Motion coherence** across various text-to-video models
- ✅ **Temporal consistency** without sacrificing visual quality
- ✅ **Prompt alignment** maintained at original levels
- ✅ **Plug-and-play compatibility** with existing models

---

## 🚧 Future Work & TODO

- [x] **Release Wan based code**
- [ ] **Release Cog based code**
- [ ] **Freeinit code**


---

## 🤝 Contributing

We welcome contributions! Please feel free to:
- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests
- 📖 Improve documentation

---

## 📚 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@article{flowmo2025,
  title={FlowMo: Variance-Based Flow Guidance for Coherent Motion in Video Generation},
  author={Shaulov, Ariel and Hazan, Itay and Wolf, Lior and Chefer, Hila},
  journal={arXiv preprint arXiv:2309.03884},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Thanks to the [Wan2.1](https://github.com/Wan-Video/Wan2.1) team for their excellent text-to-video model

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[🌐 Website](https://arielshaulov.github.io/FlowMo/) • [📖 Paper](https://arxiv.org/abs/2309.03884) • [💬 Issues](https://github.com/arielshaulov/video-motion/issues)

</div>
