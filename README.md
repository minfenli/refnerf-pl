# refnerf-pytorch-lighting

This is an implementation of [Ref-NeRF](https://dorverbin.github.io/refnerf/), which is extented from [refnerf-pytorch](https://github.com/gkouros/refnerf-pytorch) based on the original [jax code](https://github.com/google-research/multinerf) released by Google.

## Setup

```
# Clone the repo.
git clone https://github.com/gkouros/refnerf-pytorch.git
cd refnerf-pytorch

# Make a conda environment.
conda create --name refnerf-pl python=3.9
conda activate refnerf-pl
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
```

## Running

Example scripts for training, evaluating, and rendering can be found in `scripts/`. You'll need to change the paths to point to wherever the datasets
are located. [Gin](https://github.com/google/gin-config) configuration files for our model and some ablations can be found in `configs/`.

### OOM errors

You may need to reduce the batch size (`Config.batch_size`) to avoid out-of-memory errors. If you do this but want to preserve quality, be sure to increase the number of training iterations and decrease the learning rate by whatever scale factor you decrease the batch size by.

### Existing data loaders

To work from an example, you can see how this function is overloaded for the different dataloaders we have already implemented:

- Blender (*)
- LLFF (*)
- RFFR (*)
- DTU dataset
- Tanks and Temples

(*) represent the datasets that have been tested.


## Hyper-Parameter Tuning

We found that in forward-facing settings, NeRF models may generate results with poor geometric properties (e.g. predicting all normals facing forward, no density in white background, etc.). Therefore, we add several geometric losses and consistency losses to constrain the model to satisfy geometric priors, references for these ideas can be found in the list of papers below.

| | Ref-NeRF | Ref-NeRF + Geometry losses |
| :---: | :---: | :---:|
| RGB | <video width="50%" src="https://user-images.githubusercontent.com/33437552/235092130-cfa2f352-b241-46ec-98ed-a1af42fef4ae.mp4"> | <video width="50%" src="https://user-images.githubusercontent.com/33437552/235091995-daa678a4-bb34-4ee1-8d62-b83593cf0540.mp4"> |
| Accumulated Density | <video width="50%" src="https://user-images.githubusercontent.com/33437552/235092113-39d80106-2947-42d6-b586-1276e39f463e.mp4"> | <video width="50%" src="https://user-images.githubusercontent.com/33437552/235091551-eb821904-8e3b-4588-9c7e-84e2ba79f6e1.mp4"> |
| Normal | <video width="50%" src="https://user-images.githubusercontent.com/33437552/235092339-043ae1a5-36a2-423d-894a-4e03363b4f6a.mp4"> | <video width="50%" src="https://user-images.githubusercontent.com/33437552/235091817-bc62bd84-7b1c-4195-808f-be6a4ca24b53.mp4"> |
| Median Distance | <video width="50%" src="https://user-images.githubusercontent.com/33437552/235092233-5026a566-a239-44bb-9f02-e13e7392e4a2.mp4"> | <video width="50%" src="https://user-images.githubusercontent.com/33437552/235091801-4ff9ec5c-4bc8-44a1-b88a-71910b0894f1.mp4"> |
| PSNR(↑) / SSIM(↑) / LPIPS(↓) | 26.310 / 0.862 / 0.205 | **26.395** / **0.866** / **0.199** |

In our tests, improvements were not always guaranteed across different experimental settings with a given parameter setting. It still needs to be adjusted in different scenarios.

## Awesome NeRF with Geometry losses
* [Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields](https://dorverbin.github.io/refnerf/) (CVPR 2022)
* [Scalable Neural Indoor Scene Rendering](https://xchaowu.github.io/papers/scalable-nisr/) (SIGGRAPH 2022)
* [NeRFReN: Neural Radiance Fields with Reflections](https://bennyguo.github.io/nerfren/) (CVPR 2022)

The following papers mainly discuss issues of few-shot NeRF (training NeRF with limited input images).

* [Ray Priors through Reprojection: Improving Neural Radiance Fields for Novel View Extrapolation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Ray_Priors_Through_Reprojection_Improving_Neural_Radiance_Fields_for_Novel_CVPR_2022_paper.pdf) (CVPR 2022)
* [InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering](https://cv.snu.ac.kr/research/InfoNeRF/) (CVPR 2022)
* [GeoAug: Data Augmentation for Few-Shot NeRF with Geometry Constraints](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770326.pdf) (ECCV 2022)
* [GeCoNeRF: Few-shot Neural Radiance Fields via Geometric Consistency](https://ku-cvlab.github.io/GeCoNeRF/) (ICML 2023)
* [Dense Depth Priors for Neural Radiance Fields from Sparse Input Views](https://barbararoessle.github.io/dense_depth_priors_nerf/) (CVPR 2022)
* [Depth-supervised NeRF: Fewer Views and Faster Training for Free](https://www.cs.cmu.edu/~dsnerf/) (CVPR 2022)
* [RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse  Inputs](https://www.cs.cmu.edu/~dsnerf/) (CVPR 2022)
* [Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis](https://ajayj.com/dietnerf/) (ICCV 2021)
* [DiffusioNeRF: Regularizing Neural Radiance Fields with Denoising Diffusion Models](https://github.com/nianticlabs/diffusionerf) (Arxiv 2023)


## References
```
@article{verbin2022refnerf,
    title={{Ref-NeRF}: Structured View-Dependent Appearance for
           Neural Radiance Fields},
    author={Dor Verbin and Peter Hedman and Ben Mildenhall and
            Todd Zickler and Jonathan T. Barron and Pratul P. Srinivasan},
    journal={CVPR},
    year={2022}
}
```
```
@misc{refnerf-pytorch,
      title={refnerf-pytorch: A port of Ref-NeRF from jax to pytorch},
      author={Georgios Kouros},
      year={2022},
      url={https://github.com/google-research/refnerf-pytorch},
}
```
```
@misc{multinerf2022,
      title={{MultiNeRF}: {A} {Code} {Release} for {Mip-NeRF} 360, {Ref-NeRF}, and {RawNeRF}},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
      url={https://github.com/google-research/multinerf},
}
```
