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

You may need to reduce the batch size (`Config.batch_size`) to avoid out of memory
errors. If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by.

### Existing data loaders

To work from an example, you can see how this function is overloaded for the
different dataloaders we have already implemented:

- Blender (*)
- LLFF (*)
- RFFR (*)
- DTU dataset
- Tanks and Temples

(*) represent the datasets that have been tested.

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
