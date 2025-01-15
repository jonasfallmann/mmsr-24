# MMSR24

## Setting up the Conda environment
To set up the Conda environment using the `environment.yml` file, run the following commands:

```sh
conda env create -f environment.yml
conda activate mmsr
```

## Running the web interface
```sh
streamlit run web_interface.py
```

## LAION CLAP model
Download the model from https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt