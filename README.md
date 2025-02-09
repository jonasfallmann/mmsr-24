# MMSR24

## Setting up the Conda environment
To set up the Conda environment using the `environment.yml` file, run the following commands:

```sh
conda env create -f environment.yml
conda activate mmsr
```

## Download CLAP related datasets
To run calculation locally or rerun the precomputed similarities it is necessary to add CLAP related datasets into the dataset folder.

[Laion clap audio](https://drive.google.com/file/d/1TMqr1Cnh2ymEbAojpKHzx4n5UTo29buG/view?usp=drive_link)

[Laion clap song titles](https://drive.google.com/file/d/1n3c7fPwYhzYq7RSi9Oehn-98SUt8Z11F/view?usp=drive_link)

[Clap audio](https://drive.google.com/file/d/1AHpQtWh0urFjU7LtKyiiwNhXtezfz6yY/view?usp=drive_link)

[Clap song titles](https://drive.google.com/file/d/1BpJwFztRYTlVNaa86BITc7RhnTZtjBjX/view?usp=drive_link)

## Running the web interface
```sh
streamlit run web_interface.py
```

## LAION CLAP model
Download the model from https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt
