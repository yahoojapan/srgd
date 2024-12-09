# Real-SRGD: Enhancing Real-World Image Super-Resolution with Classifier-Free Guided Diffusion [ACCV2024]

[[Paper]](https://openaccess.thecvf.com/content/ACCV2024/html/Doi_Real-SRGD_Enhancing_Real-World_Image_Super-Resolution_with_Classifier-Free_Guided_Diffusion_ACCV_2024_paper.html)

This is the official PyTorch implementation of "Real-SRGD: Enhancing Real-World Image Super-Resolution with Classifier-Free Guided Diffusion (ACCV2024)".

## Installation

This repository uses Git LFS (Large File Storage) to manage large files. Please ensure you have Git LFS installed before cloning the repository. Follow the steps below to install Git LFS and set up the project:


1. **Install Git LFS**  
  If you don't have Git LFS installed, you can install it by following the instructions on the [Git LFS website](https://git-lfs.github.com/).  
  Alternatively, you can install it using a package manager:

  - **For macOS**:
    ```bash
    brew install git-lfs
    ```

  - **For Windows**:
    Download and run the [Git LFS installer](https://git-lfs.github.com/).

  - **For Linux**:
    Use your distribution's package manager. For example, on Ubuntu:
    ```bash
    sudo apt-get install git-lfs
    ```

2. **Initialize Git LFS**  
  After installing, initialize Git LFS in your repository:

    ```bash
    git lfs install
    ```

3. **Clone the repository**  
  Use Git to clone this repository. Please note that the download may take some time due to large files managed by Git LFS:

    ```bash
    git clone https://github.com/yahoojapan/srgd
    cd srgd
    ```

4. **Install packages**

    ```
    pip install -r requirements.txt
    ```

## Inference

### Step 1: Prepare testing data

Create a directory with an appropriate name and place all the images you want to super-resolve into this directory.  
This will be your `input_dir`.

### Step 2: Running testing command

Execute the following command to run the inference script. Make sure to specify your `input_dir` and `output_dir` paths accordingly:

```bash
input_dir=path/to/input_images
output_dir=path/to/output_images

conf="conf/conditional_continuous_linear_df8kost_dim128.yaml"
model="models/srgd/conditional_continuous_linear_df8kost_dim128_epoch300.pth"
test_label=0
class_cond_scale=1.0
seed=71

python inference.py -c ${conf} -m ${model} \
  --class_cond_scale ${class_cond_scale} --test_label ${test_label} --seed ${seed} \
  --input_dir ${input_dir} --output_dir ${output_dir}
```

Replace `path/to/input_images` with the path to your input directory and `path/to/output_images` with the path where you want the super-resolved images to be saved. This script will process the images in the input_dir and save the results to the output_dir.

A sample script `inference_sample.sh` is provided in the repository to help you get started with the inference process. You can modify this script to fit your specific needs.

## Citation

```
@inproceedings{doi2024,
  title={Real-SRGD: Enhancing Real-World Image Super-Resolution with Classifier-Free Guided Diffusion},
  author={Kenji Doi and Shuntaro Okada and Ryota Yoshihashi and Hirokatsu Kataoka},
  booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year={2024},
}
```

## License

[MIT](./LICENSE)
