import yaml
from dataclasses import dataclass


@dataclass
class Config:
    save_dir: str = 'srgd'
    prefix: str = 'conditional_continuous_linear'

    base_dir: str = './input/'
    dataset_name: str = 'cropped_df2kost_400x400_overlap200'

    model: str = 'continuous'  # gaussian / elucidated/ continuous
                               # conditional_gaussian / conditional_elucidated / conditional_continuous
                               # conditional_selfcond_gaussian / conditional_selfcond_continuous

    cond_drop_prob: float = 0.1
    cond_scale: float = 1.        # Classifier-free guidance scale for LR condition

    num_classes: int = 3
    conditional_task_type: str = 'realsr_denoise_sr'
    class_cond_drop_prob: float = 0.1
    class_cond_scale: float = 1.  # Classifier-free guidance scale for class condition
    test_label: int = 0

    guidance_start_steps: int = 0
    class_guidance_start_steps: int = 0
    generation_start_steps: int = 0

    # for GaussianDifussion
    objective: str = 'pred_noise'  # pred_noise / pred_x0 / pred_v
    beta_schedule: str = 'linear'  # linear / cosine / sigmoid
    timesteps: int = 1000
    sampling_timesteps: int = 250  # for DDIM sampling, less than 1000 means DDIM sampling
    offset_noise_strength: float = 0.

    loss_type: str = 'l2'  # l1 / l2 / smooth_l1

    # for ElucidatedDiffusion
    num_sample_steps: int = 32
    sigma_min: float = 0.002
    sigma_max: float = 80
    sigma_data: float = 0.5
    rho: float = 7
    P_mean: float = -1.2
    P_std: float = 1.2
    S_churn: float = 80
    S_tmin: float = 0.05
    S_tmax: float = 50
    S_noise: float = 1.003
    use_dpmpp_solver: bool = True

    # for ContinuousGaussianDiffusion
    noise_schedule: str = 'linear'  # linear / cosine / leanred
    clip_sample_denoised: bool = True
    learned_schedule_net_hidden_dim: int = 1024
    learned_noise_schedule_frac_gradient: float = 1.

    # for GaussianDiffusion and ContinuousGaussianDiffusion
    min_snr_loss_weight: bool = False
    min_snr_gamma: float = 5

    val_num_sample_steps: int = 32

    n_fold: int = 10  # Currently only 10 is supported
    train_fold: str = '0'  # Test with fold0 and train with the remaining folds

    skip_sample: bool = False  # Skip sampleing
    skip_val: bool = False     # Skip validation

    validation_ratio: float = 0.5  # Ratio of validation data used

    val_realsrv3: bool = False  # Validate with RealSRv3 dataset
    val_drealsr: bool = False   # Validate with DRealSR dataset
    val_realsrv3_scale: int = 4 # 2 / 4
    val_drealsr_scale: int = 4  # 2 / 4

    image_size: int = 128     # Image size for model input
    crop_size: int = 256      # Crop size from original image
    hr_image_size: int = 256  # High-resolution size
    lr_image_size: int = 128  # Low-resolution size
    crop_rate: int = 2        # Value of hr_image_size / image_size

    scale_size: int = 256     # Initial resize size during resize_randomcrop

    crop_size_limit: bool = False  # Filter images with a short side less than crop_size

    pixel_shuffle_upsample: bool = True

    batch_size: int = 32

    sample_size: int = 16

    hflip: bool = False   # 50% probability of horizontal flip
    rotate: bool = False  # Random rotation in 90-degree increments
    interpolation: str = 'BICUBIC'
    shuffle: bool = True

    torch_compile: bool = False  # Use torch.compile

    seed: int = 71

    amp: bool = False
    amp_dtype: str = 'float16'  # float16 / bfloat16(A100) / float32  # Currently unused

    # for U-Net
    unet_dim: int = 64
    ddpm_unet_dim_mults: str = '1,2,4,8'
    full_attn: str = 'False,False,False,True'
    learned_variance: bool = False  # Currently only False is supported
    learned_sinusoidal_cond: bool = True
    learned_sinusoidal_dim: int = 32

    ema_decay: float = 0.995
    ema_device: str = 'cuda'

    flash_attn: bool = False

    # load pretraind model manually
    ckpt_path: str = ''
    load_strict: bool = True

    # optimizer settings
    optimizer: str = 'adamw'
    lr: float = 1e-4
    min_lr: float = 1e-4
    weight_decay: float = 0.
    momentum: float = 0.9
    nesterov: bool = False
    amsgrad: bool = False
    madgrad_decoupled_decay: bool = True

    # scheduler settings
    epochs: int = 300
    warmup_epochs: int = 0
    warmup_lr_init: float = 1e-6
    plateau_mode: str = 'min'
    factor: float = 0.1
    patience: int = 4
    plateau_eps: float = 1e-8
    scheduler: str = 'cosine'  # ReduceLROnPlateau / CosineAnnealingLR /
                               # WarmupLinear / cosine
    cosine_interval_type: str = 'step'  # step or epoch  Frequency of CosineLRScheduler

    # Crop method from original image: centercrop / randomcrop / justresize / resize_randomcrop
    train_preprocess: str = 'randomcrop'
    valid_preprocess: str = 'centercrop'

    train_trans_mode: str = 'realesrgan'  # simple / aug_v1 / aug_v2 / realesrgan
                                          # When realesrgan is specified, train_preprocess is ignored
    valid_trans_mode: str = 'simple'      # simple

    usm_sharpener: bool = False           # Whether to apply unsharpmask to HR images when realesrgan is specified

    interpolation: str = 'BICUBIC'  # BILINEAR / BICUBIC / LANCZOS

    # for aug_v1 / aug_v2
    blur_prob: float = 0.5  # prob of OneOf
    advance_blur_prob: float = 0.5
    gaussian_blur_prob: float = 0.5
    sinc_blur_prob: float = 0.5
    sinc_blur_factor_min: float = 0.9
    sinc_blur_factor_max: float = 1.1
    image_compression_prob: float = 0.5  # prob of image compression
    quality_lower: int = 50
    quality_upper: int = 100
    noise_prob: float = 0.5  # prob of noise
    gauss_noise_prob: float = 0.5
    iso_noise_prob: float = 0.5
    multiplicative_noise_prob: float = 0.5

    train: bool = True
    test: bool = False
    debug: bool = False

    save_validation_sample: bool = False  # Save sample images during validation
    save_validation_hr_sample: bool = False  # Save HR sample images during validation

    save_every_epoch: bool = False  # Save model at every epoch

    test_target: str = 'best_loss'  # best_loss / best_psnr / best_ssim / best_lpips

    num_workers: int = 4
    device: str = 'cuda'
    pin_memory: bool = True
    model_dir: str = 'models'
    log_dir: str = 'logs'
    print_freq: int = 0


def load_config(config_file):
    with open(config_file, 'r') as fp:
        opts = yaml.safe_load(fp)
    return Config(**opts)
