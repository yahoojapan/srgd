import os
import torch
import random
import glob
import numpy as np
import torchvision
import torchvision.transforms as T

from config import load_config
from model import get_model
from logzero import logger

from packaging import version
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

toPILImage = torchvision.transforms.ToPILImage()
toTensor = torchvision.transforms.ToTensor()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--conf', required=True,
                        help='Path to config file')
    parser.add_argument('-m', '--ckpt_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=250)
    parser.add_argument('--interpolation', type=str, default='bicubic')
    parser.add_argument('--cond_scale', type=float, default=1.0)
    parser.add_argument('--class_cond_scale', type=float, default=1.0)
    parser.add_argument('--guidance_start_steps', type=int, default=0)
    parser.add_argument('--class_guidance_start_steps', type=int, default=0)
    parser.add_argument('--generation_start_steps', type=int, default=0)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=None)
    parser.add_argument('--test_label', type=int, default=None)
    parser.add_argument('--no_amp', dest='amp', action="store_false")
    parser.add_argument('--no_dpmpp_solver', dest='use_dpmpp_solver', action="store_false")
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--backend', type=str, default='ddp')

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def sr_target_image(image, sr_model, scale=4, batch_size=8,
                    test_label=2, cond_scale=1.0, guidance_start_steps=0,
                    class_cond_scale=1.0, class_guidance_start_steps=0,
                    generation_start_steps=0, num_sample_steps=250,
                    enable_amp=False, interpolation='bicubic', seed=71):
    width, height = image.size

    if interpolation == 'bicubic':
        interpolation_mode = T.InterpolationMode.BICUBIC
    elif interpolation == 'lanczos':
        interpolation_mode = T.InterpolationMode.BICUBIC

    resize_hr_size = T.Resize((height*scale, width*scale), interpolation=interpolation_mode)

    resized_tensor = toTensor(resize_hr_size(image)).unsqueeze(0)
    condition_x = resized_tensor.to(sr_model.device)

    if test_label is not None:
        test_label = torch.LongTensor([test_label]).to(sr_model.device)
    else:
        test_label = None

    seed_everything(seed)

    # with torch.inference_mode(), autocast(enabled=enable_amp):
    with torch.inference_mode():
        output = sr_model.tiled_sample(batch_size=batch_size,
                                       condition_x=condition_x, class_label=test_label,
                                       cond_scale=cond_scale, guidance_start_steps=guidance_start_steps,
                                       class_cond_scale=class_cond_scale,
                                       class_guidance_start_steps=class_guidance_start_steps,
                                       generation_start_steps=generation_start_steps,
                                       num_sample_steps=num_sample_steps,
                                       amp=enable_amp)

    sr_img = toPILImage(output[0])
    new_width, new_height = sr_img.size
    assert width*4 == new_width
    assert height*4 == new_height
    return sr_img


def try_open_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except (IOError, SyntaxError) as e:
        return None

def batch_sr_target_images(input_dir, output_dir, sr_model, scale=4,
                           batch_size=8, test_label=2,
                           cond_scale=1.0, guidance_start_steps=0,
                           class_cond_scale=1.0, class_guidance_start_steps=0,
                           generation_start_steps=0, num_sample_steps=250,
                           start_index=0, end_index=None,
                           enable_amp=False, interpolation='bicubic', seed=71):

    print(f"save images at: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    image_list = sorted(glob.glob(f"{input_dir}/*"))[start_index:end_index]

    for filename in tqdm(image_list, disable=False):
        save_filename = os.path.basename(filename).replace('.png', '_out.png')
        save_path = os.path.join(output_dir, save_filename)

        if os.path.exists(save_path):
            print('skip')
        else:
            image = try_open_image(filename)
            if image is not None:
                cur_sr_img = sr_target_image(image, sr_model, scale=scale,
                                             batch_size=batch_size, test_label=test_label,
                                             cond_scale=cond_scale, guidance_start_steps=guidance_start_steps,
                                             class_cond_scale=class_cond_scale,
                                             class_guidance_start_steps=class_guidance_start_steps,
                                             generation_start_steps=generation_start_steps,
                                             num_sample_steps=num_sample_steps,
                                             enable_amp=enable_amp,
                                             interpolation=interpolation, seed=seed)
                cur_sr_img.save(save_path)
            else:
                print('Invalid image or unable to open image:', filename)


if __name__ == '__main__':
    args = parse_args()
    conf = load_config(args.conf)
    conf.num_sample_steps = args.num_sample_steps
    conf.ckpt_path = args.ckpt_path

    if version.parse(torch.__version__) < version.parse("2.0.0"):
        conf.flash_attn = False

    ema_model = get_model(conf, logger)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sr_model = ema_model.module.eval().to(device)

    print(args)

    batch_sr_target_images(args.input_dir, args.output_dir, sr_model,
                           scale=4, batch_size=args.batch_size, test_label=args.test_label,
                           cond_scale=args.cond_scale, guidance_start_steps=args.guidance_start_steps,
                           class_cond_scale=args.class_cond_scale,
                           class_guidance_start_steps=args.class_guidance_start_steps,
                           generation_start_steps=args.generation_start_steps,
                           num_sample_steps=args.num_sample_steps,
                           start_index=args.start_index, end_index=args.end_index,
                           enable_amp=args.amp, interpolation=args.interpolation, seed=args.seed)
