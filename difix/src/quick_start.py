import argparse
from pipeline_difix import DifixPipeline
from diffusers.utils import load_image
import os
from tqdm import tqdm
from PIL import Image
import time

def main():
    parser = argparse.ArgumentParser(description="Run DifixPipeline on dataset.")
    parser.add_argument("--dir", type=str, required=True,
                        help="Root directory of DL3DV-val dataset")
    parser.add_argument("--w_ref", action="store_true",
                        help="Whether to use reference images")
    parser.add_argument("--eval_img_num", type=int, default=None,
                        help="Number of images to evaluate per folder (default: all)")
    args = parser.parse_args()

    dir = args.dir
    w_ref = args.w_ref
    eval_img_num = args.eval_img_num  # None means all

    if w_ref:
        pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    else:
        pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")

    gt_dir = os.path.join(dir, "gt")
    ref_dir = os.path.join(dir, "ref")
    renders_dir = os.path.join(dir, "renders")
    if w_ref:
        prediction_dir = os.path.join(dir, "VARPrediction", "difix_w_ref")
    else:
        prediction_dir = os.path.join(dir, "VARPrediction", "difix_wo_ref")
    os.makedirs(prediction_dir, exist_ok=True)

    render_files = sorted([f for f in os.listdir(renders_dir) if f.endswith('.png')])
    if eval_img_num is not None:
        render_files = render_files[:eval_img_num]

    for file in tqdm(render_files, desc=f"Processing {dir}"):
        render_path = os.path.join(renders_dir, file)
        output_path = os.path.join(prediction_dir, file)

        assert os.path.isfile(render_path)

        prompt = "remove degradation"
        input_image = load_image(render_path)
        if w_ref:
            ref_path = os.path.join(ref_dir, file)
            ref_image = load_image(ref_path)
            start_time = time.time()
            output_image = pipe(
                prompt,
                image=input_image,
                ref_image=ref_image,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0
            ).images[0]
        else:
            start_time = time.time()
            output_image = pipe(
                prompt,
                image=input_image,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0
            ).images[0]

        end_time = time.time()
        duration = end_time - start_time
        print(duration)

        output_image = output_image.resize(input_image.size, resample=Image.BICUBIC)
        output_image.save(output_path)


if __name__ == "__main__":
    main()
