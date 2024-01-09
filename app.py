from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import base64
import io
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from datetime import datetime

app = Flask(__name__)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, variant="fp16", torch_dtype=torch.float16, token="YOUR HUGGING FACE TOKEN HERE")
pipe.to(device)
max_length = pipe.tokenizer.model_max_length

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt_text = request.form['prompt']
    negative_prompt_text = request.form['negative_prompt']
    height = int(request.form['height'])
    width = int(request.form['width'])

    input_ids = pipe.tokenizer(prompt_text, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")

    negative_ids = pipe.tokenizer(negative_prompt_text, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids
    negative_ids = negative_ids.to("cuda")
    concat_embeds = []
    neg_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
    images = []
    for _ in range(4):
        with autocast(device):
            image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, guidance_scale=8.5, height=height, width=width)["images"][0]
        images.append(image)

    encoded_images = {}
    for i, img in enumerate(images, start=1):
        buffered = io.BytesIO()
        img.save(buffered, format='PNG')
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        encoded_images[f'img{i}'] = img_str

    return jsonify(encoded_images)

@app.route('/save_image', methods=['POST'])
def save_image():
    pass
    # This function should handle saving the image to disk, similar to your save_specific_image function

if __name__ == '__main__':
    app.run(debug=True)
