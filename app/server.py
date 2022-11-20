import os
import io

import torch
from torch import autocast

from diffusers import StableDiffusionPipeline
from flask_cors import CORS, cross_origin
from flask import Flask, request, send_file, make_response

# Flask boiler-plate
app = Flask(__name__)
cors = CORS(app)

hub_token = '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'
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=hub_token)  

def run_inference(prompt):
  with autocast("cuda"):
      image = pipe(prompt).images[0]  
  img_data = io.BytesIO()
  image.save(img_data, "PNG")
  img_data.seek(0)
  return img_data

@app.route('/image', methods=['GET'])
@cross_origin()
def image():
    args = request.args
    input_text = args.get("q", default="coffee devil", type=str)

    try:
        # os.system(f'python ../scripts/txt2img.py --prompt "{input_text}" --outdir "app/output" --n_iter 1 --n_samples 1 --H 256 --W 256')
        try:
            img_data = run_inference(input_text)
            return send_file(img_data, mimetype='image/jpg')
        except FileNotFoundError:
            return make_response("File not found", 500)
    except Exception as e:
        print(e)
        return make_response("Error in the stable-diffusion processor", 500)

if __name__ == '__main__':
    app.run(threaded=False)