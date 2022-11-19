import os

from flask_cors import CORS, cross_origin
from flask import Flask, request, send_from_directory, make_response

# Flask boiler-plate
app = Flask(__name__)
cors = CORS(app)
folder_path = './'
    
@app.route('/image', methods=['GET'])
@cross_origin()
def image():
    args = request.args
    input_text = args.get("q", default="coffee devil", type=str)
    try:
        exec(open('../scripts/txt2img.py').read(), 
            {
                'prompt': f"{input_text}",
                'outdir': "app/output",
                'n_iter': 1,
                'n_samples': 1,
                'H': 256,
                'W': 256
            })
        #exec(open(f'../scripts/txt2img.py --prompt "{input_text}" --outdir "app/output" --n_iter 1 --n_samples 1 --H 256 --W 256').read())
        # os.system(f'python ../scripts/txt2img.py --prompt "{input_text}" --outdir "app/output" --n_iter 1 --n_samples 1 --H 256 --W 256')

        try:
            return send_from_directory(folder_path, "test.jpg", as_attachment=True)
        except FileNotFoundError:
            return make_response("File not found", 500)
    except:
        return make_response("Error in the stable-diffusion processor", 500)

if __name__ == '__main__':
    app.run(threaded=False)