import socket
from flask import Flask, request, send_from_directory, make_response

# Flask boiler-plate
app = Flask(__name__)
folder_path = './'
    
@app.route('/image', methods=['GET'])
def image():
    args = request.args
    input_text = args.get("q", default="", type=str)
    response = make_response(send_from_directory(folder_path,
                               'test.jpg', as_attachment=True))

    return response, 200

if __name__ == '__main__':
    app.run(threaded=False)