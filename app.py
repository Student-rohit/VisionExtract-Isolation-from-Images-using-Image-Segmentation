from flask import Flask, render_template, request
import os

from inference import isolate_subject

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
OUTPUT_FOLDER = "static/outputs/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET","POST"])

def index():

    if request.method == "POST":

        file = request.files["image"]

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)

        file.save(filepath)

        output_path = os.path.join(OUTPUT_FOLDER, file.filename)

        isolate_subject(filepath, output_path)

        return render_template("index.html",
                               input_image=filepath,
                               output_image=output_path)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)