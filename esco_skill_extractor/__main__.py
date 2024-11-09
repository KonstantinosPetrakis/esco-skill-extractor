import argparse

from flask import Flask, render_template, request, jsonify
from waitress import serve

from . import SkillExtractor


# ----------- Parse command line arguments -----------
parser = argparse.ArgumentParser(
    description="ESCO Skill Extractor: Extract ESCO skills and ISCO occupations from any text."
)
parser.add_argument(
    "--threshold",
    "-t",
    type=float,
    default=0.4,
    help="Threshold for entity extraction. Default is 0.4.",
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    default="cpu",
    help="Device to use for computations. Default is cpu.",
)
parser.add_argument(
    "--host",
    "-c",
    type=str,
    default="localhost",
    help="Host to bind the server to. Default is localhost",
)
parser.add_argument(
    "--port",
    "-p",
    type=int,
    default=8000,
    help="Port to bind the server to. Default is 8000",
)

args = parser.parse_args()

# ----------- Initialize the skill extractor -----------
extractor = SkillExtractor(threshold=args.threshold, device=args.device)

# ----------- Define the Flask app -----------
BASE_DIR = __file__.replace("__main__.py", "")
app = Flask(
    __name__,
    template_folder=BASE_DIR + "templates",
    static_folder=BASE_DIR + "static",
)


@app.after_request
def handle_options(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
    return response


@app.route("/")
def index():
    return render_template("index.html", host=args.host, port=args.port)


@app.route("/extract-skills", methods=["POST"])
def extract():
    return jsonify(extractor.get_skills(request.json))


@app.route("/extract-occupations", methods=["POST"])
def extract_occupations():
    return jsonify(extractor.get_occupations(request.json))


# 20 minutes timeout, our model might take a while to infer for really big loads
print(f"Starting the server at http://{args.host}:{args.port}")
serve(
    app,
    host=args.host,
    port=args.port,
    channel_timeout=12000,
)
