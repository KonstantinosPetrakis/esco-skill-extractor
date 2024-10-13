import argparse

from flask import Flask, render_template, request, jsonify
from waitress import serve

from . import SkillExtractor


# ----------- Parse command line arguments -----------
parser = argparse.ArgumentParser(
    description="ESCO Skill Extractor: Extract ESCO skills from any text."
)
parser.add_argument(
    "--threshold",
    "-t",
    type=float,
    default=0.4,
    help="Threshold for skill extraction. Default is 0.4.",
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    default="cpu",
    help="Device to use for computations. Default is cpu.",
)
parser.add_argument(
    "--max_words",
    "-m",
    type=int,
    default=-1,
    help="Maximum number of words to process. Else summarization is used. -1 for no limit. Default is -1.",
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
extractor = SkillExtractor(
    threshold=args.threshold,
    device=args.device,
    max_words=args.max_words,
)

# ----------- Define the Flask app -----------
BASE_DIR = __file__.replace("__main__.py", "")
app = Flask(
    __name__,
    template_folder=BASE_DIR + "templates",
    static_folder=BASE_DIR + "static",
)


@app.route("/")
def index():
    return render_template("index.html", host=args.host, port=args.port)


@app.route("/extract", methods=["POST"])
def extract():
    texts = request.json

    # The tool doesn't do well with empty texts
    if all(not text for text in texts):
        return jsonify([[] * len(texts)])

    return jsonify(extractor.get_skills(texts))


# 20 minutes timeout, our model might take a while to infer for really big loads
print(f"Starting the server at http://{args.host}:{args.port}")
serve(
    app,
    host=args.host,
    port=args.port,
    channel_timeout=12000,
)
