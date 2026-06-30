from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import json
import os
import cv2
import numpy as np
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

device = torch.device("cpu")

# ---------------- CREATE FOLDERS ----------------

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ---------------- FILE VALIDATION ----------------

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- LOAD MODELS ----------------

def load_model(model_path, class_path, num_classes):

    with open(class_path, "r") as f:
        classes = json.load(f)

    model = models.resnet18(pretrained=False)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    return model, classes


brain_model, brain_classes = load_model(
    "models/brain_model.pth",
    "models/brain_classes.json",
    4
)

lung_model, lung_classes = load_model(
    "models/lung_model_best.pth",
    "models/lung_classes.json",
    2
)

# ---------------- IMAGE TRANSFORM ----------------

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# ---------------- GRAD CAM ----------------

def grad_cam(model, image, target_layer):

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(image)

    pred = output.argmax()

    model.zero_grad()

    output[0, pred].backward()

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1,2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)

    cam = cv2.resize(cam, (224,224))

    if cam.max() != 0:
        cam = cam / cam.max()

    f_handle.remove()
    b_handle.remove()

    return cam

# ---------------- HEALTH ROUTE ----------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Cancer Detection API Running"
    })

# ---------------- PREDICT ROUTE ----------------

@app.route("/predict", methods=["POST"])

def predict():

    try:

        # check file
        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "error": "Empty filename"
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type"
            }), 400

        # cancer type
        cancer_type = request.form.get("type")

        if cancer_type not in ["brain", "lung"]:
            return jsonify({
                "error": "Invalid cancer type"
            }), 400

        # save file
        filename = secure_filename(file.filename)

        filepath = os.path.join(
            "uploads",
            filename
        )

        file.save(filepath)

        # open image
        img = Image.open(filepath).convert("RGB")

        x = transform(img).unsqueeze(0)

        # select model
        if cancer_type == "brain":

            model = brain_model
            classes = brain_classes

        else:

            model = lung_model
            classes = lung_classes

        # prediction
        with torch.no_grad():

            output = model(x)

            probs = torch.softmax(output, dim=1)[0]

            pred = probs.argmax().item()

        confidence = float(probs[pred])

        # uncertainty threshold
        if confidence < 0.60:

            prediction = "Uncertain Prediction"

        else:

            prediction = classes[pred]

        # gradcam
        cam = grad_cam(
            model,
            x,
            model.layer4
        )

        img_np = np.array(
            img.resize((224,224))
        )

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        overlay = np.clip(
            heatmap * 0.4 + img_np,
            0,
            255
        ).astype(np.uint8)

        # unique gradcam filename
        gradcam_filename = f"{uuid.uuid4().hex}.jpg"

        gradcam_path = os.path.join(
            "static",
            gradcam_filename
        )

        cv2.imwrite(
            gradcam_path,
            overlay
        )

        return jsonify({

            "type": cancer_type,

            "prediction": prediction,

            "confidence": round(
                confidence * 100,
                2
            ),

            "gradcam": gradcam_path

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ---------------- RUN APP ----------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
