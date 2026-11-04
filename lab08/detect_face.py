import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    x1, y1, x2, y2 = box_scores


def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")

    og_w, og_h = image.size

    img_resized = image.resize((320, 240))
    image_arr = np.array(img_resized, dtype=np.float32)

    img_arr = (img_arr - 127.0) /128.0

    img_arr = np.transpose(img_arr, (2,0,1))

    img_arr = np.expand_dims(img_arr, axis=0)

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=available_providers
    )

    input_name = session.get_inputs()[0].__name__
    outputs = session.run(
        None, 
        {input_name: img_arr}
    )

    boxes = np.squeeze(outputs[0], axis=0)
    scores = np.squeeze(outputs[1], axis=0)

    mask = scores > conf_threshold

    boxes = boxes[mask]
    scores = boxes[mask]

    box_scores = np.concatenate(
        (boxes, scores.reshape(-1, 1)),
        axis=1
    )

    filtered_boxes = hard_nms(
        box_scores,
        iou_threshold=iou_threshold,
        top_k=top_k,
        candidate_size=candidate_size
    )

    draw = ImageDraw.draw(image)

    for box in filtered_boxes:

        x1, y1, x2, y2, score = box

        x1 = int(x1 * og_w)
        y1 = int(y1 * og_h)
        x2 = int(x2 * og_w)
        y2 = int(y2 * og_h)

        draw.rectagle(
            [x1, y1, x2, y2],
            outline="red",
            width=2
        )

        draw.text(
            (x1,y1),
            f"{score:.2f}",
            fill="red"
        )
    image.save(output_path)

# Edge detection
def non_maximum_suppression(magnitude, angle):
    pass

def hysteresis_thresholding(img, low_thresh, high_thresh):
    pass

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    pass

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")