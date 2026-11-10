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
    scores = box_scores[:, 4]
    boxes = box_scores[:, :4]

    indexes = np.argsort(scores)

    indexes = indexes[-candidate_size:]

    picked = []

    while len(indexes) > 0:

        current = indexes[-1]
        picked.append(current)
        if top_k > 0 and len(picked) >= top_k:
            break

        if len(indexes) == 1:
            break

        indexes = indexes[:-1]

        current_box = boxes[current]

        x1 = current_box[0]
        y1 = current_box[1]
        x2 = current_box[2]
        y2 = current_box[3]

        remaining_boxes = boxes[indexes]

        inter_x1 = np.maximum(x1, remaining_boxes[:, 0])
        inter_y1 = np.maximum(y1, remaining_boxes[:, 1])
        inter_x2 = np.minimum(x2, remaining_boxes[:, 2])
        inter_y2 = np.minimum(y2, remaining_boxes[:, 3])

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)

        intersection = inter_w * inter_h

        current_area = (x2 - x1) * (y2 - y1)

        remaining_area = (
            (remaining_boxes[:, 2] - remaining_boxes[:, 0]) *
            (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        )

        union = current_area + remaining_area - intersection

        iou = intersection / np.maximum(union, 1e-6)

        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked]


def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")

    og_w, og_h = image.size

    img_resized = image.resize((320, 240))
    img_arr = np.asarray(img_resized, dtype=np.float32)

    img_arr = (img_arr - 127.0) / 128.0
    img_arr = np.transpose(img_arr, (2, 0, 1))
    img_arr = np.expand_dims(img_arr, axis=0)

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=available_providers
    )

    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: img_arr}
    )
    print("Number of outputs:", len(outputs))

    for i, output in enumerate(outputs):
        print(f"Output {i} shape:", output.shape)
        print(f"Output {i} first values:", output.flatten()[:10])

    scores = np.squeeze(outputs[0], axis=0)
    boxes = np.squeeze(outputs[1], axis=0)

    scores = scores[:, 1]

    mask = scores > conf_threshold

    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        print("No faces detected.")
        image.save(output_path)
        return

    box_scores = np.concatenate(
        (boxes, scores.reshape(-1, 1)),
        axis=1
    )

    filtered_boxes = hard_nms(
        box_scores,
        iou_threshold=conf_threshold,
    )

    draw = ImageDraw.Draw(image)

    for box in filtered_boxes:
        x1, y1, x2, y2, score = box

        x1 = int(x1 * og_w)
        y1 = int(y1 * og_h)
        x2 = int(x2 * og_w)
        y2 = int(y2 * og_h)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=2
        )

        draw.text(
            (x1, y1),
            f"{score:.2f}",
            fill="red"
        )

    image.save(output_path)

    print(f"Face detection result saved to {output_path}")

# Edge detection
def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape

    output = np.zeros((height, width), dtype=np.float32)

    # Convert radians to degrees
    angle = np.degrees(angle)

    # Convert negative angles into [0, 180)
    angle[angle < 0] += 180

    # Examine every pixel except the boundary
    for i in range(1, height - 1):

        for j in range(1, width - 1):

            direction = angle[i, j]

            # Horizontal direction: 0 degrees
            if (
                (0 <= direction < 22.5) or
                (157.5 <= direction <= 180)
            ):

                neighbor1 = magnitude[i, j - 1]
                neighbor2 = magnitude[i, j + 1]

            # Diagonal direction: 45 degrees
            elif 22.5 <= direction < 67.5:

                neighbor1 = magnitude[i + 1, j - 1]
                neighbor2 = magnitude[i - 1, j + 1]

            # Vertical direction: 90 degrees
            elif 67.5 <= direction < 112.5:

                neighbor1 = magnitude[i - 1, j]
                neighbor2 = magnitude[i + 1, j]

            # Anti-diagonal direction: 135 degrees
            else:

                neighbor1 = magnitude[i - 1, j - 1]
                neighbor2 = magnitude[i + 1, j + 1]

            # Keep pixel only if it is a local maximum
            if (
                magnitude[i, j] >= neighbor1 and
                magnitude[i, j] >= neighbor2
            ):
                output[i, j] = magnitude[i, j]

    return output


def hysteresis_thresholding(img, low_thresh, high_thresh):
    strong = 255
    weak = 50

    # Output image
    output = np.zeros(img.shape, dtype=np.uint8)

    # Strong edges
    strong_mask = img >= high_thresh

    # Weak edges
    weak_mask = (
        (img >= low_thresh) &
        (img < high_thresh)
    )

    output[strong_mask] = strong
    output[weak_mask] = weak

    height, width = img.shape

    # Examine weak pixels
    for i in range(1, height - 1):

        for j in range(1, width - 1):

            if output[i, j] == weak:

                # Look at the 3x3 neighborhood
                neighborhood = output[
                    i - 1:i + 2,
                    j - 1:j + 2
                ]

                # Promote weak edge if connected to strong edge
                if np.any(neighborhood == strong):
                    output[i, j] = strong

                else:
                    output[i, j] = 0

    return output

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    image = Image.open(image_path).convert("L")

    img = np.array(image, dtype=np.float32)

    # ----------------------------------------------
    # Step 2: Gaussian smoothing
    # ----------------------------------------------

    smoothed = gaussian_filter(
        img,
        sigma=1.4
    )

    # ----------------------------------------------
    # Step 3: Sobel kernels
    # ----------------------------------------------

    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)

    # ----------------------------------------------
    # Step 4: Calculate gradients
    # ----------------------------------------------

    Gx = convolve(
        smoothed,
        Kx
    )

    Gy = convolve(
        smoothed,
        Ky
    )

    # ----------------------------------------------
    # Step 5: Gradient magnitude
    # ----------------------------------------------

    magnitude = np.hypot(Gx, Gy)

    # Rescale to [0, 255]
    if magnitude.max() != 0:
        magnitude = (
            magnitude / magnitude.max() * 255
        )

    # ----------------------------------------------
    # Step 6: Gradient direction
    # ----------------------------------------------

    angle = np.arctan2(Gy, Gx)

    # ----------------------------------------------
    # Step 7: Non-Maximum Suppression
    # ----------------------------------------------

    thinned = non_maximum_suppression(
        magnitude,
        angle
    )

    # ----------------------------------------------
    # Step 8: Hysteresis Thresholding
    # ----------------------------------------------

    edges = hysteresis_thresholding(
        thinned,
        low,
        high
    )

    # ----------------------------------------------
    # Step 9: Save result
    # ----------------------------------------------

    output_image = Image.fromarray(edges)

    output_image.save(output_path)

    print(f"Edge detection result saved to {output_path}")

    return edges
 

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")