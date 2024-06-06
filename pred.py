import difflib
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from skimage.filters import threshold_local
from train import model, device, label_encoder
import nltk
from predict import main

nltk.download("words")
from nltk.corpus import words


def noise_reduction(image):
    k = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, k, iterations=1)
    cv2.imshow("Dilation noise Image", image)
    cv2.waitKey(0)
    k = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, k, iterations=1)
    cv2.imshow("Erode noise Image", image)
    cv2.waitKey(0)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k)
    cv2.imshow("Morph noise Image", image)
    cv2.waitKey(0)
    image = cv2.medianBlur(image, 1)
    cv2.imshow("Blur noise Image", image)
    cv2.waitKey(0)
    return image


def resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def remove_borders(image):
    contours, heiarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y : y + h, x : x + w]
    return crop


def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    # Convert to grayscale
    resize_ratio = 500 / image.shape[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)
    try:
        im_copy = gray.copy()
        im_copy = resize(im_copy, resize_ratio)
        blurred = cv2.GaussianBlur(im_copy, (5, 5), 0)

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(blurred, rectKernel)

        edged = cv2.Canny(dilated, 100, 200, apertureSize=3)

        # Detect edges using Canny
        cv2.imshow("Edged Image", edged)
        cv2.waitKey(0)

        contours, hierarchy = cv2.findContours(
            edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image_with_contours = cv2.drawContours(
            im_copy.copy(), contours, -1, (0, 255, 0), 3
        )

        # Find contours
        cv2.imshow("cont Image", image_with_contours)
        cv2.waitKey(0)
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        image_with_largest_contours = cv2.drawContours(
            im_copy.copy(), largest_contours, -1, (0, 255, 0), 3
        )
        cv2.imshow("largest cont Image", image_with_largest_contours)
        cv2.waitKey(0)

        def approximate_contour(contour):
            peri = cv2.arcLength(contour, True)
            return cv2.approxPolyDP(contour, 0.032 * peri, True)

        # contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Select the largest contour that is roughly rectangular
        def get_contour(contours):
            # loop over the contours
            for c in contours:
                approx = approximate_contour(c)
                # if our approximated contour has four points, we can assume it is receipt's rectangle
                if len(approx) == 4:
                    return approx

        contours = get_contour(largest_contours)
        image_with_contour = cv2.drawContours(
            im_copy.copy(), [contours], -1, (0, 255, 0), 2
        )
        cv2.imshow("cont 2 Image", image_with_contour)
        cv2.waitKey(0)

        def contour_to_rect(contour):
            pts = contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            # top-left point has the smallest sum
            # bottom-right has the largest sum
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            # compute the difference between the points:
            # the top-right will have the minumum difference
            # the bottom-left will have the maximum difference
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect / resize_ratio

        def wrap_perspective(img, rect):
            # unpack rectangle points: top left, top right, bottom right, bottom left
            (tl, tr, br, bl) = rect
            # compute the width of the new image
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            # compute the height of the new image
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            # take the maximum of the width and height values to reach
            # our final dimensions
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))
            # destination points which will be used to map the screen to a "scanned" view
            dst = np.array(
                [
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1],
                ],
                dtype="float32",
            )
            # calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(rect, dst)
            # warp the perspective to grab the screen
            return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        scanned = wrap_perspective(image.copy(), contour_to_rect(contours))

        def bw_scanner(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            T = threshold_local(gray, 51, offset=5, method="gaussian")
            return (gray > T).astype("uint8") * 255

        result = bw_scanner(scanned)
        result = remove_borders(result)
        cv2.imshow("Final Image", image_with_contour)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    except Exception as e:
        result = gray
        result = remove_borders(result)
        print(e)

    blurred = cv2.GaussianBlur(result, (3, 3), 0)
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)  # Wait for a key press to close the window

    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    cv2.imshow("Binary Image", binary)
    cv2.waitKey(0)  # Wait for a key press to close the window
    binary = noise_reduction(binary)
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)
    cv2.imshow("Blurred2 Image", blurred)
    cv2.waitKey(0)
    # Remove guide lines (assuming they are horizontal)
    kernel = np.ones((1, 2), np.uint8)
    guide_lines_removed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Guide Lines Removed", guide_lines_removed)
    cv2.waitKey(0)  # Wait for a key press to close the window

    cv2.destroyAllWindows()

    return result  # Close all OpenCV windows


def segment_lines(image):

    img = image.copy()
    ret, thresh2 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Threshold Image", thresh2)
    cv2.waitKey(0)

    # Morphological operation to connect components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 2))
    mask = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel)
    cv2.imshow("Dilated Image", mask)
    cv2.waitKey(0)

    line_images = []
    bboxes_img = img.copy()

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if h < 7:
            continue

        # Create a mask for the line and count white pixels
        line_mask = mask[y : y + h, x : x + w]
        white_pixel_count = cv2.countNonZero(line_mask)
        print(white_pixel_count)
        # Filter out contours with too few white pixels
        if white_pixel_count < 40:
            continue
        # Add padding of 5 pixels
        x_padded = max(x - 5, 0)
        y_padded = max(y - 5, 0)
        w_padded = min(w + 10, img.shape[1] - x_padded)
        h_padded = min(h + 10, img.shape[0] - y_padded)
        line_img = img[y_padded : y_padded + h_padded, x_padded : x_padded + w_padded]
        line_images.append(line_img)
        cv2.rectangle(bboxes_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imshow(f"Lines {x}-{y}", line_img)

    cv2.imshow("Lines with Bounding Boxes", bboxes_img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # line_images.pop()
    # line_images.pop()
    return line_images


def segment_words(line_image):
    vertical_projection = np.sum(line_image, axis=0)
    words = []
    start = None
    for i, val in enumerate(vertical_projection):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            words.append((start, i))
            start = None
    if start is not None:
        words.append((start, len(vertical_projection)))

    word_images = [line_image[:, start:end] for start, end in words]
    for i, word_img in enumerate(word_images):
        cv2.imshow(f"Word {i+1}", word_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    return word_images


def pad_image(img, size=32):
    h, w = img.shape
    # Calculate padding sizes
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    # Pad the image with zeros (black) to the desired size
    padded_img = np.zeros((size, size), dtype=np.uint8)
    padded_img[pad_h : pad_h + h, pad_w : pad_w + w] = img
    return padded_img


def thin(image):
    cv2.imshow("Thin", image)
    cv2.waitKey(0)
    image = cv2.medianBlur(image, 1)
    k = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, k, iterations=1)
    cv2.imshow("Thin after", image)
    cv2.waitKey(0)
    return image


def thick(image):
    cv2.imshow("Thick", image)
    cv2.waitKey(0)
    k = np.ones((3, 3), np.uint8)
    image = cv2.filter2D(
        image, -11, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    )
    image = cv2.dilate(image, k, iterations=1)
    cv2.imshow("Thick after", image)
    cv2.waitKey(0)
    return image


def segment_characters(image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    character_images = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        print(w, h)
        # if w < 25 and h < 25:
        print(w, h)
        char_img = image[y : y + h, x : x + w]

        # Apply thresholding and resizing
        thresh = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(char_img, (22, 22), interpolation=cv2.INTER_CUBIC)

        thresh = pad_image(thresh)
        thresh = noise_reduction(thresh)
        thresh = thin(thresh)
        thresh = thick(thresh)

        cv2.imshow("char", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        character_images.append(thresh)

    cv2.waitKey(0)  # Wait for a key press to close the window

    cv2.destroyAllWindows()
    return character_images


def predict_characters(model, character_images, device):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32), antialias=True),
            transforms.ToTensor(),
        ]
    )

    model.eval()
    predictions = []
    with torch.no_grad():
        for char_img in character_images:
            # cv2.imshow(f"Word", char_img)
            cv2.waitKey(0)
            # Convert to PIL Image and apply transforms
            pil_img = Image.fromarray(char_img)
            input_tensor = transform(pil_img)
            # Predict the character
            output = model(input_tensor.unsqueeze(0).to(device))
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())

    return predictions


# Load the trained model
model_path = "best_model-v2.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

# Preprocess the input image
preprocessed_image = preprocess_image("code.png")
cv2.imwrite("./preprocess.png", preprocessed_image)

pred_words = []
# Segment lines and words
correct_words = words.words()

line_images = segment_lines(preprocessed_image)
for line_img in line_images:
    word_images = segment_words(line_img)
    for word_img in word_images:
        # char_images = segment_characters(word_img)
        # for i in char_images:
        word = main(word_img)
        pred_words.append(word)
        # predictions = predict_characters(model, char_images, device)
        # decoded_predictions = label_encoder.inverse_transform(predictions)
    # pred_words.append("".join(decoded_predictions))

print("words: ", pred_words)

# corrected_words = []
# for i in pred_words:
#     if i.isdigit():
#         continue
#     closest_matches = difflib.get_close_matches(i, correct_words, n=1, cutoff=0.0)
#     corrected_words.append(closest_matches)

# print("corrected words: ", corrected_words)
