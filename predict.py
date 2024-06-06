import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from train import model, device, label_encoder
import torch
import difflib
import nltk


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
    cv2.imshow("morph noise Image", image)
    cv2.waitKey(0)
    image = cv2.medianBlur(image, 1)
    cv2.imshow("blur noise Image", image)
    cv2.waitKey(0)
    return image


def preprocess_image(image):
    # Load the image
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)  # Wait for a key press to close the window

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
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

    cv2.destroyAllWindows()  # Close all OpenCV windows

    return guide_lines_removed


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
    cv2.imshow("thin", image)
    cv2.waitKey(0)
    image = cv2.medianBlur(image, 1)
    k = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, k, iterations=1)
    cv2.imshow("thin after", image)
    cv2.waitKey(0)
    # image = cv2.filter2D(
    #     image, -11, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # )
    return image


def thick(image):
    cv2.imshow("thick", image)
    cv2.waitKey(0)
    # image = cv2.medianBlur(image, 1)
    k = np.ones((3, 3), np.uint8)
    image = cv2.filter2D(
        image, -11, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    )
    image = cv2.dilate(image, k, iterations=1)
    cv2.imshow("thick after", image)
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
            # InvertImageTransform(),
        ]
    )

    model.eval()
    predictions = []
    with torch.no_grad():
        for char_img in character_images:
            # Convert to PIL Image and apply transforms
            pil_img = Image.fromarray(char_img)
            input_tensor = transform(pil_img)
            # Predict the character
            output = model(input_tensor.unsqueeze(0).to(device))
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())

    return predictions


def main(image):
    if type(image) == str:
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    nltk.download("words")
    from nltk.corpus import words

    # Load the trained model
    model_path = "best_model-v2.pth"
    # model = HandwritingOCR(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Preprocess the input image
    preprocessed_image = preprocess_image(image)
    # cv2.imwrite("./preprocess.png", preprocessed_image)

    # Segment the characters
    character_images = segment_characters(preprocessed_image)
    # for i in range(len(character_images)):
    #     cv2.imwrite(f"./segmented-{i}.png", character_images[i])

    # Predict the characters
    predicted_labels = predict_characters(model, character_images, device)

    # Decode the predicted labels to characters
    predicted_characters = [
        label_encoder.inverse_transform([label])[0] for label in predicted_labels
    ]

    predicted = "".join(predicted_characters).lower()
    print("Predicted characters:", predicted)

    correct_words = words.words()

    closest_matches = difflib.get_close_matches(
        predicted, correct_words, n=1, cutoff=0.0
    )

    # Print the corrected word if a match is found
    if closest_matches:
        correct_word = closest_matches[0]
        print(f"The corrected word is: {correct_word}")
    else:
        print("No close match found.")

    return correct_word


if __name__ == "__main__":
    main("TRAIN_00027.jpg")
