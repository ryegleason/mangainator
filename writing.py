import math

import cv2
import re

import numpy as np
from PIL import Image, ImageFont, ImageDraw

from contour_extraction import extract_panels

FONT_SIZE = 48
BUBBLE_BORDER = 4
WORD_SPLIT_LENGTH = 7
MIN_SPLIT_CHARS = 3

font = ImageFont.truetype("CourierPrime-Regular.ttf", FONT_SIZE)


def add_text(img, panels, text, text_ptr=0, debug_filename=""):
    im_p = Image.fromarray(img)

    if debug_filename:
        debug_img = img.copy()

    # Get a drawing context
    draw = ImageDraw.Draw(im_p)
    # Needs to be a letter with a descender
    letter_size = draw.textsize("y", font=font)

    for panel in panels:
        for bubble in panel.bubbles:
            line_structure = []
            letter_locations = []
            for letter_top_y in range(bubble.topY, bubble.bottomY, letter_size[1]):
                line_structure += [0]
                letter_locations += [[]]
                for letter_top_x in range(bubble.leftX, bubble.rightX, letter_size[0]):
                    # clockwise order
                    corners = [(letter_top_x, letter_top_y), (letter_top_x + letter_size[0], letter_top_y),
                               (letter_top_x + letter_size[0], letter_top_y + letter_size[1]),
                               (letter_top_x, letter_top_y + letter_size[1])]
                    if len(list(filter(lambda x: cv2.pointPolygonTest(bubble.contour, x, True) > BUBBLE_BORDER, corners))) == 4:
                        # print(input_text[input_text_ptr])
                        line_structure[-1] = line_structure[-1] + 1
                        letter_locations[-1].append(corners[0])
                        if debug_filename:
                            cv2.rectangle(debug_img, corners[0], corners[2], (0, 255, 0), 1)
                    elif debug_filename:
                        cv2.rectangle(debug_img, corners[0], corners[2], (0, 0, 255), 1)
            for line_num in range(len(line_structure)):
                total_letters = line_structure[line_num]
                letters_remaining = total_letters
                line_content = ""
                line_complete = False
                while letters_remaining > 0 and not line_complete:
                    word_end = text.find(" ", text_ptr)
                    if word_end == -1:
                        word_end = len(text)
                    word_length = word_end - text_ptr
                    if word_length < letters_remaining:
                        line_content += text[text_ptr:word_end] + " "
                        letters_remaining -= word_length + 1
                        text_ptr = word_end + 1
                    elif letters_remaining >= MIN_SPLIT_CHARS + 1 and word_length >= WORD_SPLIT_LENGTH and \
                            line_num + 1 < len(line_structure) and line_structure[line_num + 1] >= MIN_SPLIT_CHARS \
                            and letters_remaining + line_structure[line_num + 1] >= word_length + 1:
                        # Split text across 2 lines
                        letters_for_line = min(letters_remaining - 1, word_length - MIN_SPLIT_CHARS)
                        line_content += text[text_ptr:(text_ptr + letters_for_line)] + "-"
                        text_ptr += letters_for_line
                        line_complete = True
                    else:
                        line_complete = True

                line_content = line_content.strip()
                if line_content:
                    # Makeshift text centering
                    front_padding = math.ceil((total_letters - len(line_content)) / 2)
                    # print("Front padding: " + str(front_padding) + ", total letters: " + str(total_letters))
                    draw.text(letter_locations[line_num][front_padding], line_content, (0, 0, 0), font=font)

    if debug_filename:
        cv2.imwrite(debug_filename, debug_img)

    return np.array(im_p), text_ptr


if __name__ == "__main__":
    with open("input.txt", "r") as f:
        input_text = f.read()

    # Replace newlines with spaces
    input_text = re.sub(r"\n+", " ", input_text).strip()
    input_text_ptr = 0

    img_azumanga = cv2.imread(r'azumanga.png')
    azumanga_panels = extract_panels(img_azumanga)
    azumanga_panels.sort()

    for panel in azumanga_panels:
        panel.blank_bubbles(img_azumanga)

    result_img, input_text_ptr = add_text(img_azumanga, azumanga_panels, input_text, input_text_ptr, "debug.png")

    cv2.imwrite('result.png', result_img)
