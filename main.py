import cv2
import pytesseract
from PIL import Image
import numpy as np


def find_contours_with_points(points, contours, top_contour_indicies, children):
    to_ret = []
    search_list = top_contour_indicies
    found_subcontour = True
    while search_list and found_subcontour:
        found_subcontour = False
        for idx in search_list:
            contains = True
            for point in points:
                if cv2.pointPolygonTest(contours[idx], point, False) < 0:
                    contains = False
                    break
            if contains:
                to_ret.append(idx)
                search_list = children[idx]
                found_subcontour = True
                break
    return to_ret


def get_contours(img, thresh=200, max_contour_area_frac=0.9, min_contour_area_frac=0.0004, erode_size=-1):
    height = img.shape[0]
    width = img.shape[1]
    area = width * height

    # Threshhold image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshed = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    # Erode and dilate
    if erode_size > 0:
        erode_kernel = np.ones((erode_size, erode_size), np.uint8)
        threshed = cv2.erode(threshed, erode_kernel)
        threshed = cv2.dilate(threshed, erode_kernel)

    # Get all contours
    contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Organize hierarchy in useful way
    top_contour_candidates = []
    top_contours = []
    contour_children = [[] for _ in range(len(contours))]
    print(len(hierarchy[0]))
    for i in range(len(contours)):
        _, _, cont_w, cont_h = cv2.boundingRect(contours[i])
        if cont_w * cont_h > area * min_contour_area_frac:
            parent = hierarchy[0][i][3]
            if parent == -1:
                top_contour_candidates.append(i)
            else:
                contour_children[parent].append(i)

    # Get panels, can't be too big
    top_contours = get_top_contours(contours, top_contour_candidates, contour_children, area * max_contour_area_frac)

    return contours, top_contours, contour_children


def get_top_contours(contours, candidates, contour_children, max_contour_area):
    top_contours = []
    while candidates:
        contour_idx = candidates.pop(0)
        if cv2.contourArea(contours[contour_idx]) < max_contour_area:
            top_contours.append(contour_idx)
        else:
            candidates += contour_children[contour_idx]
    return top_contours


def get_speech_contours(img, contours, top_contours, contour_children, max_letter_frac=0.1, contour_average_cutoff=240, debug_image_name=""):
    img_cv = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    good_contours = set()
    count = 0

    while True:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        if debug_image_name:
            img_display = img_cv.copy()

        containing_contours = set()
        new_good_contours = set()

        boxes = pytesseract.image_to_boxes(img_rgb)
        boxes_list = boxes.split("\n")

        for box in boxes_list:
            box_elems = box.split(" ")
            if len(box_elems) >= 5:
                box_coords = list(map(int, box_elems[1:5]))
                corners = [(box_coords[0], height - box_coords[1]), (box_coords[2], height - box_coords[3])]

                # Check that letter box isn't too big
                if abs(box_coords[0] - box_coords[2]) / width < max_letter_frac and \
                        abs(box_coords[1] - box_coords[3]) / height < max_letter_frac:

                    if debug_image_name:
                        img_display = cv2.rectangle(img_display, corners[0], corners[1], (255, 0, 0), 2)

                    # White out letter so it's not detected in future rounds
                    img_cv = cv2.rectangle(img_cv, corners[0], corners[1], (255, 255, 255), -1)

                    # find containing contour
                    if find_contours_with_points(corners, contours, top_contours, contour_children):
                        containing_contours.add(find_contours_with_points(corners, contours, top_contours,
                                                                          contour_children)[-1])
                elif debug_image_name:
                    print(debug_image_name + " round " + str(count) + ": too large letter")
                    img_display = cv2.rectangle(img_display, corners[0], corners[1], (0, 0, 255), 2)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        for contour_idx in containing_contours:
            # Mask to just the contour and check the average color
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [contours[contour_idx]], -1, 255, -1)
            mean = cv2.mean(gray, mask=mask)

            # If the contour is a speech bubble, it'll be almost all white
            if mean[0] > contour_average_cutoff:
                if debug_image_name:
                    cv2.drawContours(img_display, contours, contour_idx, (0, 255, 0), 3)
                cv2.drawContours(img_cv, contours, contour_idx, (255, 255, 255), -1)
                new_good_contours.add(contour_idx)

        if debug_image_name:
            print("Added " + str(len(new_good_contours)) + " new contours")
            cv2.imwrite(debug_image_name + "_" + str(count) + ".png", img_display)

        if len(new_good_contours) == 0:
            break
        else:
            good_contours.update(new_good_contours)

        count = count + 1

    return good_contours, img_cv


img_azumanga = cv2.imread(r'azumanga.png')
contours_azu, top_contours_azu, contour_children_azu = get_contours(img_azumanga)

for i in range(len(top_contours_azu)):
    panel_x, panel_y, panel_w, panel_h = cv2.boundingRect(contours_azu[top_contours_azu[i]])
    panel_img = img_azumanga[panel_y:(panel_y+panel_h), panel_x:(panel_x+panel_w)]
    cv2.imwrite("panel_" + str(i) + "_pre.png", panel_img)
    panel_contours, _ = get_speech_contours(panel_img, contours_azu, contour_children_azu[top_contours_azu[i]],
                                            contour_children_azu, debug_image_name="panel_" + str(i) + "_debug")
    for bubble_idx in panel_contours:
        cv2.drawContours(panel_img, contours_azu, bubble_idx, (255, 255, 255), -1)
    cv2.imwrite("panel_" + str(i) + ".png", panel_img)

# good_contours, _ = get_speech_contours(img_azumanga, contours_azu, top_contours_azu, contour_children_azu)

# img_out = img_azumanga.copy()
# for i in good_contours:
#     cv2.drawContours(img_out, contours_azu, i, (255, 255, 255), -1)
#
# cv2.imwrite("img_out.png", img_out)
