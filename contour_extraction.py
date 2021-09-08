import cv2
import numpy as np
import pytesseract
from functools import cmp_to_key

def find_contours_with_points(points, contours, top_contour_indicies, children, point_shift=(0, 0)):
    to_ret = []
    search_list = top_contour_indicies
    found_subcontour = True
    while search_list and found_subcontour:
        found_subcontour = False
        for idx in search_list:
            contains = True
            for point in points:
                shifted_point = (point[0] + point_shift[0], point[1] + point_shift[1])
                if cv2.pointPolygonTest(contours[idx], shifted_point, False) < 0:
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

def get_speech_contours(img, contours, top_contours, contour_children, point_shift=(0, 0), max_letter_frac=0.1,
                        contour_average_cutoff=240, min_convexity=0.9, debug_image_name=""):
    offset = (-point_shift[0], -point_shift[1])
    height = img.shape[0]
    width = img.shape[1]
    img_cv = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if debug_image_name:
        img_display = img.copy()

    containing_contours = set()
    good_contours = set()

    boxes = pytesseract.image_to_boxes(img_rgb, lang="eng", config="--psm 11")
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

                # White out letter for speech bubble check
                img_cv = cv2.rectangle(img_cv, corners[0], corners[1], (255, 255, 255), -1)

                # find containing contour
                found_contours = find_contours_with_points(corners, contours, top_contours, contour_children,
                                                           point_shift)
                if found_contours:
                    containing_contours.add(found_contours[-1])

            elif debug_image_name:
                img_display = cv2.rectangle(img_display, corners[0], corners[1], (0, 0, 255), 2)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    for contour_idx in containing_contours:
        contour = contours[contour_idx]
        if cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour)) > min_convexity:
            # Mask to just the contour and check the average color
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1, offset=offset)
            mean = cv2.mean(gray, mask=mask)

            # If the contour is a speech bubble, it'll be almost all white
            if mean[0] > contour_average_cutoff:
                if debug_image_name:
                    cv2.drawContours(img_display, [contour], 0, (0, 255, 0), 3, offset=offset)
                good_contours.add(contour_idx)
            elif debug_image_name:
                cv2.drawContours(img_display, [contour], 0, (255, 0, 255), 3, offset=offset)
        elif debug_image_name:
            cv2.drawContours(img_display, [contour], 0, (0, 255, 255), 3, offset=offset)

    if debug_image_name:
        print("Added " + str(len(good_contours)) + " new contours")
        cv2.imwrite(debug_image_name + ".png", img_display)

    # if len(new_good_contours) == 0:
    #     break
    # else:
    good_contours.update(good_contours)

    return good_contours


def extract_panels(img):
    contours, top_contours, contour_children = get_contours(img)
    to_ret = []
    for i in range(len(top_contours)):
        panel_x, panel_y, panel_w, panel_h = cv2.boundingRect(contours[top_contours[i]])
        panel_img = img[panel_y:(panel_y + panel_h), panel_x:(panel_x + panel_w)]
        # cv2.imwrite("panel_" + str(i) + "_pre.png", panel_img)
        # print(str(len(contour_children_azu[top_contours_azu[i]])) + " contours in panel")
        panel_contours = get_speech_contours(panel_img, contours, contour_children[top_contours[i]],
                                             contour_children, (panel_x, panel_y))
        to_ret.append(Panel(contours[top_contours[i]], list(map(lambda x: contours[x], panel_contours))))
    return to_ret


class Bubble:

    def __init__(self, contour):
        self.contour = contour
        x, y, w, h = cv2.boundingRect(contour)
        self.leftX = x
        self.rightX = x + w
        self.topY = y
        self.bottomY = y + h

    def __lt__(self, other):
        # If the right edge of the other bubble is to the left of the left edge of this bubble
        # i.e. this bubble is completely to the right of the other
        if other.rightX < self.leftX:
            return True
        # opposite situation
        if self.rightX < other.leftX:
            return False
        # So the bubbles must lie in the same "column." Just go off of top Y
        return self.topY < other.topY


class Panel:

    def __init__(self, border, bubble_contours, ordering="4koma"):
        self.border = border
        x, y, w, h = cv2.boundingRect(border)
        self.leftX = x
        self.rightX = x + w
        self.topY = y
        self.bottomY = y + h
        self.bubbles = list(map(lambda x: Bubble(x), bubble_contours))
        self.bubbles.sort()
        self.ordering = ordering

    def blank_bubbles(self, img):
        for bubble in self.bubbles:
            cv2.drawContours(img, [bubble.contour], 0, (255, 255, 255), -1)

    def __lt__(self, other):
        if self.ordering == "4koma":
            # If the right edge of the other panel is to the left of the left edge of this panel
            # i.e. this panel is completely to the right of the other
            if other.rightX < self.leftX:
                return True
            # opposite situation
            if self.rightX < other.leftX:
                return False
            # So the panels must lie in the same column. Just go off of top Y
            return self.topY < other.topY


if __name__ == "__main__":
    img_azumanga = cv2.imread(r'azumanga.png')
    azumanga_panels = extract_panels(img_azumanga)

    for panel in azumanga_panels:
        panel.blank_bubbles(img_azumanga)

    cv2.imwrite("out.png", img_azumanga)
    # img_bloom = cv2.imread(r"bloom.png")
    # img_bloom = cv2.blur(img_bloom, (5, 5))
    # border_thickness = 1
    # bloom_shape = img_bloom.shape
    # bordered_img = np.zeros((bloom_shape[0] + border_thickness * 2, bloom_shape[1] + border_thickness * 2, bloom_shape[2]), np.uint8)
    # # bordered_img = cv2.bitwise_not(bordered_img)
    # bordered_img[border_thickness:border_thickness+bloom_shape[0], border_thickness:border_thickness+bloom_shape[1]] = img_bloom
    # contours, top_contours, contour_children = get_contours(bordered_img)
    # for contour in top_contours:
    #     cv2.drawContours(bordered_img, contours, contour, (255, 0, 0), 3)
    # cv2.imwrite("contours.png", bordered_img)



