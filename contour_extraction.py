import cv2
import numpy as np
import pytesseract
import glob
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


def get_contours(img, thresh=200, max_contour_area_frac=0.9, min_contour_area_frac=0.0004, erode_size=-1,
                 threshed_out_file=""):
    height = img.shape[0]
    width = img.shape[1]
    area = width * height

    # Threshhold image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshed = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    if threshed_out_file:
        cv2.imwrite(threshed_out_file, threshed)

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


def get_bg_contour(contours, candidates, min_hull_area):
    best_candidate = None
    best_candidate_hull_area = 0
    for candidate_idx in candidates:
        candidate_hull_area = cv2.contourArea(cv2.convexHull(contours[candidate_idx]))
        if candidate_hull_area > best_candidate_hull_area:
            best_candidate = candidate_idx
            best_candidate_hull_area = candidate_hull_area
    if best_candidate_hull_area > min_hull_area:
        return best_candidate
    else:
        return None


def get_panel_lines(img, thresh=50, erode_size=1, line_min_frac=0.25):
    height = img.shape[0]
    width = img.shape[1]
    smaller_dim = min(height, width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshed = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*erode_size + 1, 2*erode_size + 1), (erode_size, erode_size))
    threshed = cv2.erode(threshed, struct_element)
    threshed = cv2.dilate(threshed, struct_element)
    edges = cv2.Canny(threshed, 100, 200)
    edges = cv2.dilate(edges, struct_element)
    return cv2.HoughLines(edges & threshed, 1, np.pi / 180, int(smaller_dim * line_min_frac))


def draw_hough_lines(img, lines, thickness=3, color=(0, 0, 255)):
    diag = int(np.linalg.norm(img.shape[:2]))
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + diag * b)
            y1 = int(y0 - diag * a)
            x2 = int(x0 - diag * b)
            y2 = int(y0 + diag * a)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


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
    # img_azumanga = cv2.imread(r'azumanga.png')
    # azumanga_panels = extract_panels(img_azumanga)
    #
    # for panel in azumanga_panels:
    #     panel.blank_bubbles(img_azumanga)
    #
    # cv2.imwrite("out.png", img_azumanga)
    #
    # window_name = "Debug"
    #
    # img_bloom = cv2.imread("bloom.png")
    # height = img_bloom.shape[0]
    # width = img_bloom.shape[1]
    # diag = int(np.linalg.norm([height, width]))
    #
    # gray = cv2.cvtColor(img_bloom, cv2.COLOR_BGR2GRAY)
    # ret, threshed = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    # struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
    # threshed = cv2.erode(threshed, struct_element)
    # threshed = cv2.dilate(threshed, struct_element)
    # edges = cv2.Canny(threshed, 100, 200)
    # edges = cv2.dilate(edges, struct_element)
    # edges = edges & threshed
    #
    # # ret, threshed = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # # cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    #
    # def debug_callback(arg):
    #     draw_on = img_bloom.copy()
    #     lines = cv2.HoughLines(edges, 1, np.pi / 180, arg)
    #     if lines is not None:
    #         print(len(lines))
    #         for line in lines:
    #             print(line)
    #             rho, theta = line[0]
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a * rho
    #             y0 = b * rho
    #             x1 = int(x0 + diag * b)
    #             y1 = int(y0 - diag * a)
    #             x2 = int(x0 - diag * b)
    #             y2 = int(y0 + diag * a)
    #             cv2.line(draw_on, (x1, y1), (x2, y2), (0, 0, 255), 5)
    #     cv2.imshow(window_name, draw_on)
    #
    # cv2.createTrackbar("Thresh", window_name, 400, 2000, debug_callback)
    #
    # debug_callback(400)
    # c = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite("debug.png", threshed)

    for input_file in glob.glob("input/*.jpg"):
        img_bloom = cv2.imread(input_file)
        lines = get_panel_lines(img_bloom)
        draw_hough_lines(img_bloom, lines)
    #     height = img_bloom.shape[0]
    #     width = img_bloom.shape[1]
    #     area = width * height
    #
    #     mask = np.full((height, width), 0, dtype="uint8")
    #
    #     contours, top_contours, contour_children = get_contours(img_bloom, max_contour_area_frac=1)
    #     bg_contour = get_bg_contour(contours, top_contours, area * 0.5)
    #
    #     if bg_contour is not None:
    #         cv2.drawContours(mask, contours, bg_contour, 127, -1)
    #         for child in contour_children[bg_contour]:
    #             cv2.drawContours(mask, contours, child, 255, -1)
    #         img_bloom[mask > 0] = (0, 255, 0)
    #         img_bloom[mask > 128] = (255, 0, 0)
        cv2.imwrite("output/" + input_file.split("/")[-1], img_bloom)



