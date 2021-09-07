import cv2
import pytesseract
from PIL import Image
import numpy as np

erode_size = 2
erode_kernel = np.ones((erode_size, erode_size), np.uint8)

MAX_LETTER_FRAC = 0.1
CONTOUR_AREA_FRAC = 0.9
CONTOUR_AVERAGE_CUTOFF = 240


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


# img_cv = cv2.imread(r'Azumanga_Daioh_p007.png')
img_cv = cv2.imread(r'bloom.png')
height = img_cv.shape[0]
width = img_cv.shape[1]

img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
img_out = img_cv.copy()

gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

ret, threshed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
# threshed = cv2.erode(threshed, erode_kernel)
# threshed = cv2.dilate(threshed, erode_kernel)
cv2.imwrite("thresh.png", threshed)
# edges = cv2.Canny(threshed, 100, 200)
# cv2.imwrite("edging.png", edges)
contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
top_contours = []
contour_children = [[] for i in range(len(contours))]
print(len(hierarchy[0]))
for i in range(len(contours)):
    parent = hierarchy[0][i][3]
    if parent == -1:
        top_contours.append(i)
    else:
        contour_children[parent].append(i)

# contoured = cv2.drawContours(img_cv, contours, -1, (0, 255, 0), 1)
# cv2.imwrite("contours.png", contoured)

# By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# we need to convert from BGR to RGB format/mode:

good_contours = set()

while True:
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_display = img_cv.copy()
    boxes = pytesseract.image_to_boxes(img_rgb)

    containing_contours = set()
    new_good_contours = set()

    boxes_list = boxes.split("\n")

    for box in boxes_list:
        box_elems = box.split(" ")
        if len(box_elems) >= 5:
            box_coords = list(map(int, box_elems[1:5]))
            corners = [(box_coords[0], height - box_coords[1]), (box_coords[2], height - box_coords[3])]
            if abs(box_coords[0] - box_coords[2]) / width < MAX_LETTER_FRAC and abs(box_coords[1] - box_coords[3]) / height < MAX_LETTER_FRAC:
                img_display = cv2.rectangle(img=img_display, pt1=corners[0], pt2=corners[1], color=(255, 0, 0), thickness=2)
                img_cv = cv2.rectangle(img=img_cv, pt1=corners[0], pt2=corners[1], color=(255, 255, 255), thickness=-1)
                if find_contours_with_points(corners, contours, top_contours, contour_children):
                    containing_contours.add(find_contours_with_points(corners, contours, top_contours, contour_children)[-1])

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    containing_contours = list(filter(lambda x: cv2.contourArea(contours[x]) < width * height * CONTOUR_AREA_FRAC, containing_contours))

    for contour_idx in containing_contours:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [contours[contour_idx]], -1, 255, -1)
        mean = cv2.mean(gray, mask=mask)
        if mean[0] > CONTOUR_AVERAGE_CUTOFF:
            cv2.drawContours(img_display, contours, contour_idx, (0, 255, 0), 3)
            cv2.drawContours(img_cv, contours, contour_idx, (255, 255, 255), -1)
            new_good_contours.add(contour_idx)

    cv2.namedWindow('letters', cv2.WINDOW_NORMAL)
    cv2.imshow("letters", img_display)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()

    if len(new_good_contours) == 0:
        break
    else:
        good_contours.update(new_good_contours)
        print("Added " + str(len(new_good_contours)) + " new contours")

for contour_idx in good_contours:
    cv2.drawContours(img_out, contours, contour_idx, (255, 255, 255), -1)

cv2.imwrite("img_out.png", img_out)
