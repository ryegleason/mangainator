from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

src = None
erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
dilation_title = "Dilation:"
erosion_title = "Erosion:"
title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'


## [main]
def main(image):
    global src
    src = cv.imread(image)
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)
    ret, src = cv.threshold(src, 50, 255, cv.THRESH_BINARY_INV)
    struct_element = cv.getStructuringElement(cv.MORPH_RECT, (2 * 1 + 1, 2 * 1 + 1),
                                               (1, 1))
    src = cv.erode(src, struct_element)
    src = cv.dilate(src, struct_element)
    src = cv.Canny(src, 100, 200)

    cv.namedWindow(title_erosion_window, cv.WINDOW_NORMAL)
    cv.createTrackbar(title_trackbar_element_shape, title_erosion_window, 0, max_elem, do_both)
    cv.createTrackbar(dilation_title, title_erosion_window, 0, max_kernel_size, do_both)
    # cv.createTrackbar(title_trackbar_element_shape, title_erosion_window, 0, max_elem, erosion)
    cv.createTrackbar(erosion_title, title_erosion_window, 0, max_kernel_size, do_both)

    # cv.namedWindow(title_dilation_window, cv.WINDOW_NORMAL)

    do_both(0)
    cv.waitKey()
## [main]

# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE

def do_both(val):
    dilatation_size = cv.getTrackbarPos(dilation_title, title_erosion_window)
    erosion_size = cv.getTrackbarPos(erosion_title, title_erosion_window)
    shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))

    element = cv.getStructuringElement(shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    erosion_element = cv.getStructuringElement(shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    ## [kernel]
    dst = cv.dilate(src, element)
    dst = cv.erode(dst, erosion_element)
    cv.imshow(title_erosion_window, dst)


## [erosion]
def erosion(val):
    erosion_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
    erosion_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))

    ## [kernel]
    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    ## [kernel]
    erosion_dst = cv.erode(src, element)
    cv.imshow(title_erosion_window, erosion_dst)
## [erosion]


## [dilation]
def dilatation(val):
    dilatation_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
    dilation_shape = morph_shape(cv.getTrackbarPos(title_trackbar_element_shape, title_dilation_window))

    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv.dilate(src, element)
    cv.imshow(title_dilation_window, dilatation_dst)
## [dilation]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Code for Eroding and Dilating tutorial.')
    # parser.add_argument('--input', help='Path to input image.', default='LinuxLogo.jpg')
    # args = parser.parse_args()

    main("bloom.png")
