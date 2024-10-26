from __future__ import print_function
import sys
import cv2 as cv

## [global_variables]
use_mask = False
img = None
templ = None
mask = None
image_window = "Source Image"
result_window = "Result window"

match_method = 0
max_Trackbar = 5
## [global_variables]

def main(argv):

    if (len(sys.argv) < 3):
        print('Not enough parameters')
        print('Usage:\nmatch_template_demo.py <image_name> <template_name> [<mask_name>]')
        return -1

    ## [load_image]
    global img
    global templ
    img = ncvslideio.imread(sys.argv[1], ncvslideio.IMREAD_COLOR)
    templ = ncvslideio.imread(sys.argv[2], ncvslideio.IMREAD_COLOR)

    if (len(sys.argv) > 3):
        global use_mask
        use_mask = True
        global mask
        mask = ncvslideio.imread( sys.argv[3], ncvslideio.IMREAD_COLOR )

    if ((img is None) or (templ is None) or (use_mask and (mask is None))):
        print('Can\'t read one of the images')
        return -1
    ## [load_image]

    ## [create_windows]
    ncvslideio.namedWindow( image_window, ncvslideio.WINDOW_AUTOSIZE )
    ncvslideio.namedWindow( result_window, ncvslideio.WINDOW_AUTOSIZE )
    ## [create_windows]

    ## [create_trackbar]
    trackbar_label = 'Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED'
    ncvslideio.createTrackbar( trackbar_label, image_window, match_method, max_Trackbar, MatchingMethod )
    ## [create_trackbar]

    MatchingMethod(match_method)

    ## [wait_key]
    ncvslideio.waitKey(0)
    return 0
    ## [wait_key]

def MatchingMethod(param):

    global match_method
    match_method = param

    ## [copy_source]
    img_display = img.copy()
    ## [copy_source]
    ## [match_template]
    method_accepts_mask = (ncvslideio.TM_SQDIFF == match_method or match_method == ncvslideio.TM_CCORR_NORMED)
    if (use_mask and method_accepts_mask):
        result = ncvslideio.matchTemplate(img, templ, match_method, None, mask)
    else:
        result = ncvslideio.matchTemplate(img, templ, match_method)
    ## [match_template]

    ## [normalize]
    ncvslideio.normalize( result, result, 0, 1, ncvslideio.NORM_MINMAX, -1 )
    ## [normalize]
    ## [best_match]
    _minVal, _maxVal, minLoc, maxLoc = ncvslideio.minMaxLoc(result, None)
    ## [best_match]

    ## [match_loc]
    if (match_method == ncvslideio.TM_SQDIFF or match_method == ncvslideio.TM_SQDIFF_NORMED):
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    ## [match_loc]

    ## [imshow]
    ncvslideio.rectangle(img_display, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0,0,0), 2, 8, 0 )
    ncvslideio.rectangle(result, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0,0,0), 2, 8, 0 )
    ncvslideio.imshow(image_window, img_display)
    ncvslideio.imshow(result_window, result)
    ## [imshow]
    pass

if __name__ == "__main__":
    main(sys.argv[1:])
