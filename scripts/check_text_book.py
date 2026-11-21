#!/usr/bin/env python3
"""Read-only OCR check on a saved robot-camera image, after sourcing the install."""
import argparse
import sys

import cv2

from solution_4to1mux.text_detection import read_text_label_boxes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', help='A robot-camera image or a crop of the chosen spine')
    parser.add_argument('--text', required=True, help='Exact text printed on the spine')
    parser.add_argument('--roi', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'),
                        help='Optional target-column/spine crop in image pixels')
    args = parser.parse_args(argv)
    image = cv2.imread(args.image)
    if image is None:
        parser.error('Could not read the image')
    if args.roi:
        x, y, w, h = args.roi
        if x < 0 or y < 0 or w < 12 or h < 12 or x+w > image.shape[1] or y+h > image.shape[0]:
            parser.error('ROI must lie inside the image and be at least 12 pixels wide/high')
        image = image[y:y+h, x:x+w]
    result = read_text_label_boxes(image, args.text)
    print('OCR readings:', result.readings)
    if not result.matches:
        print('NO EXACT MATCH. Check the crop, glare, letter size, spelling and orientation.')
        return 1
    for box, confidence, recognized in result.matches:
        print(f'EXACT MATCH: {recognized!r}; OCR score={confidence:.1f}; crop box={box}')
    print('This checks one image only; the OCR score is not a measured success probability.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
