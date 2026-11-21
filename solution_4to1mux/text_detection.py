"""Exact-title OCR on isolated dark print; no white-on-white colour segmentation."""
import csv
from dataclasses import dataclass
import io
import subprocess

import cv2
import numpy as np

WHITE_BOOK_TITLES = ('ROBOT', 'LIBRARY', 'AGENT', 'FUTURE')


def normalized_label(text):
    return ''.join(character for character in text.casefold() if character.isalnum())


def label_matches(candidate, target):
    target = normalized_label(target)
    return bool(target) and normalized_label(candidate) == target


def title_from_reading(text, confidence):
    value = normalized_label(text)
    exact = next((word for word in WHITE_BOOK_TITLES if normalized_label(word) == value), None)
    if exact is not None:
        return exact
    # A single inserted/missing/misread glyph is allowed only within this
    # closed four-title vocabulary, with one uniquely closest high-score word.
    def distance(a, b):
        previous = list(range(len(b)+1))
        for i, ca in enumerate(a, 1):
            current = [i]
            for j, cb in enumerate(b, 1):
                current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(ca != cb)))
            previous = current
        return previous[-1]
    ranked = sorted((distance(value, normalized_label(word)), word) for word in WHITE_BOOK_TITLES)
    if confidence >= 60 and ranked[0][0] <= 1 and ranked[1][0] > ranked[0][0]:
        return ranked[0][1]
    return None


@dataclass
class OCRResult:
    matches: list
    readings: list


def print_regions(image):
    """Group black letters in both directions; exclude solid dark objects."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dark = np.uint8(gray < 110)*255
    regions = []
    for kernel in ((3, 13), (13, 3)):
        joined = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones(kernel[::-1], np.uint8))
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            maximum_area = .70 if max(gray.shape) > 2*min(gray.shape) else .12
            if min(w, h) < 4 or max(w, h) < 18 or w*h > maximum_area*gray.size:
                continue
            density = np.count_nonzero(dark[y:y+h, x:x+w])/float(w*h)
            if not .06 <= density <= .70:
                continue
            pad = 3
            box = (max(0, x-pad), max(0, y-pad), min(gray.shape[1], x+w+pad), min(gray.shape[0], y+h+pad))
            if box not in regions:
                regions.append(box)
    # Bound work in a cluttered frame. The shelf crop normally has few labels.
    regions.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return gray, regions[:32]


def read_text_label_boxes(crop, target):
    if crop.size == 0:
        return OCRResult([], [])
    gray, regions = print_regions(crop)
    tiles, boxes = [], []
    for x0, y0, x1, y1 in regions:
        patch = gray[y0:y1, x0:x1]
        variants = ((cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    if patch.shape[0] > 1.3*patch.shape[1] else (None,))
        for direction in variants:
            view = cv2.rotate(patch, direction) if direction is not None else patch
            scale = min(360./view.shape[1], 44./view.shape[0])
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            _, view = cv2.threshold(view, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            tile = np.full((72, 400), 255, np.uint8)
            y, x = (72-view.shape[0])//2, (400-view.shape[1])//2
            tile[y:y+view.shape[0], x:x+view.shape[1]] = view
            tiles.append(tile)
            boxes.append((x0, y0, x1-x0, y1-y0))
    if not tiles:
        return OCRResult([], [])
    success, encoded = cv2.imencode('.png', np.vstack(tiles))
    if not success:
        return OCRResult([], [])
    try:
        result = subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '6', '-l', 'eng', 'tsv'],
            input=encoded.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=1.3)
    except subprocess.TimeoutExpired:
        return OCRResult([], [])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors='replace').strip() or 'Tesseract failed')
    grouped = {}
    for word in csv.DictReader(io.StringIO(result.stdout.decode(errors='replace')), delimiter='\t'):
        if word.get('level') != '5' or not word.get('text', '').strip():
            continue
        try:
            confidence = float(word['conf'])
            row = int((int(word['top'])+.5*int(word['height']))//72)
            left = int(word['left'])
        except (ValueError, KeyError):
            continue
        if confidence >= 30 and 0 <= row < len(boxes):
            grouped.setdefault(row, []).append((left, word['text'], confidence))
    matches, readings = [], []
    for row, words in grouped.items():
        words.sort()
        text, confidence = ' '.join(w[1] for w in words), float(np.mean([w[2] for w in words]))
        if text not in readings:
            readings.append(text)
        corrected = title_from_reading(text, confidence)
        matched = label_matches(text, target) or (corrected is not None and label_matches(corrected, target))
        if confidence >= 45 and matched:
            box = boxes[row]
            if not any(old[0] == box for old in matches):
                matches.append((box, confidence, corrected or text))
    if not matches:
        for row, words in list(grouped.items()):
            raw = ' '.join(word[1] for word in sorted(words))
            suggestion = title_from_reading(raw, 100.)
            if suggestion is None or not label_matches(suggestion, target):
                continue
            # A weak contact-sheet reading must be re-read as an isolated
            # line, not promoted solely because its spelling is close.
            x0, y0, w, h = boxes[row]
            patch = gray[y0:y0+h, x0:x0+w]
            if h > 1.3*w:
                patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
            patch = cv2.resize(patch, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
            patch = cv2.copyMakeBorder(patch, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
            ok, data = cv2.imencode('.png', patch)
            if not ok:
                continue
            try:
                reread = subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '7', '-l', 'eng', 'tsv'],
                    input=data.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=.5)
            except subprocess.TimeoutExpired:
                break
            valid = []
            for word in csv.DictReader(io.StringIO(reread.stdout.decode(errors='replace')), delimiter='\t'):
                if word.get('level') == '5' and word.get('text', '').strip():
                    try:
                        valid.append((word['text'], float(word['conf'])))
                    except (ValueError, KeyError):
                        pass
            text = ' '.join(word[0] for word in valid)
            score = float(np.mean([word[1] for word in valid])) if valid else 0.
            readings.append(text)
            if score >= 45 and label_matches(text, target):
                matches.append((boxes[row], score, text))
            break
    return OCRResult(matches, readings[:24])
