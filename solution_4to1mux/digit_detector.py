"""
Shelf-column digit detection using only OpenCV + numpy - both already present
in the ERC container via cv_bridge, so this adds no new dependencies.

Why this works here when general OCR is hard: the search space is five known
digits (1-5), rendered large, dark-on-light, by a fixed simulator. That is
close to the easiest possible case for classical template matching.

Two-step workflow:
  1. CALIBRATE (once): run with save_candidates=True, which dumps every digit
     candidate it finds to candidates/. Look through them, and copy one clean
     example of each digit to templates/1.png ... templates/5.png.
  2. RUN: with real templates present, matching is fast and accurate.

If no templates exist, it falls back to synthetic ones rendered with
cv2.putText. Those work, but noticeably worse than real crops - the simulator's
font is not Hershey Simplex. Treat the fallback as "works out of the box",
not "good enough to submit".
"""

import os

import cv2
import numpy as np

TEMPLATE_SIZE = (32, 32)  # every candidate and template is normalized to this
VALID_DIGITS = [1, 2, 3, 4, 5]


class DigitDetector:
    def __init__(self, template_dir='templates', candidate_dir='candidates',
                 save_candidates=False, match_threshold=0.45,
                 min_area=60, max_area_fraction=0.05,
                 min_aspect=0.8, max_aspect=4.5):
        self.template_dir = template_dir
        self.candidate_dir = candidate_dir
        self.save_candidates = save_candidates
        self.match_threshold = match_threshold
        self.min_area = min_area
        self.max_area_fraction = max_area_fraction
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

        self._candidate_counter = 0
        if self.save_candidates:
            os.makedirs(self.candidate_dir, exist_ok=True)

        self.real_template_digits = set()
        self.templates = self._load_templates()
        self.using_synthetic = not self._has_real_templates()

    # ---------------- templates ----------------

    def _has_real_templates(self):
        return len(self.real_template_digits) == len(VALID_DIGITS)

    def _load_templates(self):
        templates = {}
        for digit in VALID_DIGITS:
            path = os.path.join(self.template_dir, f'{digit}.png')
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[digit] = self._normalize(img)
                    self.real_template_digits.add(digit)
        for digit in VALID_DIGITS:
            if digit not in templates:
                templates[digit] = self._synthetic_template(digit)
        return templates

    def _synthetic_template(self, digit):
        """Fallback only - renders the digit with a built-in OpenCV font."""
        canvas = np.zeros((64, 64), dtype=np.uint8)
        cv2.putText(canvas, str(digit), (14, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 3)
        return self._normalize(canvas)

    @staticmethod
    def _normalize(gray_img):
        """Binarize, crop to the ink's bounding box, then resize. Cropping
        first is what makes matching scale- and padding-invariant."""
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if binary.mean() > 127:          # ensure ink is white on black
            binary = cv2.bitwise_not(binary)
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            binary = binary[y:y + h, x:x + w]
        if binary.size == 0:
            return np.zeros(TEMPLATE_SIZE, dtype=np.uint8)
        return cv2.resize(binary, TEMPLATE_SIZE, interpolation=cv2.INTER_NEAREST)

    # ---------------- detection ----------------

    def detect_with_diagnostics(self, bgr_image):
        """Like detect(), but also returns every contour it considered and why
        it was rejected. Use this when nothing is being found and you need to
        know whether the digits are being missed at thresholding, filtered out
        by size/shape, or found but not matching any template.

        Returns (results, diagnostics, binary_image)."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        frame_area = gray.shape[0] * gray.shape[1]

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results, diagnostics = [], []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect = h / float(w) if w else 0.0
            entry = {'bbox': (x, y, w, h), 'area': area, 'aspect': aspect,
                     'digit': None, 'score': 0.0, 'rejected': None}

            if area < self.min_area:
                entry['rejected'] = f'area {area:.0f} < min {self.min_area}'
            elif area > frame_area * self.max_area_fraction:
                entry['rejected'] = f'area {area:.0f} > max {frame_area * self.max_area_fraction:.0f}'
            elif not (self.min_aspect <= aspect <= self.max_aspect):
                entry['rejected'] = f'aspect {aspect:.2f} outside [{self.min_aspect}, {self.max_aspect}]'
            else:
                roi = gray[y:y + h, x:x + w]
                if roi.size:
                    digit, score = self._classify(self._normalize(roi))
                    entry['digit'], entry['score'] = digit, score
                    if digit is None:
                        entry['rejected'] = f'best template score {score:.2f} < {self.match_threshold}'
                    else:
                        results.append((digit, score, (x, y, w, h)))
            diagnostics.append(entry)

        results.sort(key=lambda r: r[1], reverse=True)
        return results, diagnostics, binary

    def detect(self, bgr_image):
        """Returns a list of (digit, confidence, (x, y, w, h)) for every digit
        found, sorted by confidence descending."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        frame_area = gray.shape[0] * gray.shape[1]

        # Digits are dark on a light shelf/sign, so invert to make them blobs.
        # Adaptive threshold handles uneven lighting better than a global one,
        # which matters for Phase 2's "lighting variations" on real hardware.
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > frame_area * self.max_area_fraction:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0:
                continue
            aspect = h / float(w)
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue

            roi = gray[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            normalized = self._normalize(roi)

            if self.save_candidates:
                self._dump_candidate(normalized, roi)

            digit, confidence = self._classify(normalized)
            if digit is not None:
                results.append((digit, confidence, (x, y, w, h)))

        results.sort(key=lambda r: r[1], reverse=True)
        return results

    def _classify(self, normalized_roi):
        """Compare against every template; return the best match above
        threshold, or (None, best_score) if nothing is convincing."""
        best_digit, best_score = None, 0.0
        for digit, template in self.templates.items():
            result = cv2.matchTemplate(normalized_roi, template, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if score > best_score:
                best_digit, best_score = digit, score
        if best_score >= self.match_threshold:
            return best_digit, best_score
        return None, best_score

    def _dump_candidate(self, normalized, raw_roi):
        """Saves both the normalized 32x32 and the raw crop, so you can pick
        good template examples during calibration."""
        idx = self._candidate_counter
        self._candidate_counter += 1
        cv2.imwrite(os.path.join(self.candidate_dir, f'cand_{idx:04d}_norm.png'), normalized)
        cv2.imwrite(os.path.join(self.candidate_dir, f'cand_{idx:04d}_raw.png'), raw_roi)

    def find_digit(self, bgr_image, target_digit):
        """Convenience wrapper: returns (confidence, bbox) for the best match
        of one specific digit, or (None, None)."""
        for digit, confidence, bbox in self.detect(bgr_image):
            if digit == target_digit:
                return confidence, bbox
        return None, None


