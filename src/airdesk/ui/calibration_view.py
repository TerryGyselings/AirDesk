"""Affichage plein écran de l'étalonnage guidé (`airdesk --calibrate`).

Le canevas est dessiné à une résolution réduite (moitié de l'écran) puis étiré en plein
écran par OpenCV : c'est plus léger à 30 images par seconde. Les positions sont données en
fractions de l'écran (0 à 1). Police OpenCV sans accents : les textes affichés n'en ont pas.
"""

from __future__ import annotations

import unicodedata

import cv2
import numpy as np

BACKGROUND = (30, 30, 30)
TEXT = (235, 235, 235)
CURRENT = (0, 220, 255)  # cible en cours (jaune)
DONE = (80, 200, 80)  # cible mesurée (vert)
WAITING = (110, 110, 110)  # cible à venir (gris)
CURSOR = (255, 170, 60)  # point qui suit la main (validation)
WARNING = (0, 140, 255)
ERROR = (60, 60, 255)


def make_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND
    return canvas


def to_pixel(canvas: np.ndarray, u: float, v: float) -> tuple[int, int]:
    h, w = canvas.shape[:2]
    return int(u * (w - 1)), int(v * (h - 1))


def ascii_text(text: str) -> str:
    """Enlève les accents (é -> e) : la police d'OpenCV ne sait pas les dessiner."""
    decomposed = unicodedata.normalize("NFKD", text)
    letters = []
    for char in decomposed:
        if ord(char) < 128:
            letters.append(char)
    return "".join(letters)


def wrap(text: str, max_chars: int) -> list[str]:
    """Coupe un texte en lignes d'au plus `max_chars` caractères, entre les mots."""
    lines = []
    current = ""
    for word in ascii_text(text).split():
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = word if not current else current + " " + word
    if current:
        lines.append(current)
    return lines


def draw_text_block(canvas: np.ndarray, lines: list[str], top: int, scale: float = 0.8,
                    color: tuple[int, int, int] = TEXT) -> None:
    """Lignes centrées horizontalement, à partir de la hauteur `top`."""
    w = canvas.shape[1]
    y = top
    for line in lines:
        line = ascii_text(line)
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(canvas, line, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += th + 14


def draw_target(canvas: np.ndarray, u: float, v: float, color: tuple[int, int, int],
                progress: float = 0.0) -> None:
    """Cible (croix + cercle) et, si progress > 0, anneau de progression du maintien."""
    x, y = to_pixel(canvas, u, v)
    radius = max(12, canvas.shape[0] // 40)
    cv2.circle(canvas, (x, y), radius, color, 2)
    cv2.line(canvas, (x - radius, y), (x + radius, y), color, 1)
    cv2.line(canvas, (x, y - radius), (x, y + radius), color, 1)
    if progress > 0:
        cv2.ellipse(canvas, (x, y), (radius * 2, radius * 2), -90, 0, 360 * progress, color, 4)


def paste_camera(canvas: np.ndarray, camera_image: np.ndarray | None) -> None:
    """Incruste la vue caméra (avec le squelette déjà dessiné) en bas à droite, 1/4 de largeur."""
    if camera_image is None:
        return
    h, w = canvas.shape[:2]
    inset_w = w // 4
    inset_h = int(camera_image.shape[0] * inset_w / camera_image.shape[1])
    small = cv2.resize(camera_image, (inset_w, inset_h))
    y0 = h - inset_h - 10
    x0 = w - inset_w - 10
    canvas[y0:y0 + inset_h, x0:x0 + inset_w] = small
    cv2.rectangle(canvas, (x0, y0), (x0 + inset_w, y0 + inset_h), TEXT, 1)


def draw_capture(
    canvas: np.ndarray,
    targets: list[tuple[str, float, float]],
    current: int,
    progress: float,
    hand_visible: bool,
    camera_image: np.ndarray | None,
    waiting_for_movement: bool = False,
) -> None:
    """Étape de capture : `targets` = (nom, u, v) dans l'ordre ; `current` = indice en cours."""
    for i in range(len(targets)):
        _, u, v = targets[i]
        if i < current:
            draw_target(canvas, u, v, DONE)
        elif i == current:
            draw_target(canvas, u, v, CURRENT, progress)
        else:
            draw_target(canvas, u, v, WAITING)
    h = canvas.shape[0]
    name = targets[current][0].replace("_", " ")
    lines = [f"Cible {current + 1}/{len(targets)} : {name}"]
    lines += wrap("Place ta main la ou tu voudrais etre pour viser ce point, dans une position "
                  "confortable, et garde-la immobile.", 60)
    draw_text_block(canvas, lines, int(h * 0.30))
    if not hand_visible:
        draw_text_block(canvas, ["Main pilote non visible"], int(h * 0.62), 0.7, WARNING)
    elif waiting_for_movement:
        draw_text_block(canvas, ["Mesure prise : va vers la cible suivante"], int(h * 0.62),
                        0.7, DONE)
    draw_text_block(canvas, ["Retour arriere = cible precedente    Echap = annuler"],
                    int(h * 0.70), 0.6, WAITING)
    paste_camera(canvas, camera_image)


def draw_validation(
    canvas: np.ndarray,
    corners: list[tuple[str, float, float]],
    reached: set[str],
    cursor: tuple[float, float] | None,
    warning: str | None,
    camera_image: np.ndarray | None,
) -> None:
    """Validation : un point suit la main avec la NOUVELLE zone ; coins atteints en vert."""
    for name, u, v in corners:
        draw_target(canvas, u, v, DONE if name in reached else CURRENT)
    h = canvas.shape[0]
    lines = ["Verification de la nouvelle zone"]
    lines += wrap("Le point suit ta main (la vraie souris ne bouge pas). Verifie que tu "
                  "atteins les 4 coins confortablement.", 60)
    draw_text_block(canvas, lines, int(h * 0.30))
    if warning:
        draw_text_block(canvas, wrap(warning, 60), int(h * 0.55), 0.7, WARNING)
    draw_text_block(canvas, ["Entree = enregistrer    R = recommencer    Echap = annuler"],
                    int(h * 0.70), 0.7, TEXT)
    if cursor is not None:
        x, y = to_pixel(canvas, cursor[0], cursor[1])
        cv2.circle(canvas, (x, y), max(8, h // 60), CURSOR, -1)
    paste_camera(canvas, camera_image)


def draw_refusal(canvas: np.ndarray, message: str, camera_image: np.ndarray | None) -> None:
    h = canvas.shape[0]
    draw_text_block(canvas, ["Zone refusee"], int(h * 0.30), 1.0, ERROR)
    draw_text_block(canvas, wrap(message, 60), int(h * 0.40), 0.7, TEXT)
    draw_text_block(canvas, ["R = recommencer    Echap = annuler"], int(h * 0.70), 0.7, TEXT)
    paste_camera(canvas, camera_image)
