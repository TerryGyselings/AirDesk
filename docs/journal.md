# Journal de bord

Une entrée par séance ou par décision importante : date, ce qui a été fait, ce qui a été mesuré, ce qui bloque.
Ce journal sert de matière première au rapport final.

## 2026-11-24 — Base du projet (v0.1)

- Choix de la pile : Python + MediaPipe Hand Landmarker + OpenCV + pynput, un seul programme
  pour les 5 couches de l'architecture (voir rapport, section 4).
- Écarté pour l'instant : partir de gesture-synth (TypeScript, navigateur) — un navigateur ne peut
  pas piloter la souris ni les fenêtres du système, et le dépôt n'a pas de licence.
- Implémenté : suivi de la main, filtre 1€, machine à états (veille / actif / pincement / glisser),
  pointage relatif, clic et glisser par pincement, fenêtre de debug.
- 16 tests unitaires (filtre, descripteurs, machine à états), sans webcam.
- À faire : premier essai réel sur Windows et macOS, mesurer images/s et latence, régler les seuils.

## 2026-11-24 — Ouverture fiable de la caméra (Windows)

- Symptôme : fenêtre de debug noire sous Windows 11, aucune main détectée.
- Diagnostic (`scripts/check_camera.py`, 4 index × DSHOW/MSMF × 1280x720/640x480) :
  - index 0 = carte d'acquisition Elgato HD60 S+ (1920x1080 à 60 i/s imposés, image noire
    sans source HDMI) : c'est elle qu'ouvrait AirDesk par défaut, d'où l'écran noir ;
  - index 1 = webcam Logitech C920 : DSHOW reste en YUY2 (MJPG ignoré, quel que soit l'ordre
    de réglage) → 10 i/s en 1280x720, 30 i/s en 640x480 ; MSMF bloque indéfiniment ;
  - index 2 = OBS Virtual Camera éteinte : image fixe (écart entre images = 0,000).
- Décisions :
  - backend choisi selon l'OS, DirectShow en premier sous Windows ; MJPG demandé avec DSHOW ;
  - chaque essai dans un thread avec délai de 5 s (un appel bloqué dans le pilote ne peut pas
    être interrompu depuis Python) ; vérifié : DSHOW fonctionne encore après un blocage MSMF ;
  - essai rejeté si image noire (luminosité < 5/255), figée (écart moyen < 0,5 ; bruit de
    capteur mesuré ≈ 2,4 sur la C920) ou < 20 i/s en 1280x720 ;
  - ordre de repli : backend suivant, puis 640x480 ;
  - en cas d'échec total, recherche des autres index pour suggérer celui qui marche.
- Mesure : avec la C920, ouverture automatique en ~12 s (720p trop lent ~4 s + MSMF bloqué 5 s
  + 640x480 ~3 s) ; ~3 s avec `--width 640 --height 480 --backend dshow`.
- À faire : vérifier si 640x480 suffit à la précision du suivi ; essayer MJPG avec un autre
  pilote (Logitech) ou un autre backend pour retrouver le 720p à 30 i/s.
