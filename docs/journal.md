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

## 2026-11-24 — Dépôt git, réglages par machine, journal de mesures

- Dépôt git initialisé et poussé sur GitHub (un seul commit de départ : l'archive v0.1
  d'origine n'était pas disponible pour séparer la base du travail caméra).
- Réglages par machine dans `config/local.yaml` (non versionné), priorité :
  ligne de commande > local.yaml > gestures.yaml > défauts. `check_camera.py --save` l'écrit.
  Mesure : ouverture de la C920 en 2,7 s au lieu de 11,9 s.
- `airdesk --log` : un CSV par session (une ligne par image, aucune image enregistrée) et un
  JSON des réglages (config fusionnée, version, commit + modifications non commitées, système,
  caméra). Sans ce JSON, un log ne dit plus avec quels seuils il a été mesuré.
- Durée d'un pincement calculée depuis `pinch_ratio` et l'hystérésis enter/exit, et non depuis
  les états de la machine : ceux-ci dépendent de `click_max_ms`, le seuil qu'on veut régler.
  L'issue observée (clic / glisser) est gardée à côté pour comparer.
- Premier log réel (10 s, sans main) : 30,1 i/s, traitement 11,2 ms en moyenne, 11,8 ms au
  95e percentile.
- À faire : sessions de mesure avec plusieurs utilisateurs (clics volontaires puis glisser)
  pour fixer `click_max_ms` et `pinch.enter/exit`.
