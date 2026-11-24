# AirDesk

Piloter son bureau **sans clavier ni souris**, avec la main devant une simple webcam.

Projet innovation-recherche — ESIR, Université de Rennes (2026-2027).

## État actuel (v0.1)

- Suivi de la main en temps réel (MediaPipe Hand Landmarker, 21 points)
- Lissage du curseur par filtre 1€
- Machine à états d'engagement : rien ne se passe tant que la main ouverte n'est pas montrée
- Pointage relatif (comme un pavé tactile), clic et glisser-déposer par pincement
- Fenêtre de debug : squelette de la main, état, images/s, temps de traitement

## Gestes

| Geste | Effet |
| --- | --- |
| Main ouverte face à la caméra pendant 0,3 s | Active le contrôle |
| Déplacer la main | Déplace le curseur |
| Pincement bref pouce-index (< `click_max_ms`, 0,4 s) | Clic gauche |
| Pincement maintenu puis déplacement | Glisser-déposer (ex. déplacer une fenêtre par sa barre de titre) |
| Main hors du champ plus de 1 s | Retour en veille |

Les seuils se règlent dans [`config/gestures.yaml`](config/gestures.yaml).

**Bien se placer** :

- **Distance de travail : 40 à 70 cm** de la caméra, la **main entière dans l'image**
  (poignet compris). Trop loin, la main devient petite et le suivi plus bruité ; trop près,
  les doigts sortent du cadre.
- **Geste de pincement : le « OK »**. Le bout du pouce touche le bout de l'index, **paume
  tournée vers l'écran**, les autres doigts **détendus** (ni serrés en poing, ni crispés).
  Pendant un glisser, garder pouce et index collés et bouger la main sans à-coups.
- **Non reconnu** : le pincement index pointé vers la caméra, autres doigts repliés. C'est
  une décision de conception : ce geste ressemble trop à un poing pour être distingué de
  façon fiable.

## Installation

Prérequis : **Python 3.10 à 3.12** (MediaPipe ne suit pas toujours la dernière version) et une webcam.

```bash
git clone https://github.com/TerryGyselings/AirDesk.git
cd AirDesk
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

Pour vérifier qu'on est bien dans le venv : `where python` (Windows) ou `which python`
(macOS / Linux). La première ligne doit pointer vers `...\.venv\Scripts\python.exe`
(ou `.venv/bin/python`). Si elle pointe vers le Python du système, réactiver le venv.

Au premier lancement, le modèle MediaPipe (environ 8 Mo) est téléchargé dans `models/`.

## Lancer

```bash
airdesk --dry-run     # conseillé au début : la vraie souris n'est pas touchée
airdesk               # pilote la vraie souris
airdesk --camera 1    # autre webcam
airdesk --width 640 --height 480   # résolution demandée (défaut : 1280x720)
airdesk --backend dshow            # forcer l'API vidéo : dshow, msmf, avfoundation, v4l2, any
airdesk --dry-run --log            # enregistre les mesures dans data/logs/ (voir « Mesures »)
```

`q` ou `Échap` dans la fenêtre de debug pour quitter.

Au démarrage, AirDesk affiche la caméra réellement utilisée, par exemple
`Caméra 1 : backend DSHOW, 640x480 YUY2, 29.4 images/s mesurées`, précédée des essais écartés.

## Réglages propres à la machine (`config/local.yaml`)

Le bon index de caméra, le backend et la résolution dépendent de chaque ordinateur. On les
enregistre une fois pour toutes dans `config/local.yaml`, **non versionné** (ignoré par git) :

```bash
python scripts/check_camera.py --save   # diagnostic + écrit la 1re combinaison qui marche
airdesk --dry-run                       # utilise directement cette caméra
```

Ou à la main : copier [`config/local.example.yaml`](config/local.example.yaml) en
`config/local.yaml` et l'adapter. Ordre de priorité, du plus fort au plus faible :

1. options de la ligne de commande (`--camera`, `--backend`, `--width`, `--height`) ;
2. `config/local.yaml` (peut aussi surcharger un seuil de geste pour cette machine) ;
3. `config/gestures.yaml` (commun à l'équipe) ;
4. valeurs par défaut du code.

Exemple mesuré (webcam C920 à l'index 1) : ouverture en 2,7 s avec `local.yaml`, contre 11,9 s
en essayant toutes les combinaisons.

## Mesures (pour régler les seuils)

```bash
airdesk --dry-run --log            # faire quelques clics et glisser, puis q
python scripts/analyse_log.py      # analyse le log le plus récent
```

`--log` écrit dans `data/logs/` (ignoré par git) deux fichiers par session :

- `AAAA-MM-JJ_HH-MM-SS.csv` : une ligne par image : horodatage, main présente, `pinch_ratio`,
  doigts tendus, extension de chaque doigt en 2D et 3D, doigts repliés, état, événements émis,
  temps de traitement. **Aucune image** (RGPD) ;
- `AAAA-MM-JJ_HH-MM-SS.json` : les réglages en vigueur (config complète fusionnée, version,
  commit git et modifications non commitées, système, caméra réellement ouverte).

`analyse_log.py` affiche les images/s, le temps de traitement (moyenne et 95e percentile),
le nombre de clics et de glisser, puis la durée de chaque pincement avec un histogramme texte
et la position de `click_max_ms`. Les durées sont calculées à partir de `pinch_ratio` et des
seuils `enter`/`exit` lus dans le `.json` du log, et non à partir des états de la machine,
qui dépendent justement de `click_max_ms`.

## Jeux de données étiquetés (poing contre pincement)

Problème étudié : un poing fermé pose le pouce sur l'index replié, ce que `pinch_ratio` seul
prend pour un pincement (« Midas touch »). Pour choisir une règle à partir de mesures :

```bash
python scripts/record_gestures.py --participant P01     # ~2 min, guidé à l'écran
python scripts/analyse_gestures.py data/datasets/A.csv data/datasets/B.csv --test data/datasets/C.csv
```

- `record_gestures.py` enchaîne 8 séquences de 10 s (main ouverte, pincements répétés au
  métronome, pincement maintenu, poing, poing qui s'ouvre et se ferme, main relâchée, gestes
  parasites, pincements index vers la caméra), avec un compte à rebours de 3 s. Il écrit dans
  `data/datasets/` (ignoré par git) l'étiquette, les descripteurs et les 21 points normalisés
  de chaque image, plus un `.json` des réglages. **Aucune image** ; `--participant` est un code
  pseudonyme, jamais un nom.
- `analyse_gestures.py` donne, par étiquette, les distributions de `pinch_ratio` et des
  extensions de doigts, compare les règles A, B, C, D (et leurs versions 3D) avec une
  matrice de confusion, et propose des seuils avec leur marge. Il simule ensuite les
  **épisodes** de pincement (ce que ressent l'utilisateur) pour plusieurs valeurs de `exit` et
  de délai de relâchement : ruptures d'un glisser, faux clics, faux glisser, pincements
  répétés détectés, et recommande la meilleure configuration. Les seuils sont appris sur les
  fichiers d'entraînement et évalués sur le fichier `--test` ; sans `--test`, les résultats
  sont optimistes. Le pincement « index vers la caméra » est hors vocabulaire (information).
  La configuration est choisie sur l'entraînement seulement, le test ne sert qu'à vérifier.
  Options : `--sweep-csv data/analyses/balayage.csv` (toutes les configurations),
  `--ruptures-non-fiables` (si le pincement maintenu a pu être relâché pendant l'enregistrement).
- Mesures de pincement comparées : `pinch_ratio` (pouce → bout de l'index) et `pinch_seg`
  (pouce → dernière phalange de l'index, variante `pinch_seg_68` sur deux phalanges).
- Protocole v2 de `record_gestures.py` : consigne du pincement maintenu précisée (doigts
  collés, petits mouvements lents) et position / vitesse du poignet enregistrées.
- **Configuration en service** (`config/gestures.yaml`) : règle D, mesure `ratio`,
  enter 0,25 / exit 0,40, délai de relâchement 66 ms, sans sortie sur repli. `rule: A` et
  `release_grace_ms: 0` redonnent le comportement d'origine, pour comparer.
- Descripteur clé, l'**extension** d'un doigt : distance bout du doigt → centre de la paume,
  divisée par la taille de la main (tendu ≈ 1,5, replié ≈ 0,4). Calculée en 2D et en 3D :
  un index pointé vers la caméra paraît court en 2D (perspective).
- Les nouveaux paramètres (`pinch.rule`, `min_index_extension`, `fingers.folded_below`...) sont
  dans `config/gestures.yaml` mais **pas encore utilisés** par la machine à états.

## Dépannage de la caméra

**Diagnostic** : `python scripts/check_camera.py` teste les index 0 à 3, chaque backend du
système et deux résolutions, puis affiche pour chaque combinaison la résolution obtenue, le
format, les images/s mesurées, la luminosité moyenne et un verdict. Aucune image n'est enregistrée.

| Symptôme | Cause probable | Que faire |
| --- | --- | --- |
| Image noire | Mauvais index (carte d'acquisition, caméra infrarouge) ou backend MSMF | `check_camera.py`, puis `--camera N` |
| Image figée | Caméra virtuelle éteinte (OBS Virtual Camera...) | Choisir un autre index |
| Démarrage long (~10 s), puis 640x480 | Webcam trop lente en 1280x720 (format YUY2) et MSMF bloqué | `python scripts/check_camera.py --save` : démarrage en ~3 s |
| « Impossible d'obtenir une image » | Caméra prise par une autre appli, ou accès refusé | Fermer Teams/Zoom/Discord/navigateur ; voir autorisations ci-dessous |

Ce que fait AirDesk à l'ouverture : il essaie les backends du système dans l'ordre
(Windows : DirectShow puis Media Foundation ; macOS : AVFoundation ; Linux : V4L2), lit
10 images de « chauffe » et rejette l'essai si l'image est noire, figée, ou à moins de
20 images/s en 1280x720. Il passe alors au backend suivant, puis à 640x480. Chaque essai est
limité à 5 s, car certains pilotes (MSMF) bloquent indéfiniment.

Autorisations :
- **Windows** : *Paramètres > Confidentialité et sécurité > Caméra* : activer l'accès à la
  caméra **et** « Autoriser les applications de bureau à accéder à la caméra ».
- **macOS** : *Réglages Système > Confidentialité et sécurité > Caméra* : autoriser le terminal.
- **Linux** : l'utilisateur doit appartenir au groupe `video` (`ls -l /dev/video*`).

## Compatibilité

| Système | État | Remarque |
| --- | --- | --- |
| Windows 10/11 | ✅ | Aucun réglage |
| macOS | ✅ | Autoriser le terminal dans *Réglages Système > Confidentialité et sécurité > Accessibilité* (et *Caméra*) |
| Linux X11 | ✅ | — |
| Linux Wayland | ⚠️ | L'injection d'événements souris est bloquée : ouvrir une session « Xorg » à l'écran de connexion |

## Tests

```bash
pytest          # tests unitaires (pas besoin de webcam)
ruff check .    # style et erreurs courantes
```

## Architecture

```
src/airdesk/
├── capture/         1. webcam (OpenCV) : choix du backend, détection d'image noire, repli
├── perception/      2. squelette de la main (MediaPipe) + filtre 1€
├── interpretation/  3. descripteurs (pincement, doigts tendus) + machine à états
├── actuator/        4. actions sur le système (souris via pynput)
├── ui/              5. retour visuel (fenêtre de debug)
├── app.py           boucle principale qui relie les couches
├── config.py        réglages : gestures.yaml + local.yaml + ligne de commande
├── session_log.py   écriture des mesures (--log)
├── log_analysis.py  analyse des mesures (utilisé par scripts/analyse_log.py)
└── gesture_analysis.py  analyse des jeux de données étiquetés (règles de pincement)

scripts/
├── check_camera.py      diagnostic des webcams (--save : écrit config/local.yaml)
├── analyse_log.py       résumé et histogramme d'un log
├── record_gestures.py   enregistrement guidé de gestes étiquetés
└── analyse_gestures.py  comparaison des règles de pincement, seuils proposés
```

Chaque couche ne dépend que de celles du dessus : la machine à états, par exemple, ne connaît ni la caméra ni la souris, ce qui permet de la tester sans matériel.

## Feuille de route

- [x] v0.1 — pointage, clic, glisser
- [ ] Défilement (deux doigts) et changement de fenêtre (balayage)
- [ ] Gestion directe des fenêtres (PyWinCtl) : agrandir, réduire, fermer
- [ ] Overlay transparent de retour visuel (PyQt6)
- [ ] Banc de mesure : latence, faux positifs, tâche de Fitts ISO 9241-411

## Références principales

- Zhang et al. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. arXiv:2006.10214
- Casiez, Roussel & Vogel (2012). *1€ Filter*. CHI '12. doi:10.1145/2207676.2208639
- Jacob (1991). *What You Look At Is What You Get* (problème du « Midas touch »). ACM TOIS. doi:10.1145/123078.128728
- Hincapié-Ramos et al. (2014). *Consumed Endurance* (fatigue du bras). CHI '14. doi:10.1145/2556288.2557130
