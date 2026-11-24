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

## 2026-11-24 — Poing pris pour un pincement : outillage de mesure

- Constat en réel : poing fermé = pouce posé sur l'index replié → `pinch_ratio` petit →
  clics et glisser involontaires (« Midas touch », Jacob 1991).
- Nouveau descripteur, l'extension d'un doigt : distance bout → centre de la paume (moyenne
  des points 0, 5, 9, 13, 17), divisée par la taille de la main. Insensible à la position, à
  la rotation et à la distance. Sur main synthétique : tendu ≈ 1,5, replié ≈ 0,4.
- Calculée en 2D et en 3D : un index pointé vers la caméra paraît court en 2D
  (raccourcissement de perspective). La 3D utilise z de MediaPipe, à l'échelle de x.
- Unités : MediaPipe divise x par la largeur et y par la hauteur de l'image. En 640x480, sans
  correction, une main tournée de 90° change de proportions (1,48 contre 1,54 sur la main de
  test). Les extensions sont donc calculées en unités homogènes (x et z multipliés par
  largeur/hauteur). Décision en attente : `pinch_ratio` et `extended_fingers` n'ont PAS cette
  correction, pour ne pas changer le comportement actuel ; à revoir avec la nouvelle règle.
- Règles candidates (`interpretation/pinch_rules.py`) : A = actuelle, B = A + index tendu,
  C = B + au moins un autre doigt non replié, B3D / C3D. Paramètres préparés dans
  gestures.yaml, **non branchés** sur la machine à états : décision après mesures.
- Protocole : `scripts/record_gestures.py`, 8 séquences étiquetées de 10 s ; pincements
  répétés au métronome (1 par seconde) pour connaître le nombre attendu.
- Analyse : distributions par étiquette ; matrices de confusion image par image (positif =
  pincement maintenu sans sa première seconde) ; épisodes avec hystérésis pour les séquences
  au métronome ; seuil proposé = milieu entre P95 des négatifs et P5 des positifs (marge =
  écart), vérifié par l'indice de Youden (Youden, 1950). Seuils appris sur l'entraînement,
  évalués sur un fichier de test séparé.
- Essai technique : 8 séquences enregistrées et analysées de bout en bout (sans main devant la
  caméra), puis supprimées.
- À faire : enregistrer (plusieurs sessions, plusieurs participants), analyser, puis choisir
  la règle et ses seuils.

## 2026-11-24 — Règle D, métriques par épisode (2 sessions de P01)

- Données : entraînement 16-00-54, test 16-03-49, même participant (P01) : le test mesure la
  stabilité d'une séance à l'autre, pas la généralisation à une autre personne. Main deux
  fois plus petite au test (hand_size médiane 0,116 contre 0,212) : assis plus loin.
- Constats image par image (seuils appris sur l'entraînement) :
  - `ext_autres_max` sépare poing et pincement (marge +0,26, seuil 0,87 ; 3D : +0,22, 0,91) ;
    l'extension de l'index non (chevauchement : l'index se plie pour pincer) ;
  - B / C : rappel test 35-37 % ; D : 58 % (A : 69 %), 7 faux positifs sur 299 images de poing
    au test contre 126 pour A.
- Décision : le pincement « index vers la caméra » (autres doigts repliés) sort du vocabulaire.
- Nouvelle règle D (et D3D) = A + au moins un doigt parmi majeur/annulaire/auriculaire non
  replié, sans condition sur l'index.
- Garde-fou de taille de main (hand_size / médiane des 2 s précédentes) : **réfuté**. Les
  images « décrochées » du pincement maintenu ont une taille de main normale (P95 du rapport
  1,13 contre P5 0,94 pour les images saines, J = 0). Code conservé, désactivé (0).
- Les « ruptures » du pincement maintenu sont des plages de 0,6 à 0,8 s avec pinch_ratio
  ≈ 0,9-1,0, taille de main normale, doigts tendus, à intervalle régulier (~2,5 s) : soit de
  vraies réouvertures pendant le déplacement, soit un décrochage du pouce lié au mouvement.
  Un délai de relâchement (66-150 ms) ne les absorbe pas. À éclaircir (question posée ; le jeu
  de données ne garde pas la position absolue de la main, donc pas la vitesse).
- Métriques par épisode (simulation fidèle à la machine : une image sans main ne relâche pas,
  seule une absence > 1 s le fait) : aucune configuration (A/D/D3D × exit 0,40/0,50 × délai
  0/66/100/150 ms) n'atteint 0 faux glisser et ≤ 1 faux clic au test.
- Cause principale des faux glisser de D : la sortie ne regarde que pinch_ratio. Un poing
  entré par erreur (un doigt dépasse un instant le seuil) reste « pincé » 5,9 s. Piste :
  appliquer aussi la condition « autre doigt ouvert » pour rester en pincement. Autre source :
  main relâchée avec pinch_ratio ≈ 0,3-0,4, entre enter et exit.
- Rien n'est branché sur la machine à états.

## 2026-11-24 — pinch_seg, sortie sur repli, sélection sur l'entraînement

- Réponse de P01 sur le pincement maintenu : « pas sûr » d'avoir gardé le pincement tout le
  long. Les ruptures de ces deux fichiers sont donc traitées comme non fiables
  (`--ruptures-non-fiables`) : affichées, mais hors classement. Consigne de
  record_gestures.py précisée (doigts collés, petits mouvements lents, sans revenir en
  arrière) ; protocole v2, qui enregistre aussi la position et la vitesse du poignet (en
  tailles de main par seconde) pour relier les ruptures aux mouvements rapides.
- Méthode : filtres et classement sur l'ENTRAÎNEMENT seulement. Avec 180 configurations,
  choisir d'après le test reviendrait à régler les paramètres sur le test, qui ne mesurerait
  plus rien ; le test sert uniquement à vérifier le choix.
- `pinch_seg` (pouce -> segment 7-8 de l'index) et `pinch_seg_68` (segment 6-8), recalculés
  à partir des points enregistrés. Distributions proches de pinch_ratio ; seg68 est plus
  petite sur le poing (P95 0,25 contre 0,40), ce qui rapproche le poing du pincement.
  Hypothèse « moins de ruptures » non confirmée sur ces fichiers (ruptures identiques,
  3 / 5), mais ces ruptures ne sont pas fiables : à refaire avec le protocole v2.
- Sortie sur repli (`exit_on_fold`) : supprime les faux glisser au test (D, ratio, 0,25 /
  0,40, 66 ms : 3 -> 0), mais coupe les longs faux épisodes en courts (faux clics 6 -> 9) et
  ajoute des ruptures au test (5 -> 8).
- Résultat : aucune des 180 configurations n'atteint 0 faux glisser ET au plus 1 faux clic
  sur l'entraînement (au mieux 4 faux clics). Configuration d'origine : 16 faux clics et
  8 faux glisser à l'entraînement.
- Décision (P01) : brancher malgré tout la 1re configuration du classement, D, ratio,
  enter 0,25 / exit 0,40, délai 66 ms, sans sortie sur repli. Chiffres entraînement / test :
  faux clics 4 / 6, faux glisser 0 / 3, répétés 8/10 / 9/10, contre 16 / 11 faux clics et
  8 / 11 faux glisser pour la configuration d'origine. folded_below = 0,87 (appris).
- Branchement : la machine à états délègue l'entrée / sortie du pincement à PinchHysteresis
  (même code que l'analyse). `rule: A` + `release_grace_ms: 0` redonne l'ancien comportement
  (tests d'origine inchangés). Démarrage réel vérifié sans erreur ; pas encore testé en usage.
- À faire : sessions protocole v2 (plusieurs participants), puis réévaluer pinch_seg,
  exit_on_fold et le délai sur des ruptures fiables, et relier les ruptures à la vitesse du
  poignet.
