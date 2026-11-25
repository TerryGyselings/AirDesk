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

## 2026-11-24 — Premier usage réel : paume de profil, pointage absolu, choix de la main

Retour d'usage réel de P01 (sans --dry-run) : « ça marche bien », avec trois défauts.

1. **Main de profil au repos = faux clic / glisser.** Au repos, main de côté, le pouce
   rejoint l'index et les autres doigts sont à moitié ouverts : la règle D accepte.
   - Descripteurs ajoutés : `palm_width` (distance 5 -> 17 / taille de main, 2D homogène)
     et `palm_facing` (|composante z| de la normale (P5−P0) × (P17−P0)).
   - Condition d'entrée `require_palm_facing` + `min_palm_width`, et nouvelle séquence
     négative `main_repos_profil` (protocole v3).
   - **Surprise des mesures** : les vrais pincements « OK » de P01 ont une paume étroite
     (palm_width médiane 0,46, P5 0,26 ; main synthétique de face : 0,80). La valeur
     provisoire synthétique de 0,40 rejetait 22 % / 35 % des images de pincement maintenu
     et faisait tomber les pincements répétés de 9/10 à 5/10 au test. **Décision (P01) :
     0,22**, sous le P5 des vrais pincements ; elle écarte encore une main franchement de
     profil (synthétique à 80° : 0,14). À régler avec une session v3.
   - Sur les anciens fichiers, ni palm_width ni palm_facing ne séparent pincement et
     négatifs (J ≈ 0). C'est attendu : ils n'ont pas de séquence de main de profil.
2. **Désynchronisation main / curseur** (pointage relatif) : nouveau pointage **absolu** par
   défaut. La zone d'interaction (15 %..85 % × 15 %..75 % de l'image) est projetée sur
   l'écran principal (PyMonCtl ; pywinctl.getScreenSize est déprécié), et collée aux bords
   hors zone. Le curseur reste figé pendant un pincement court.
   - Tremblement : en absolu, 1 px caméra ≈ 5,7 px écran (contre 3,75 en relatif). Filtre 1€
     réglé par mode ; valeur de départ en absolu min_cutoff 0,5 (au lieu de 1,0), car pour
     un passe-bas et un bruit blanc, l'écart-type restant varie comme √(coupure)
     (Casiez et al., 2012).
   - Mesure ajoutée : position du curseur et vitesse du poignet dans `--log` ;
     `analyse_log.py` donne l'écart-type du curseur pendant les périodes immobiles
     (≥ 1 s, poignet < 0,5 taille de main/s). Objectif < 2 px, **pas encore mesuré**.
   - Multi-écrans : `pointer.screen: primary | all`. Limite : le rectangle englobant de
     deux écrans de formats différents contient des zones sans écran.
3. **Suivi de la mauvaise main** : 2 mains détectées. En veille, on prend la main qui fait le
   geste d'activation (préférence `hand.preferred`). Une fois actif, on suit la main par
   continuité (poignet le plus proche, saut maximal de 1,5 taille de main), et l'autre est
   ignorée. Le verrou saute au retour en veille.
   - **Latéralité MediaPipe inversée** avec notre image en miroir : sur 2 sessions main
     droite, « Left » sur 97 % des images. Corrigé à un seul endroit
     (`hand_tracker.user_side`).
- Vérifié : 178 tests ; démarrage réel sans erreur (écran 2560x1440 détecté, colonnes du
  log). **Pas encore testé en usage réel** : pointage absolu, deux mains, main de profil.

## 2026-11-24 — Deuxième essai réel : zone, main baissée, rôles des mains, performances

Retour d'usage de P01 (commit 843a220) :
- pointage absolu **précis** (croix d'une fenêtre en 1-2 essais), léger tremblement non
  gênant ; mais **zone inconfortable** (il faut reculer et lever le bras) ;
- main au repos de profil, pouce contre l'index : **encore un clic** avec min_palm_width 0,22 ;
- deux mains : la main activée reste suivie, mais **quand elle sort du champ, l'autre main
  prend le contrôle** ;
- log 17-33-21 : 24 i/s et 18,8 ms de traitement (contre 30 i/s et 9,2 ms), section
  « tremblement » absente, issue « aucune » pour les vrais clics.

Analyses et décisions :
- **Changement de main, cause** (log 17-33-21) : la main droite sort à 43,0 s ; 1 s plus
  tard, veille ; le sélecteur prend la main gauche, à demi fermée, qui **s'ouvre** (4 doigts,
  paume de face 0,96) 300 ms : activation à 44,2 s. Ce n'était pas une erreur de suivi : le
  geste d'activation, trop facile pour une main au repos, pouvait être fait par n'importe
  quelle main. **Décision (P01) : des rôles** (Guiard, 1987, modèle de la chaîne
  cinématique : main dominante pour la précision, main non dominante comme référence).
  Seule la main préférée active et pilote ; l'autre est une main d'appoint, sans effet seule,
  sans rôle dès que la main pilote est perdue (structure prête pour de futurs gestes à deux
  mains, ignorée pour l'instant). Toute activation exige en plus la main ouverte DANS la zone.
- **Main baissée = veille** : pincement seulement dans la zone (`only_in_zone`), veille après
  500 ms sous la zone (`rest_ms`, Release propre si glisser), activation seulement dans la
  zone (`activate_in_zone`, indispensable : sinon une main ouverte posée en bas se
  réactiverait et la machine oscillerait). Règle de conception, indépendante des seuils : elle
  vise aussi la main au repos de profil, souvent basse.
- **Zone** : zone par défaut abaissée de 0,05 (y 0,20..0,80) : en usage réel,
  point de pointage collé en bas 3 % du temps contre 0,15 % en haut, hauteur médiane 0,50.
- **Performances** : avec une main visible, 8,6 ms (modèle 1 main) contre 19,2 ms (modèle 2
  mains). MediaPipe ne suit une main qu'une fois TOUTES les mains demandées trouvées ; sinon
  il relance la détection de paume à chaque image. num_hands étant fixé à la création, deux
  modèles : 2 mains en veille (choisir la main pilote) et après une perte, 1 main une fois
  actif. Le gain n'apparaît qu'une fois actif (en veille, main visible : toujours ≈ 19 ms).
  Caméra à 24,8 i/s en fin de journée : avertissement au démarrage sous 25 i/s.
- **Tremblement, mesure corrigée** : la section s'affichait bien avec le code actuel, mais la
  mesure comptait la dérive lente (médiane 16 px, max 30 px). Désormais : tremblement =
  écart-type autour d'une moyenne glissante. **Le résultat dépend de la fenêtre** : le curseur
  est déjà lissé par le 1€ (coupure 0,5 Hz), son tremblement restant est lent, et une fenêtre
  de 300 ms le prend en partie pour de la dérive (mesure optimiste ; vérifié par un test :
  sinusoïde ±3 px à 1 Hz, 2,1 px mesurés sur 1 s, bien moins sur 300 ms). Objectif jugé sur
  1 s. Log 17-33-21 : **médiane 2,75 px, maximum 4,36 px sur 1 s** (0,45 à 3,3 px sur 300 ms),
  au-dessus de l'objectif de 2 px mais jugé « non gênant » par P01.
- **Issue des clics** : avec release_grace_ms = 66, le clic est émis après la fin mesurée
  sur pinch_ratio ; l'issue est désormais cherchée jusqu'à fin + délai + 1 image. Log
  17-33-21 : les 5 clics et 3 glisser sont retrouvés.
- **Main de profil** : pas de session v3 encore. L'analyse comparative (main_repos_profil
  contre pincement maintenu et répétés ; pinch_ratio, pinch_seg, palm_width, palm_facing,
  ext_autres_max, wrist_y ; marge et Youden dans le meilleur sens) est prête.
  min_palm_width reste à 0,22.
- À faire : étalonnage de la zone par poste, session v3, puis décider de l'indice contre la main de
  profil ; vérifier en réel le gain de fluidité une fois actif.

## 2026-11-24 — Étalonnage de la zone mis de côté

- Deux essais d'étalonnage de la zone d'interaction, tous deux **non concluants en réel** :
  1. 5 s de mouvements libres, zone = percentiles 3-97 : trop court, zone incohérente
     (x 0,05..0,89, y 0,37..0,86). Jamais commité sur main.
  2. Étalonnage guidé (5 cibles en plein écran, capture quand la main est immobile 1 s,
     zone depuis les 4 coins étirée de 3 %, validation avant d'enregistrer) : **les cibles
     bas-gauche et bas-droite ne se valident pas**.
- **Cause : la position de la caméra.** Placée haut, elle voit mal le bas : en position
  confortable pour viser les coins du bas, la main sort du cadre, et aucune capture n'est
  possible. Le problème n'est pas l'algorithme, mais le champ de la caméra par rapport à la
  posture de travail.
- L'étalonnage guidé est conservé sur la branche **`wip/etalonnage-guide`** (commit
  5400355, « non validé en réel : coins bas hors du champ quand la caméra est haute »),
  avec sa logique testée (StillnessDetector, zone_from_corners, center_offset) et son
  affichage plein écran.
- Sur main : zone de l'ancien étalonnage retirée de config/local.yaml ; retour à la zone par
  défaut de gestures.yaml (x 0,15..0,85, y 0,20..0,80).
- Pistes pour plus tard :
  - **conseiller à l'utilisateur** de placer la caméra plus bas (sur le bureau, devant
    l'écran) ou inclinée vers le bas, pour que la main reste dans le champ en position
    basse et détendue (conseil ajouté au README, section Installation) ;
  - **afficher un avertissement** dans la fenêtre de debug (et pendant l'étalonnage) quand
    la main sort par le bas de l'image, ou quand le poignet s'approche du bord bas ;
  - reprendre ensuite l'étalonnage guidé de la branche, une fois la caméra bien placée.

## 2026-11-24 — Essai réel complet ; clic devenu glisser ; main qui pivote vers la gauche

Retour d'usage de P01 (logs 18-19-43 et 18-22-34, commit fb6d299) :
- **OK** : activation, glisser de fenêtre (7/7), croix de fermeture, main baissée = veille,
  poings, gestes parasites, main gauche ignorée. Tremblement médian **1,43 px** sur 1 s
  (objectif < 2 px atteint ; maximum 4,97 px).
- **Constat clé : le pivot de la main vers la gauche.** En pointage absolu, quand la main
  droite va vers la GAUCHE de l'écran, l'avant-bras pivote et la main se retrouve de profil ;
  elle fait alors des clics / pincements involontaires, comme la main au repos de profil. La
  condition sur la paume (palm_width) ne peut donc pas être la solution : elle refuse aussi
  les vrais clics à gauche. Preuve, log 18-19-43 à 82 515 ms : vrai clic à gauche (curseur
  x ≈ 290 px) refusé par la paume (palm_width 0,06-0,15 < 0,22). **`require_palm_facing`
  désactivé.**
- **Clic devenu glisser (79 296 ms, 344 ms < click_max_ms 400).** Cause : en relâchant
  lentement, pinch_ratio oscille autour d'exit (0,3983, 0,4035, 0,3969, 0,4014, 0,3970...) ;
  chaque passage juste sous 0,40 annulait la confirmation du relâchement (66 ms), et le Press
  partait à 422 ms. Corrections : (1) pas de Press pendant qu'un relâchement est en cours de
  confirmation, la durée s'arrêtant au début du relâchement ; (2) un relâchement en cours
  n'est annulé que si la mesure redescend sous exit − 0,05 (`release_cancel_margin`). La
  correction (1) seule ne suffisait pas sur ce cas réel (Press à 469 ms) ; avec (2), la
  séquence réelle donne bien un clic (testé). Contrepartie : pendant un glisser, laisser la
  mesure filer juste au-dessus de 0,40 puis revenir à 0,36 ne rattrape plus le glisser.
- **Pincements refusés sans explication** : 82 515 ms = paume (ci-dessus) ; 95 515 ms =
  AirDesk en veille (main sous la zone). Nouvelle colonne `entry_block` du log, et raison
  affichée par analyse_log.py pour chaque pincement sans issue.
- **Performances** : 24 i/s (caméra), 12,5 ms en moyenne mais 26,6 ms au 95e percentile.
  Colonne `tracker_hands` (1 ou 2 mains demandées) et temps par mode dans analyse_log.py pour
  localiser les pics (hypothèse : images en mode 2 mains, en veille ou après une perte).
- **Hypothèse pour la main de profil**, à tester avec la session v4 :
  - de profil, le pouce passe DEVANT l'index vu de la caméra : pinch_ratio en 2D voit un
    contact qui n'existe pas en profondeur → `pinch_ratio_3d` (z mis à l'échelle de x,
    divisé par la taille de main 3D) ;
  - un vrai pincement FERME activement les doigts, un pivot non → `closing_speed` (baisse de
    pinch_ratio par seconde sur ~200 ms) et `closing_speed_3d` ;
  - règles candidates E (= D avec pinch_ratio_3d), F (= D + closing_speed minimale) et F3D.
    Non branchées : décision après la session v4.
- Protocole v4 : `pincements_repetes_gauche` (positif), `main_repos_profil_gauche` et
  `va_et_vient_gauche` (négatifs, un aller-retour centre ↔ gauche toutes les 2 s, le geste qui
  provoque les faux clics en réel).

## 2026-11-24 — Clic « comme une souris » (mode press) ; moins de mode 2 mains

Retour d'usage (log 18-43-22, commit 1c3379b) : le clic devenu glisser est corrigé (12 clics,
dont 328 et 344 ms, aucun glisser parasite) et les clics à gauche passent. **Mais un délai
est ressenti à chaque clic.**

- **Cause de la latence** : en mode « release », le clic n'est émis qu'au relâchement
  confirmé, soit durée du pincement + release_grace_ms. Estimé depuis pinch_ratio :
  **363 ms en moyenne, 422 ms au P95** entre le contact des doigts et le clic (log 18-43-22),
  sans compter la caméra ni MediaPipe.
- **Décision : `click_mode: press` par défaut**, comme une souris : Press dès que le
  pincement est présent sur `min_press_frames` = 2 images (filtre anti-glitch, une image
  isolée ne produit rien), Release au relâchement confirmé. Le système distingue clic,
  double-clic et glisser. Latence attendue : environ 1 image (≈ 40 ms à 24 i/s).
  `click_mode: release` reste disponible pour comparer, la colonne `contact_ms` du log et
  analyse_log.py mesurent la latence des deux modes.
- **Curseur figé pendant l'appui, puis glisser** : sinon le tremblement de la main pendant
  le pincement déplacerait le clic. Le curseur reste figé tant que la main ne s'est pas
  éloignée du point de contact de plus de `drag_threshold`. Déplacement du poignet pendant
  24 vrais clics (logs 18-19-43, 18-22-34, 18-43-22, en tailles de main) : 20 sous 0,12,
  traîne 0,23 / 0,34 / 0,36 / 0,50. **Choix : 0,15**, à revoir avec la colonne
  `anchor_distance` (en mode press, l'utilisateur sait que le curseur est figé).
  Contrepartie : un clic qui bouge plus que ça (la traîne) devient un petit glisser.
- **Rattrapage au début du glisser** (`drag_catchup_ms: 100`) : en pointage absolu, au
  dépassement du seuil le curseur devrait sauter du point figé à la main (≈ 0,15 taille de
  main, de l'ordre de 100 px, à vérifier : analyse_log.py affiche ce saut pour chaque
  glisser). Il rejoint la main par interpolation linéaire en 100 ms ; 0 = saut direct.
- **Perte de la main ou main baissée, bouton enfoncé** : Release, y compris pendant un
  appui sur place (mode press).
- **Mode 2 mains** : 479 images sur 1106 (43 %) dans le log 18-43-22, dont **435 en veille
  sans aucune main visible**, 24 en veille avec une main, et seulement 20 une fois actif
  (19 juste après une perte). Chercher 2 mains coûte 18,9 ms par image sur un champ vide,
  contre 11 ms pour une main. Nouvelle règle (`TrackerModePolicy`) : en veille, 2 mains
  seulement si une main NON pilote est visible sans la main pilote (pendant 1 s) ; une fois
  actif, seulement après 3 images d'affilée sans main pilote. Essai sans fenêtre, main
  pilote visible en veille : 100 % des images en mode 1 main, 9,1 ms en moyenne (18,4 ms
  avant la règle).
- Correction de l'analyse : le nombre de glisser comptait chaque Press ; il compte désormais
  les passages à l'état « glisser », pour que les clics du mode press (Press + Release) ne
  soient pas comptés comme des glisser.

## 2026-11-24 — Double-clic, pincements en mouvement, tremblement

Retour d'usage du mode press (log 19-12-46) : latence 42 ms (P95 62) au lieu de ~363 ms,
9,3 ms par image, 2 % d'images en mode 2 mains. Restaient le double-clic et 2 pincements en
trop.

- **Double-clic, cause 1** : entre deux tapes rapides, les doigts ne restent ouverts que 1 à
  2 images ; le délai de relâchement (66 ms) et la marge d'annulation fusionnaient les tapes.
  Rejoué dans la machine (64,9-67,8 s, 14 tapes) : 5 Press / 4 Release avant ; **11 / 11 avec
  `release_fast_margin: 0.05`** (relâchement immédiat au-dessus de 0,45) ; 8 / 7 avec 0,15.
  Choix de 0,05 plutôt que 0,15 : pics pendant un glisser qui continue (24 glisser, 4 logs) :
  3 au-dessus de 0,40, aucun au-dessus de 0,45. Les 2 tapes restantes ne dépassent jamais
  0,40. Le clic réel de 79 296 ms reste un clic (testé).
- **Double-clic, cause 2** : le curseur bougeait de 1 à 4 px entre deux clics (Windows :
  ±2 px). Verrou `double_click_lock_ms: auto` (GetDoubleClickTime, 500 ms ici) : après un
  clic, curseur figé tant que la main reste à moins de drag_threshold ; rejoué : les 11
  Press au même pixel. Fin du verrou : rattrapage progressif.
- **Pincements en mouvement** (72 140 et 72 421 ms, devenus des glisser) : vitesse du
  poignet au contact 3,06 et 1,87 taille de main/s, contre une médiane de 0,34 (P95 1,65) sur
  80 pincements suivis d'un clic ou d'un glisser. Seuil 1,8 : les 2 refusés, 3 des 80 aussi
  (3,13 / 3,63 / 4,82, peut-être involontaires). Marge mince : `max_contact_speed` laissé à
  0 (désactivé) ; règle candidate G (D + vitesse) ajoutée à l'analyse v4.
- Saut du curseur au début des glisser (sans rattrapage) : médiane 95 px, max 153.
- **Tremblement : médiane 3,01 px sur 1 s** (1,43 px au log 18-19-43), au-dessus de
  l'objectif de 2 px. À revoir plus tard : filtre ou éclairage (caméra à 24 i/s le soir).

## 2026-11-25 — Curseur en temps réel, clics épinglés, tapes fusionnées

Essai réel du double-clic (log 16-35-12, 30 i/s, latence 34 ms) : deux défauts.

**1. Curseur figé gênant.** Le verrou du double-clic (500 ms après chaque clic) est supprimé.
- **Clics épinglés** : la machine émet `PressAt(u, v)` / `ReleaseAt(u, v)` (pointage
  absolu, mode press). Press au point du contact, Release au même point pour un clic sur
  place ; au-delà de `drag_threshold`, glisser et Release simple à la position réelle.
- **Le curseur reste figé bouton enfoncé sur place** (décision après mesure) : la main dérive
  pendant les clics de 4 à 11 px en médiane (logs 19-12, 19-28, 16-35), jusqu'à 118 px, et
  **57 clics sur 82 dépassent 4 px**, le seuil de glisser de Windows (SM_CXDRAG). Un curseur
  en temps réel bouton enfoncé ferait partir des débuts de sélection ou de glisser-déposer.
  Avant le Press (1 image) et après le Release, le curseur suit la main (rattrapage
  `drag_catchup_ms`).
- **Écart avec la demande** : l'aller-retour « point épinglé → position réelle » juste après le
  Press n'est pas fait. Bouton enfoncé, ce retour serait lui-même un glisser (jusqu'à ~50 px
  pour un clic aimanté). L'actionneur place le curseur au point épinglé puis appuie ; c'est
  la machine qui le ramène vers la main après le Release.
- **Double-clic aimanté** : un 2e Press à moins de `double_click_ms` (auto = GetDoubleClickTime,
  500 ms) du Press précédent et à moins de `double_click_snap` tailles de main de son contact
  est émis au pixel exact du 1er clic (Windows exige ±2 px). Écart entre deux contacts
  rapprochés (< 500 ms, 3 logs en mode press) : 1 à 21 px ; 570 à 700 px par taille de main
  (rapport cursor / anchor_distance), donc 0,03 à 0,037. **Choix : 0,08** (45 à 56 px,
  marge d'environ x2). Rejoué sur les 14 tapes du log 19-12-46 : les 11 Press et 11 Release
  au même point.
- Mesure : colonnes `pin_kind` et `pin_offset_px` (écart main → point épinglé), répartition
  dans analyse_log.py.

**2. Doubles-clics rapides non reconnus : tapes fusionnées.** Courbes pinch_ratio :
- 13 250 ms (« clic » de 796 ms) : creux 0,11 → remontée à **0,392** → creux 0,128, puis
  plateau 0,399-0,401 pendant ~200 ms qui retarde aussi le Release ;
- 58 375 ms (« clic » de 406 ms) : creux 0,162 → **0,338** → creux 0,105 ;
- 15 171-15 546 ms : **pas de fusion** (Press 15 218, Release 15 406, Press 15 577, Release
  15 875 ; 359 ms entre les Press, curseur verrouillé), séquence identique au double-clic
  réussi de 52 577. **Non expliqué par le log** (cible qui change au 1er clic ?).

Dans les deux premiers cas, la remontée entre les tapes ne dépasse pas exit (0,40) : un seul
appui. **Relâchement relatif** (`release_rise`, pinch_rules.py) : relâchement quand la mesure
remonte de `release_rise` au-dessus de son minimum depuis le contact (et au-dessus de enter).
C'est l'idée de la proéminence d'un détecteur de pics. Rejeu sur pinch_ratio (`rise_splits`, log_analysis.py),
62 glisser dans 9 logs ; une ligne = un appui coupé en deux :

| release_rise | garde | ruptures de glisser | appuis de clic coupés | 13 250 / 58 375 séparés |
|---|---|---|---|---|
| 0,15 | aucune | 20 (dont 8 dans les logs du mode press) | 26 | oui / oui |
| 0,12 | main < 95 px du contact | 9 | 26 | oui / oui |
| **0,15** | main < 95 px du contact | **7** (17-22-29 : 4, 17-33-21 : 3) | 26 | oui / oui |
| 0,18 | main < 95 px du contact | 7 | 22 | oui / **non** |

- **Garde** : la machine ne l'autorise que pendant un appui sur place (état PINCHING), jamais
  pendant un glisser.
- Les 7 ruptures restantes sont toutes dans des logs du mode release, 170-234 ms après le
  contact (creux 0,07-0,15, pics 0,27-0,35). Les vraies tapes se font 79-313 ms après le
  contact, avec des pics de 0,27 à 0,55 : ni le temps ni l'amplitude ne les distinguent.
  Aucune rupture dans les 3 logs du mode press.
- Risque : en mode press, une rupture au début d'un glisser donne clic + glisser, soit sous
  Windows un « double-clic glissé » (sélection par mots).
- Rejoué dans la machine (règle D) sur les deux épisodes réels : 2 Press / 2 Release chacun
  avec 0,15, 1 seul sans.
- **Décision : livré désactivé (`release_rise: 0.0`), proposition 0,15.** analyse_log.py
  affiche le rejeu de chaque log pour 0,12 / 0,15 / 0,18 ; à activer après un essai.

**Commit séparé** : le mode press et TrackerModePolicy (état du log 19-12-46 : 42 ms, ~9 ms
par image) ont été commités à part (f72f52c), testés dans une copie de travail temporaire ;
le travail sur le double-clic reste dans un second commit.

## 2026-11-25 — Essai réel : release_rise activé à 0,15

Essai réel avec `release_rise: 0.15` (log 17-10-10, 1603 images en 53,4 s, 30 i/s) :
**concluant**. Les doubles-clics rapides passent, les clics rapides sont séparés, aucun glisser
n'est coupé, et il n'y a plus d'à-coup du curseur après un clic.

- Latence contact → Press : moyenne 34 ms, P95 47 ms (30 appuis).
- **4 doubles-clics aimantés** : écart main → point du 1er clic, médiane 2,9 px au Press
  (max 8,1), 7,1 px au Release (max 12,2).
- Autres clics épinglés : press 26 (médiane 1,7 px, max 18), release 24 (médiane 8,3 px,
  P95 38, max 42 px) ; c'est la dérive pendant l'appui que l'épinglage annule.
- Rejeu de release_rise 0,15 sur ce log : 0 rupture de glisser.
- Déplacement de la main pendant les clics : max 0,10 taille de main, sous drag_threshold
  (0,15).
- **Décision : `release_rise: 0.15` activé.**
- **À surveiller** : 4 pincements refusés pour « autres_doigts » (règle D : majeur, annulaire
  et auriculaire repliés), dont 3 en début de session (11 860, 13 735, 15 469 ms, 375 à
  562 ms chacun) et un à 58 203 ms (360 ms). Hypothèse : le geste « OK » (autres doigts
  ouverts) n'est pas toujours respecté, surtout en début de session. Si ça persiste,
  comparer les extensions des doigts de ces pincements avec `folded_below` (0,87).
