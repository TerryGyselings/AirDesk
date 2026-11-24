# CLAUDE.md — consignes pour Claude Code sur AirDesk

## Le projet

IHM gestuelle sans clavier ni souris : une webcam suit la main (MediaPipe), et les gestes
pilotent le curseur et les fenêtres du bureau. Projet étudiant de recherche (ESIR, 2026-2027) :
le code doit rester **lisible et pédagogique**, l'équipe doit comprendre chaque ligne.

## Architecture en couches (à respecter)

1. `capture/` — webcam (OpenCV). Ne connaît rien d'autre.
2. `perception/` — MediaPipe + filtre 1€. Ne connaît ni les gestes ni la souris.
3. `interpretation/` — descripteurs et machine à états. **Logique pure, sans E/S** : reçoit des
   mesures, renvoie des événements (`Move`, `Click`, `Press`, `Release`). Testable sans webcam.
4. `actuator/` — transforme les événements en actions système. Toute dépendance système
   (pynput, PyWinCtl, API natives) vit ici et seulement ici.
5. `ui/` — retour visuel.

`app.py` est le seul endroit qui relie les couches.

## Conventions

- Python 3.10-3.12. Identifiants en anglais, **commentaires et docstrings en français**.
- Code explicite plutôt que malin : boucles `for i in range(...)` lisibles plutôt que
  compréhensions imbriquées, `lambda` ou `zip` compliqués.
- Tous les seuils de gestes vont dans `config/gestures.yaml`, jamais en dur dans le code.
- Distances de la main toujours normalisées par la taille de la main (`features.hand_size`).
- Toute nouvelle logique d'interprétation arrive avec ses tests dans `tests/`.
- Citer la source scientifique dans la docstring quand un choix vient de la littérature.

## Commandes

```bash
pip install -e ".[dev]"
pytest
ruff check .
airdesk --dry-run    # tester sans toucher à la vraie souris
```

## Points d'attention

- MediaPipe en mode VIDEO exige des horodatages strictement croissants (géré dans `capture/camera.py`).
- L'image est retournée en miroir dans `Camera` : gauche/droite de l'utilisateur = gauche/droite à l'écran.
- macOS : autorisations Accessibilité et Caméra. Linux Wayland : pynput ne peut pas bouger la souris.
- Ne jamais enregistrer de vidéo ou d'image de la webcam dans le dépôt (RGPD).

## Façon de travailler

- Toujours répondre en français (explications, plans, résumés, messages de commit).
- Toujours présenter un plan avant de modifier le code.
- À la fin de chaque tâche : lister les fichiers créés/modifiés et expliquer ce qui change et pourquoi (objectif pédagogique).
- Après chaque tâche : lancer pytest et ruff, puis proposer un message de commit en français.
- Ajouter une entrée dans docs/journal.md pour toute décision technique.
