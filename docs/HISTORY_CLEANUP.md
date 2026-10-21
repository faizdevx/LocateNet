# Purging personal data from git history

Early commits of this repository contain data that must not be public:

- `database.db`: case records, public submissions (names, mobile numbers, emails) and the
  original `admin` account hash.
- `resources/`: face crops and sighting photos of real people.
- `__pycache__/`: compiled files (noise).
- `yolov8n.pt`: a 6 MB model file (re-downloadable, bloats the repo).

Untracking them (done in this branch) only hides them from new commits. Every old commit still
contains them, and anyone who cloned or forked the repository already has a copy. Treat the data as
**already exposed**, and rotate the admin password regardless of whether you rewrite history.

## WARNING: this cannot be undone

Rewriting history changes **every commit hash**, invalidates open pull requests and existing clones,
and needs a force-push. Do it once, deliberately, after merging `week1-hardening`.

- Make a backup first (step 1). Keep it somewhere private; it still contains the data.
- Tell every collaborator to delete their clones and re-clone afterwards. If they push an old
  clone, the purged data comes back.
- Forks, GitHub's cached views of old commit URLs and any pull request refs can keep the data.
  After the force-push, contact GitHub Support to run garbage collection and remove cached views,
  and delete or re-create forks you control.

## Steps

Requires [git-filter-repo](https://github.com/newren/git-filter-repo)
(`pip install git-filter-repo`).

```bash
# 1. Private backup of the full history (keep it outside the project)
git clone --mirror https://github.com/faizdevx/LocateNet.git LocateNet-backup.git

# 2. Work in a fresh clone (filter-repo refuses to run in a used clone)
git clone https://github.com/faizdevx/LocateNet.git LocateNet-clean
cd LocateNet-clean

# 3. Remove the paths from every commit on every branch and tag
git filter-repo --invert-paths \
  --path database.db \
  --path resources/ \
  --path yolov8n.pt \
  --path-glob '**/__pycache__/*' \
  --path-glob '__pycache__/*'

# 4. Check that nothing is left (each command must print nothing)
git log --all --oneline -- database.db resources yolov8n.pt
git rev-list --all --objects | grep -E 'database\.db|^[0-9a-f]+ resources/|yolov8n\.pt|__pycache__'

# 5. filter-repo removes the remote on purpose; add it back, then force-push everything
git remote add origin https://github.com/faizdevx/LocateNet.git
git push origin --force --all
git push origin --force --tags
```

## After the force-push

1. Rotate the admin password. The old `admin123` hash was stored in `database.db`. Create a new
   password with `LOCATENET_ADMIN_PASSWORD` or a fresh `python init_system.py` on a new database.
2. Rotate `LOCATENET_SECRET_KEY`. The old key was hardcoded in `app.py`.
3. Ask collaborators to re-clone (do not `git pull`).
4. Contact GitHub Support (cached views, orphaned objects, pull request refs).
5. Keep your local `database.db` and `resources/` files; they are now ignored by `.gitignore`.
6. If real people appear in `app_snaps/` screenshots, review those images too; they are not
   covered by the commands above.
7. Consider notifying the people whose data was exposed (see the DPDP Act 2023 note in the README).
