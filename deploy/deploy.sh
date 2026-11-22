#!/usr/bin/env bash
# Deploy DogBot to the Pi and (re)start the service.
#
#   ./deploy/deploy.sh            # uses admin@192.168.7.208
#   DOGBOT_HOST=admin@ai-robot ./deploy/deploy.sh
#
# Keeps the Pi's config/, data/ and .venv. The venv uses --system-site-packages
# for picamera2/libcamera/hailo_platform, which need the *system* numpy 1.24.x:
# pip runs with a constraints file pinning that exact version, and the script
# fails if numpy changes anyway.
set -euo pipefail

HOST="${DOGBOT_HOST:-admin@192.168.7.208}"
DEST=/home/admin/dogbot
cd "$(dirname "$0")/.."

echo "==> rsync to $HOST:$DEST"
rsync -az --delete \
  --exclude /.venv/ --exclude /config/ --exclude /data/ \
  --exclude .git/ --exclude /.superpowers/ --exclude __pycache__/ \
  --exclude '*.egg-info/' --exclude .pytest_cache/ --exclude hailort.log \
  ./ "$HOST:$DEST/"

echo "==> venv, install, systemd unit"
ssh "$HOST" "DEST=$DEST bash -s" <<'REMOTE'
set -euo pipefail
cd "$DEST"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv
fi
before=$(.venv/bin/python -c 'import numpy; print(numpy.__version__)')
case "$before" in
  1.*) ;;
  *) echo "venv numpy is $before, expected system 1.x; aborting" >&2; exit 1 ;;
esac
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
printf 'numpy==%s\nnumpy<2\n' "$before" > "$constraints"
.venv/bin/pip install -q --disable-pip-version-check -c "$constraints" -e .
after=$(.venv/bin/python -c 'import numpy; print(numpy.__version__)')
if [ "$after" != "$before" ]; then
  echo "numpy changed $before -> $after; picamera2/hailo need the system numpy" >&2
  exit 1
fi
echo "numpy $after (unchanged)"
mkdir -p config data
sudo install -m 0644 deploy/dogbot.service /etc/systemd/system/dogbot.service
sudo systemctl daemon-reload
sudo systemctl enable dogbot
sudo systemctl restart dogbot
sleep 3
systemctl is-active dogbot
REMOTE
echo "==> http://${HOST#*@}:8080"
