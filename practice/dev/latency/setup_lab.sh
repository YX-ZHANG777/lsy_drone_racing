# Lab PC setup. Run from the fork root AFTER unzipping:
#   cd /path/to/fork && unzip -o /path/to/to_lab_1118.zip && source practice/dev/latency/setup_lab.sh
export FORK=$PWD LAB=$PWD/practice/dev/latency
mkdir -p "$LAB/logs"
echo "FORK=$FORK"
echo "LAB=$LAB"
echo "--- fork version (developed on c576683, 2026-10-20):"
git log -1 --oneline
echo "--- changed files (expected: 4 config/level*.toml, 7 new control/ files, practice/):"
git status --short
echo "--- drone id / channel in the configs (must match the drone the TA gives you):"
grep -H -A2 "deploy.drones" config/level3.toml "$LAB/probe.toml" | grep -E "id =|channel ="
echo "--- controller load test:"
python -c "import lsy_drone_racing.control.race_ppo as m; print('OK', m.POLICY)"
