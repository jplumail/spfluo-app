set -x

cd /home/plumail/code/spfluo-app/pyapp
mkdir build

uv pip compile -i https://spfluo.ovh/jean/dev/+simple ../pyproject.toml > build/requirements.txt

cd build
curl https://github.com/ofek/pyapp/releases/download/v0.21.1/source.tar.gz -Lo pyapp-source.tar.gz
tar -xzf pyapp-source.tar.gz
cd pyapp-v0.21.1
export PYAPP_PROJECT_NAME=spfluo-app
export PYAPP_PROJECT_DEPENDENCY_FILE=/home/plumail/code/spfluo-app/pyapp/build/requirements.txt
export PYAPP_PROJECT_VERSION=0.1.4
export PYAPP_IS_GUI=1
export PYAPP_PIP_EXTRA_ARGS="--index-url https://spfluo.ovh/jean/dev/+simple"
export PYAPP_EXEC_SCRIPT=/home/plumail/code/spfluo-app/src/spfluo_app/__main__.py
export PYAPP_PYTHON_VERSION=3.12
export PYAPP_PASS_LOCATION=1
export PYAPP_UV_ENABLED=1
cargo build --release
mv target/release/pyapp spfluo-app && chmod +x spfluo-app
mv spfluo-app ../spfluo-app
