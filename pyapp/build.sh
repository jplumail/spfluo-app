set -x

mkdir build
cd build
curl https://github.com/ofek/pyapp/releases/download/v0.21.1/source.tar.gz -Lo pyapp-source.tar.gz
tar -xzf pyapp-source.tar.gz
cd pyapp-v0.21.1
export PYAPP_PROJECT_NAME=spfluo-app
export PYAPP_PROJECT_VERSION=0.1.4
export PYAPP_IS_GUI=1
export PYAPP_PIP_EXTRA_ARGS="--index-url https://spfluo.ovh/jean/dev/+simple"
cargo build --release
mv target/release/pyapp spfluo-app && chmod +x spfluo-app
mv spfluo-app ../spfluo-app
