# support git lfs
apt-get update -y
apt-get install git-lfs -y
git lfs install
git lfs pull

# For a more compact command line
echo "export PS1='\w$ '" >> ~/.bashrc

# Prerequisite to use cv2 on cnvrg
apt-get update
apt-get install -y libgl1-mesa-glx

wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
unzip -y ninja-linux.zip -d /usr/local/bin/
yes | update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

export PYTHONPATH=/cnvrg;$PYTHONPATH

apt-get install gettext-base
apt-get install psmisc #to enable gpu memory cleanup using fuser -v /dev/nvidia* and then kill -9 PID
envsubst < /cnvrg/clearml_template.conf > /cnvrg/clearml.conf
mv /cnvrg/clearml.conf /root
