wget http://vision.cs.illinois.edu/projects/divcolor/data.zip
unzip data.zip
rm data.zip
wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
tar -xvzf lfw-deepfunneled.tgz
mv lfw-deepfunneled data/lfw_images
rm lfw-deepfunneled.tgz
