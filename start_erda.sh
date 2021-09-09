# Startup ssh agent
eval "$(ssh-agent -s)"
ssh-add ~/work/.ssh/id_rsa

# Go to folder and branch
cd ~/work/MasterThesis2021_src/
git checkout lorentz-block-experiment-v1-31.08.2021
git pull

# Conda env stuff
source activate python3
conda env update --file environment.yml

