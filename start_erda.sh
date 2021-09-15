# Startup ssh agent
eval "$(ssh-agent -s)"
ssh-add ~/work/.ssh/id_rsa

# Go to folder and branch
cd ~/work/MasterThesis2021_src/
git checkout lorentz-block-experiment-v1-31.08.2021
git pull

# Conda env stuff
conda init
conda activate python3
conda install -y --file requirements.txt --channel conda-forge

# Install git autocomplete
curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash
chmod u+x ~/.git-completion.bash