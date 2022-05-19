# Startup ssh agent
eval "$(ssh-agent -s)"
ssh-add ~/work/.ssh/id_rsa

# Go to folder and branch
cd ~/work/MasterThesis2021_src/
git pull

# Conda env stuff
conda init
conda activate python3
conda install -y --file requirements.txt --channel conda-forge

# Install git autocomplete
curl https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -o ~/.git-completion.bash
chmod u+x ~/.git-completion.bash

# Edit numba config files
echo "cache_dir : '~/work/_nbcompiled'" > ./lorentz63_experiments/.numba_config.yaml
echo "cache_dir : '~/work/_nbcompiled'" > ./shell_model_experiments/.numba_config.yaml