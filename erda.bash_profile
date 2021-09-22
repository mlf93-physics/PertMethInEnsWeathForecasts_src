if [ -f ~/.git-completion.bash ]; then
  . ~/.git-completion.bash
fi

alias login="eval '$(ssh-agent -s)'; ssh-add ~/work/.ssh/id_rsa"