rsync -avz -e "ssh -i ~/.ssh/schemmy" -r \
--exclude='data/' --include='*/' --include='*.py' --exclude='*' \
~/Documents/project4_end2end/ \
chenxinma@10.252.195.8:~/project4_end2end/