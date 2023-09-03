# GAN
https://medium.com/@mnkrishn/gan-build-using-tensorflow-3d9ca7cef21f

# Keras debugging tips
https://www.google.com/search?client=safari&rls=en&q=tf+keras+debugging+tips&ie=UTF-8&oe=UTF-8

# Log in to server
ssh bhsoft@117.6.135.148 -p 8691

# Password
bhsoft132

# Server
folder = eric
conda env = ericenv


# Upload file from local to server
➜  ~ 
scp -r -P 8691 path_local path_server

# Download file from server to local
scp bhsoft@117.6.135.148  /Local/Path/
scp -r -P 8691 path_server path_local


# Dowload checkpoints and results into the projects main folder
scp -r -P 8691 bhsoft@117.6.135.148:/home/bhsoft/eric/DCGAN_CELEB_A/results /Users/deric_lee/Downloads/Server/result


Generator and discriminator architecture

https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


# Used Commands 
python main.py --