# launch jupyter notebook
docker run --runtime=nvidia -it --rm --ipc=host --net=host -v $HOME/Desktop/Chatbot/www:/work -w /work flask python main.py