sudo docker build -t mahjong:1.0 .
sudo docker run -d -p 2222:22 -v /data/satori_hdd4/Ren/Code:/home/Code --name ren_mahjong mahjong:1.0
