wget https://www.cmlab.csie.ntu.edu.tw/~rickyyuan/ADL/HW3/ckpt/final.ckpt
mkdir ckpt
chmod 755 ckpt
mv final.ckpt ckpt
wget https://www.cmlab.csie.ntu.edu.tw/~rickyyuan/ADL/HW3/script.sh
chmod 755 script.sh
bash script.sh
rm script.sh