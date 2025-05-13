str1="run"
str2="debug"

if [ $1 = $str1 ];
then
    CUDA_VISIBLE_DEVICES=$1 accelerate launch --main_process_port=2000 ./acc_main.py train --model $2  --dt $3
elif [ $1 = $str2 ];
then
    CUDA_VISIBLE_DEVICES=$1 python3 ./acc_main.py train --model $2  --dt $3 
else
    echo "$1 is wrong, please choose run or debug."
fi

# try to run this model
# sh run1.sh run CNRMS cumsum-sum nce 0,1,2,3 large
# or debug
# sh run1.sh debug CNRMS cumsum-sum nce 1 large


# sh run1.sh run MINER 2,3 small
# sh run1.sh run UCMINER cumsum-sum nce 2 small