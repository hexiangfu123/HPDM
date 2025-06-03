
CUDA_VISIBLE_DEVICES=$5 accelerate launch --main_process_port=3000 ./acc_main_cal.py train --model $2 --norm $3 --loss $4 --gamma 0.01 --pop_size 10 --dt $6 --pal $1 --mode $7 --enc $8 > ./checkpoints/$6_$2_$8_$3_$4_$7_2.log;


# sh run1.sh run HPDM cumsum-sum nce 1,2,3,4 large HPDM bert