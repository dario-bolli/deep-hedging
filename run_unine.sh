source myenv/bin/activate

# If you want more info about available options of the main, do python ./main_Deep_Heedging.py -h

for d in 5 , 10 , 15 , 30
do
  for m in 1 , 2
  do
    for maxT in 1 , 3 , 6 , 11 , 30
    do
        srun -N 1 -n 1 -c 63 --time 480 \
            python main_Deep_Heedging.py --d d --maxT maxT \
             --epochs 80 --model Deep_Hedging_Model_Transformer \
             --m m
    done
  done
done