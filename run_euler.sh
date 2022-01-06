
# to create venv follow this https://scicomp.ethz.ch/wiki/Python_virtual_environment, then manually install
# pip install keras-tcn --no-dependencies
# pip install QuantLib
# Then you need to make the file executable like this
# chmod +x main_Deep_Heedging.py
# chmod +x run_euler.sh
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy

source myenv/bin/activate

# If you want more info about available options of the main, do python ./main_Deep_Heedging.py -h

for d in 5 10 15 30
do
  for m in 1 2
  do
    for maxT in  1  3 6 11 30
    do
      bsub -G "s_stud_infk" -n 4 -W 08:00\
           -R "rusage[mem=4800]" -R "rusage[ngpus_excl_p=1]" \
            python main_Deep_Heedging.py --d $d --maxT $maxT \
             --epochs 100 --model Deep_Hedging_Model_MLP_CLAMP \
             --m $m
    done
  done
done