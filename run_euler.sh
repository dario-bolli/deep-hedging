
# to create venv follow this https://scicomp.ethz.ch/wiki/Python_virtual_environment, then manually install
# pip install keras-tcn --no-dependencies
# pip install QuantLib
# Then you need to make the file executable like this
# chmod +x main_Deep_Heedging.py
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy

source myenv/bin/activate

# If you want more info about available options of the main, do python ./main_Deep_Heedging.py -h
bsub -G "s_stud_infk" -n 4 -W 08:00\
     -R "rusage[mem=4800]" -R "rusage[ngpus_excl_p=1]" \
      python main_Deep_Heedging.py --N 93 --d 2 --maxT 10 --epochs 80
