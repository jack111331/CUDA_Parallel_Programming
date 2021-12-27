# set GPU Id
gpu_id=0
# set N * N size matrix
N=6400
# Experiment different block size(threadPerBlock)
for i in 4 8 10 16 20 32;
do
        echo -e "$gpu_id\n$N\n$i $i" >> Input
        echo "Current block size is $i, the input is:"
        cat Input
        ./matAdd < Input > Output_$i
        rm Input
done
