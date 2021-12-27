mkdir -p result

# Use CPU
rm Input
echo -e "0\n81920000\n0 16\n16\n16\n32\n1" >> Input
echo "Current block size is 16, the input is:"
cat Input
mkdir -p result/block_CPU
./hist_1gpu_shmem < Input > result/block_CPU/Output

# Experiment different block size(threadsPerBlock)
# Not use CPU
for i in 2 4 8 16 32 64 128 256 512;
do
	rm Input
	echo -e "0\n81920000\n0 16\n16\n$i\n32\n0" >> Input
	echo "Current block size is $i, the input is:"
	cat Input
	mkdir -p result/gm_block_${i}
	./hist_1gpu_gmem < Input > result/gm_block_${i}/Output_${i}
done

# Experiment different block size(threadsPerBlock)
# Not use CPU
for i in 2 4 8 16 32 64 128 256 512;
do
	rm Input
	echo -e "0\n81920000\n0 $i\n$i\n$i\n32\n0" >> Input
	echo "Current block size is $i, the input is:"
	cat Input
	mkdir -p result/shm_block_${i}
	./hist_1gpu_shmem < Input > result/shm_block_${i}/Output_${i}
done