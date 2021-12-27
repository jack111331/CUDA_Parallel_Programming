# set N size array
N=1024

mkdir -p result

# Test with 1 GPU
# Experiment different block size(threadsPerBlock)
# Not use CPU
for i in 32 16 8 4;
do
	rm Input
	echo -e "1 1\n0\n$N $N\n$i $i\n0" >> Input
	echo "Current block size is $i, the input is:"
	cat Input
	./heatDiffusion2D_NGPU < Input > result/Output_1GPU_${i}
	mkdir -p result/1GPU_block_${i}
	mv result/Output_1GPU_${i} result/1GPU_block_${i}/
	mv phi_initial.dat result/1GPU_block_${i}/
	mv phi_GPU.dat result/1GPU_block_${i}/
done

# Test with 2 GPU
# Experiment different block size(threadsPerBlock)
# Not use CPU
for i in 32 16 8 4;
do
	rm Input
	echo -e "1 2\n0 1\n$N $N\n$i $i\n0" >> Input
	echo "Current block size is $i, the input is:"
	cat Input
	./heatDiffusion2D_NGPU < Input > result/Output_2GPU_${i}
	mkdir -p result/2GPU_block_${i}
	mv result/Output_2GPU_${i} result/2GPU_block_${i}/
	mv phi_initial.dat result/2GPU_block_${i}/
	mv phi_GPU.dat result/2GPU_block_${i}/
done
