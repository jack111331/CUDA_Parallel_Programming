mkdir -p result

# Experiment different block size(threadsPerBlock)
# Simple sampling using 1 GPU
for i in 6 7 8 9 10 11 12 13 14 15 16;
do
	for j in 2 4 8 16 32 64 128 256 512;
	do
		rm Input
		echo -e "$i\n1\n0\n$j" >> Input
		echo "Simple sampling using 1 GPU, current N is $i, block size is $j, the input is:"
		cat Input
		mkdir -p result/ss/GPU_1/N_${i}_block_${j}/
		./monteCarlo_ss_ngpu < Input > result/ss/GPU_1/N_${i}_block_${j}/Output
	done
done

# Simple sampling using 2 GPU
for i in 6 7 8 9 10 11 12 13 14 15 16;
do
	for j in 2 4 8 16 32 64 128 256 512;
	do
		rm Input
		echo -e "$i\n2\n0 1\n$j" >> Input
		echo "Simple sampling using 2 GPU, current N is $i, block size is $j, the input is:"
		cat Input
		mkdir -p result/ss/GPU_2/N_${i}_block_${j}/
		./monteCarlo_ss_ngpu < Input > result/ss/GPU_2/N_${i}_block_${j}/Output
	done
done

# Important sampling with metropolis algorithm using 1 GPU
for i in 6 7 8 9 10 11 12 13 14 15 16;
do
	for j in 2 4 8 16 32 64 128 256 512;
	do
		rm Input
		echo -e "$i\n1\n0\n$j" >> Input
		echo "Important sampling with metropolis algorithm using 1 GPU, current N is $i, block size is $j, the input is:"
		cat Input
		mkdir -p result/mis/GPU_1/N_${i}_block_${j}/
		./monteCarlo_mis_ngpu < Input > result/mis/GPU_1/N_${i}_block_${j}/Output
	done
done

# Important sampling with metropolis algorithm using 2 GPU
for i in 6 7 8 9 10 11 12 13 14 15 16;
do
	for j in 2 4 8 16 32 64 128 256 512;
	do
		rm Input
		echo -e "$i\n2\n0 1\n$j" >> Input
		echo "Important sampling with metropolis algorithm using 2 GPU, current N is $i, block size is $j, the input is:"
		cat Input
		mkdir -p result/mis/GPU_2/N_${i}_block_${j}/
		./monteCarlo_mis_ngpu < Input > result/mis/GPU_2/N_${i}_block_${j}/Output
	done
done
