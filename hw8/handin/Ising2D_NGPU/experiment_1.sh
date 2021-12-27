mkdir -p result

# NGx NGy  // 
# device...  // 
# 12345   // seed for RNG
# 200      // the number of sites in each dimension
# 5000    // the # of sweeps for thermalization
# 1000    // the # of measurements
# 10      // the interval between successive measurements
# 100     // the display interval
# 10      // the interval for saving the spin configuration
# 2.269     // the temperature in units of J/k
# 0.0     // the external magnetization
# 1       // Initialize spins configurations (0/1: cold/hot start) 
# 5 10     // block size for even-odd scheme

# Experiment different block size(threadsPerBlock)
# Ising model simulation using 1 GPU
gpu_1_block_size=10
gpu_2_block_size=5

for i in 2.0 2.1 2.2 2.3 2.4 2.5;
do
	rm Input
	echo -e "1 1\n0\n12345\n200\n5000\n1000\n10\n100\n10\n$i\n0.0\n1\n$gpu_1_block_size $((2*gpu_1_block_size))" >> Input
	echo "2D Ising model using 1 GPU, block size is $gpu_1_block_size, temperature is $i, the input is:"
	cat Input
	mkdir -p result/GPU_1/T_${i}/
	./ising2d_ngpu_gmem_v2 < Input > result/GPU_1/T_${i}/Output
	mv ./ising2d_ngpu_gmem.dat result/GPU_1/T_${i}/
	mv ./spin_ngpu_gmem.dat result/GPU_1/T_${i}/
done

# Experiment different block size(threadsPerBlock)
# Ising model simulation using 2 GPU
for i in 2.0 2.1 2.2 2.3 2.4 2.5;
do
	rm Input
	echo -e "2 1\n0\n1\n12345\n200\n5000\n1000\n10\n100\n10\n$i\n0.0\n1\n$gpu_2_block_size $((2*gpu_2_block_size))" >> Input
	echo "2D Ising model using 2 GPU, block size is $gpu_2_block_size, temperature is $i, the input is:"
	cat Input
	mkdir -p result/GPU_2/T_${i}/
	./ising2d_ngpu_gmem_v2 < Input > result/GPU_2/T_${i}/Output
	mv ./ising2d_ngpu_gmem.dat result/GPU_2/T_${i}/
	mv ./spin_ngpu_gmem.dat result/GPU_2/T_${i}/
done