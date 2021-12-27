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
for i in 1 2 5 10;
do
	rm Input
	echo -e "1 1\n0\n12345\n200\n5000\n1000\n10\n100\n10\n2.269\n0.0\n1\n$i $((2*i))" >> Input
	echo "2D Ising model using 1 GPU, current block size is $i, the input is:"
	cat Input
	mkdir -p result/GPU_1/block_${i}/
	./ising2d_ngpu_gmem_v2 < Input > result/GPU_1/block_${i}/Output
	mv ising2d_ngpu_gmem.dat result/GPU_1/block_${i}/
	mv spin_ngpu_gmem.dat result/GPU_1/block_${i}/
done

# Experiment different block size(threadsPerBlock)
# Simple sampling using 2 GPU
for i in 1 5;
do
	rm Input
	echo -e "2 1\n0\n1\n12345\n200\n5000\n1000\n10\n100\n10\n2.269\n0.0\n1\n$i $((2*i))" >> Input
	echo "2D Ising model using 2 GPU, current block size is $i, the input is:"
	cat Input
	mkdir -p result/GPU_2/block_${i}/
	./ising2d_ngpu_gmem_v2 < Input > result/GPU_2/block_${i}/Output
	mv ising2d_ngpu_gmem.dat result/GPU_2/block_${i}/
	mv spin_ngpu_gmem.dat result/GPU_2/block_${i}/
done

