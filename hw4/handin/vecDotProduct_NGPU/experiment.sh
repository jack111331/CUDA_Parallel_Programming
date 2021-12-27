mkdir -p result

# 1 GPU
# Experiment different block size(threadsPerBlock) and grid size(blocksPerGrid)
# Not use CPU
for i in 2 4 8 16 32 64 128 256 512;
do
	for j in 2 4 8 16 32 64 128 256;
	do
		rm Input
		echo -e "1\n0\n40960000\n$i $j" >> Input
		echo "Current block size is $i, grid size is $j, the input is:"
		cat Input
		mkdir -p result/1GPU_block_${i}_grid_${j}
		./vecDotProduct_ngpu < Input > result/1GPU_block_${i}_grid_${j}/Output_${i}_${j}
	done
done

# 2 GPU
# Experiment different block size(threadsPerBlock) and grid size(blocksPerGrid)
# Not use CPU
for i in 2 4 8 16 32 64 128 256 512;
do
	for j in 2 4 8 16 32 64 128 256;
	do
		rm Input
		echo -e "2\n0 1\n40960000\n$i $j" >> Input
		echo "Current block size is $i, grid size is $j, the input is:"
		cat Input
		mkdir -p result/2GPU_block_${i}_grid_${j}
		./vecDotProduct_ngpu < Input > result/2GPU_block_${i}_grid_${j}/Output_${i}_${j}
	done
done
