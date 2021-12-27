# set GPU Id
gpu_id=0
# set N size array
N=81920007
# Experiment different grid size(blocksPerGrid)
for i in 2 4 8 16 32 64 128 256 512;
do
	# Experiment different block size(threadsPerBlock)
	for j in 2 4 8 16 32 64 128 256 512;
	do
		echo -e "$gpu_id\n$N\n$i $j" >> Input
		echo "Current grid size is $i block size is $j, the input is:"
		cat Input
		./findMin < Input > Output_${i}_${j}
		rm Input
	done
done
