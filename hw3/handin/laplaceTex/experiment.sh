# set GPU Id
gpu_id=0
# set N size array
N=512

mkdir -p result

rm Input
echo -e "$gpu_id\n$N $N\n2 2\n2" >> Input
echo "Current block size is 2, the input is:"
cat Input
./laplace < Input > result/Output_2
mkdir -p result/block_2
mv result/Output_2 result/block_2/
mv phi_initial.dat result/block_2/
mv phi_CPU.dat result/block_2/
mv phi_GPU.dat result/block_2/
./laplaceTex < Input > result/Output_Tex_2
mkdir -p result/Tex_block_2
mv result/Output_Tex_2 result/Tex_block_2/
mv phi_initial.dat result/Tex_block_2/
mv phi_CPU.dat result/Tex_block_2/
mv phi_GPU_Tex.dat result/Tex_block_2/

# Experiment different block size(threadsPerBlock)
# Not use CPU
for i in 4 8 16 32;
do
	rm Input
	echo -e "$gpu_id\n$N $N\n$i $i\n1" >> Input
	echo "Current block size is $i, the input is:"
	cat Input
	./laplace < Input > result/Output_${i}
	mkdir -p result/block_${i}
	mv result/Output_${i} result/block_${i}/
	mv phi_initial.dat result/block_${i}/
	mv phi_GPU.dat result/block_${i}/
	./laplaceTex < Input > result/Output_Tex_${i}
	mkdir -p result/Tex_block_${i}
	mv result/Output_Tex_${i} result/Tex_block_${i}/
	mv phi_initial.dat result/Tex_block_${i}/
	mv phi_GPU_Tex.dat result/Tex_block_${i}/
done
