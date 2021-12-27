# set GPU Id
gpu_id=0
# set N size array
N=4

mkdir -p result

rm Input
echo -e "$gpu_id\n$N $N $N\n2 2 2\n2" >> Input
echo "Current block size is 2, the input is:"
cat Input
./laplace < Input > result/Output_3D
mkdir -p result/Laplace_3D
mv result/Output_3D result/Laplace_3D/
mv phi_initial.dat result/Laplace_3D/
mv phi_CPU.dat result/Laplace_3D/
mv phi_GPU.dat result/Laplace_3D/


for i in 8 16 32 64;
do
	rm Input
	echo -e "$gpu_id\n$i $i $i\n4 4 4\n2" >> Input
	echo "Current L is ${i}, block size is 4, the input is:"
	cat Input
	./poisson_pc < Input > result/Output_3D_pc_${i}
	mkdir -p result/Poisson_3D_${i}
	mv result/Output_3D_pc_${i} result/Poisson_3D_${i}/
	mv phi_initial.dat result/Poisson_3D_${i}/
	mv phi_CPU.dat result/Poisson_3D_${i}/
	mv phi_GPU.dat result/Poisson_3D_${i}/
done
