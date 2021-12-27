mkdir -p result
for i in 1 2 4 8 16 32 64 128 256 512;
do
	rm Input
	echo -e "0\n$i\n1\n" >> Input
	echo "Solving poisson equation using lattice size=$i, the input is:"
	cat Input
	mkdir -p result/Lattice_${i}/
	./poisson3d_fft < Input > result/Lattice_${i}/Output
	mv poissonSystem.dat result/Lattice_${i}/
	mv poissonSystemDiagonal.dat result/Lattice_${i}/
	mv poissonSystemXAxis.dat result/Lattice_${i}/
done
