
#include <vector>
#include "Lattice.h"
#include <iostream>

Lattice::Lattice(std::vector<float> R1_in, std::vector<float> R2_in)
{
	R1 = R1_in;
	R2 = R2_in;
	LS = R1.size();
	ComputeR_diffs();
}

void Lattice::ComputeR_diffs() {

	for (int i = 0; i < LS; i++) {
		R1_diff.push_back(std::vector<float>(LS));
		R2_diff.push_back(std::vector<float>(LS));
		for (int j = 0; j < LS; j++) {
			R1_diff[i][j] = (R1[i] - R1[j]);
			R2_diff[i][j] = (R2[i] - R2[j]);
		}
	}
}

std::vector<float> Lattice::GetR1()
{
	return R1;
}

std::vector<float> Lattice::GetR2()
{
	return R2;
}

int Lattice::GetLS()
{
	return LS;
}

void Lattice::PrintLatticePos()
{
	std::cout << "R1:" << std::endl;
	for (int i = 0; i < LS; i++) {
		std::cout << R1[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "R2:" << std::endl;
	for (int i = 0; i < LS; i++) {
		std::cout << R2[i] << " ";
	}

}

void Lattice::PrintLatticePosDiff()
{
	std::cout << "R1_diff:" << std::endl;
	for (int i = 0; i < LS; i++) {
		for (int j = 0; j < LS; j++) {
			std::cout << R1_diff[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "R2_diff:" << std::endl;
	for (int i = 0; i < LS; i++) {
		for (int j = 0; j < LS; j++) {
			std::cout << R2_diff[i][j] << " ";
		}
		std::cout << std::endl;
	}

}

std::vector<float> Lattice::GetCentralPoint()
{
	float Rx_mean = 0.0;
	float Ry_mean = 0.0;
	for (int ri = 0; ri < LS; ri++) {
		Rx_mean = Rx_mean + R1[ri];
		Ry_mean = Ry_mean + R2[ri];
	}
	Rx_mean = Rx_mean / (float)LS;
	Ry_mean = Ry_mean / (float)LS;

	std::vector<float> R0 = { Rx_mean, Ry_mean };
	return R0;
}