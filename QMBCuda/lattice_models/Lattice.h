#pragma once
#include <vector>

class Lattice
{
public:

	Lattice() {};
	Lattice(std::vector<float> R1_in, std::vector<float> R2_in);

	virtual ~Lattice() {};

	std::vector<float> GetR1();
	std::vector<float> GetR2();
	int GetLS();

	void PrintLatticePos();
	void PrintLatticePosDiff();
	void ComputeR_diffs();
	std::vector<float> GetCentralPoint();

protected:
	std::vector<float> R1;
	std::vector<float> R2;

	std::vector<std::vector<float>> R1_diff;
	std::vector<std::vector<float>> R2_diff;

	int LS;

};
