#pragma once
#include <string>

enum class OperatorType { Sz, Sp, Sm, c_up, c_dag_up, c_down, c_dag_down, b, b_dag, I};

OperatorType GetConjugateOperatorType(OperatorType orig_op);

std::string PrintOperatorType(OperatorType op);

template <typename T>
class Operator
{
public:

	Operator(int site_in, OperatorType type_in, T scalar_in);
	virtual~Operator() {};

	int GetSite();
	T GetScalar();
	OperatorType GetType();

	void SetScalar(T _scalar);
	void SetSite(int new_site);
	
protected:
	int site; // lattice site
	T scalar; // possible scalar
	OperatorType type;
};

















