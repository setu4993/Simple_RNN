#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <math.h>
#include <stdlib.h>

using namespace std;

#define input_nodes 3
#define layers 3
#define output_nodes 1
#define hidden_layer_nodes 8
#define recursion_number 12

double wih[recursion_number][input_nodes + 1][hidden_layer_nodes];	//weight matrix for (input + bias) -> hidden layer
double who[recursion_number][hidden_layer_nodes + 1][output_nodes];	//weight matrix for (hidden layer + bias) -> output
double whh[recursion_number][hidden_layer_nodes][hidden_layer_nodes];
double inp[input_nodes + 1];	//input layer node values + bias
int tmp[input_nodes];
double hid[hidden_layer_nodes + 1];	//hidden layer node values + bias
double prev_hid[recursion_number][hidden_layer_nodes];	//previous hidden layer node values
double tv[output_nodes];	// target value
double op[output_nodes]; //output values
double alpha = 0.43;	//learning rate
						//variables for the error back propagation
double deltaho[output_nodes];
double deltah[hidden_layer_nodes];
double deltahh[hidden_layer_nodes];
double delWho[hidden_layer_nodes + 1][output_nodes];
double delin[hidden_layer_nodes];
double delWih[input_nodes + 1][hidden_layer_nodes];
long maxEpoch = 200000; //max epochs
int error_rec[2][recursion_number];

int cur_count = 1;
int tst = 1;
double crr = 0, err = 0;
int curr_rec = 0;

double mse = 0;
double initial_weight = 1;

ifstream file("water_6_11-15_reduced_2_cap_rs.csv");
ifstream weights("100k_8_hidden_daily_rec_022_parallel_rs.txt");
ofstream outfile;
ofstream errfile;
string value;

void create_network();
void clear_values();
void run_network();
void recal_weights();
void next_iter();
void printweights();
void printinputs();
void writeweights();
void set_weights();
void write_errfile();
void test_network();

int main()
{
	int j = 0;
	long epoch = 0;
	curr_rec = 15;
	set_weights();
	printweights();
	clear_values();
	for (int i = 0; i < recursion_number; i++)
	{
		error_rec[0][i] = 0;
	}
	for (int i = 0; i < recursion_number; i++)
	{
		error_rec[1][i] = 0;
	}
	errfile.open("error.csv");
	while (!file.eof())
	{
		getline(file, value);
		next_iter();
		run_network();
		test_network();
		write_errfile();
		recal_weights();
		clear_values();
	}
	file.clear();
	file.seekg(0, file.beg);
	file.close();
	outfile.open("100k_8_hidden_daily_retrained.txt");
	cout << "\n\nCorrect: " << crr << "\n\nError: " << err;
	cout << "\n\n\nCorrect and errors for each recursion: ";
	for (int i = 0; i < recursion_number; i++)
	{
		cout << "\n" << error_rec[0][i] << "\t" << error_rec[1][i];
	}
	writeweights();
	errfile.close();
	outfile.close();
	getchar();
	return 0;
}

void create_network()
{
	int i, j, k;
	double tmp;
	//initializing all weights randomly
	for (k = 0; k < recursion_number; k++)
	{

		for (i = 0; i <= input_nodes; i++)
		{
			for (j = 0; j < hidden_layer_nodes; j++)
			{
				tmp = rand() % 2000000 - 1000000;
				wih[k][i][j] = tmp / 1000000;
			}
		}
	}
	for (k = 0; k < recursion_number; k++)
	{
		for (i = 0; i <= hidden_layer_nodes; i++)
		{
			for (j = 0; j < output_nodes; j++)
			{
				tmp = rand() % 2000000 - 1000000;
				who[k][i][j] = tmp / 1000000;
			}
		}
	}
	for (k = 0; k < recursion_number; k++)
	{
		for (i = 0; i < hidden_layer_nodes; i++)
		{
			for (j = 0; j < hidden_layer_nodes; j++)
			{
				tmp = rand() % 2000000 - 1000000;
				whh[k][i][j] = tmp / 1000000;
			}
			prev_hid[curr_rec][i] = 0;
		}
	}
}

void clear_values()
{
	int i = 0, j = 0;
	for (i = 0; i < input_nodes; i++)
	{
		inp[i] = 0;
	}
	inp[i] = 1;	//bias

	for (i = 0; i < hidden_layer_nodes; i++)
	{
		hid[i] = 0;
		deltah[j] = 0;
		delin[i] = 0;
	}
	hid[i] = 1;	//bias

				//clear all output and target values
	for (i = 0; i < output_nodes; i++)
	{
		op[i] = 0;
		tv[i] = 0;
		deltaho[i] = 0;
	}
}

void next_iter()
{
	int i = 0;
	int temp = 0;
	char seps[] = ",";
	char *token;

	token = strtok(&value[0], seps);
	while (token != NULL)
	{
		tmp[i] = atof(token);
		switch (i)
		{
		case 0:
			inp[i] = tmp[i] - 92;
			inp[i] = tanh(inp[i] / 46);
			break;
		case 1:
			inp[i] = tmp[i] - 100;
			inp[i] = tanh(inp[i] / 100);
			break;
		case 2:
			inp[i] = tanh((tmp[i] - 126) / 63);
			break;
		case 5:
			tv[0] = atof(token) - 140;
			tv[0] = tanh(tv[0] / 20);
			break;
		case 6:
			curr_rec = tmp[i] - 1;
		default: break;
		}

		token = strtok(NULL, ",");
		i++;
	}

	for (i = 0; i < hidden_layer_nodes; i++)
	{
		hid[i] = 0;
	}

	hid[hidden_layer_nodes] = 1;	//bias	
}


void run_network()
{
	int i, j;

	//calculation of hidden layer

	//cout << "\n";

	for (j = 0; j < hidden_layer_nodes; j++)
	{
		for (i = 0; i <= input_nodes; i++)
		{
			hid[j] = hid[j] + (wih[curr_rec][i][j] * inp[i]);
		}
	}

	for (j = 0; j < hidden_layer_nodes; j++)
	{
		for (i = 0; i < hidden_layer_nodes; i++)
		{
			hid[j] = hid[j] + (whh[curr_rec][i][j] * prev_hid[curr_rec][i]);
		}
	}

	for (j = 0; j < hidden_layer_nodes; j++)
	{
		//cout << "\n hid " << hid[j];
		//hid[j] = 1.0000 / (1.0000 + exp(-1 * hid[j]));	//sigmoid
		//hid[j] = hid[j] / (1.0 + abs(hid[j]));	//softsign
		//cout << "\n hid " << hid[j];
		/**/

		hid[j] = tanh((hid[j]));
	}


	//calculation of output layer
	for (j = 0; j <= hidden_layer_nodes; j++)
	{
		for (i = 0; i < output_nodes; i++)
		{
			op[i] = op[i] + (who[curr_rec][j][i] * hid[j]);
		}

	}

	for (j = 0; j < output_nodes; j++)
	{
		//op[j] = op[j] / (1.0 + abs(op[j]));	//softsign at the output
		//op[j] = 1.0000 / (1.0000 + exp(-1 * op[j]));	//sigmoid at the output

		op[j] = tanh((op[j]));

		/*
		if (op[j] < 0.5)
		{
		op[j] = 0;
		}
		else
		{
		op[j] = 1;
		}
		*/
	}
}

void recal_weights()
{
	int i = 0, j = 0;

	for (i = 0; i < output_nodes; i++)
	{
		deltaho[i] = (tv[i] - op[i])*(1 - (tanh((op[j])) * tanh((op[j])))); //tanh

																			//deltaho[i] = (tv[i] - op[i])*exp(op[i])/((1 + exp(op[i]))*(1 + exp(op[i])));	//for sigmoid
																			//deltaho[i] = (tv[i] - op[i]) / ((1 + abs(op[i]))*(1 + abs(op[i])));	//for softsign function
	}

	for (i = 0; i <= hidden_layer_nodes; i++)	//including bias, calculation of delta h -> o
	{
		for (j = 0; j < output_nodes; j++)
		{
			delWho[i][j] = deltaho[j] * hid[i] * alpha;
		}
	}

	//calculation of delin h -> o
	for (i = 0; i < hidden_layer_nodes; i++)
	{
		for (j = 0; j < output_nodes; j++)
		{
			delin[i] += deltaho[j] * who[curr_rec][i][j];
		}
	}

	//calculation of delih i -> h, incl bias

	for (j = 0; j < hidden_layer_nodes; j++)
	{

		deltah[j] = delin[j] * (1 - (tanh((hid[j])) * tanh((hid[j]))));	//tanh
																		//deltah[j] = delin[j] * exp(hid[j])/((1 + exp(hid[j])) * (1 + exp(hid[j])));	//sigmoid
																		//deltah[j] = delin[j] / ((1 + abs(hid[j])) * (1 + abs(hid[j])));	//softsign
	}
	for (i = 0; i <= input_nodes; i++)
	{
		for (j = 0; j < hidden_layer_nodes; j++)
		{
			wih[curr_rec][i][j] = wih[curr_rec][i][j] + (alpha * deltah[j] * inp[i]);
		}
	}

	for (i = 0; i < hidden_layer_nodes; i++)
	{
		for (j = 0; j < hidden_layer_nodes; j++)
		{
			whh[curr_rec][i][j] = whh[curr_rec][i][j] + (alpha * deltah[j] * prev_hid[curr_rec][i]);
		}

		prev_hid[curr_rec][i] = hid[i];
	}

	//weight updation, i -> h
	/*for (i = 0; i <= input_nodes; i++)
	{
	for (j = 0; j < hidden_layer_nodes; j++)
	{
	}
	}*/

	//weight updation, h -> o
	for (i = 0; i <= hidden_layer_nodes; i++)
	{
		for (j = 0; j < output_nodes; j++)
		{
			who[curr_rec][i][j] = who[curr_rec][i][j] + delWho[i][j];
		}
	}
}

void printweights()
{
	int i, j, k;
	for (k = 0; k < recursion_number; k++)
	{
		cout << "\nRecursion number: " << k;
		cout << "\nI -> H\n";
		for (i = 0; i <= input_nodes; i++)
		{
			for (j = 0; j < hidden_layer_nodes; j++)
			{
				cout << wih[k][i][j];
				if (j != (hidden_layer_nodes - 1))
				{
					cout << ",";
				}
			}
			cout << "\n";
		}

		cout << "\n\nP_H -> H\n";
		for (i = 0; i < hidden_layer_nodes; i++)
		{
			for (j = 0; j < hidden_layer_nodes; j++)
			{
				cout << whh[k][i][j];
				if (j != (hidden_layer_nodes - 1))
				{
					cout << ",";
				}
			}
			cout << "\n";
		}

		cout << "\n\nH -> O\n";
		for (i = 0; i <= hidden_layer_nodes; i++)
		{
			for (j = 0; j < output_nodes; j++)
			{
				cout << who[k][i][j];
				if (j != (output_nodes - 1))
				{
					cout << ",";
				}
			}
			cout << "\n";
		}

		cout << "\n\nH\n";
		for (i = 0; i < hidden_layer_nodes; i++)
		{
			cout << prev_hid[k][i] << "\n";
		}
	}
}

void writeweights()
{
	int i, j, k;
	for (k = 0; k < recursion_number; k++)
	{
		outfile << k << ",";
		//outfile << "\nI -> H\n";
		for (i = 0; i <= input_nodes; i++)
		{
			for (j = 0; j < hidden_layer_nodes; j++)
			{
				outfile << wih[k][i][j];
				if (j != (hidden_layer_nodes - 1))
				{
					outfile << ",";
				}
			}
			outfile << "\n";
		}

		//outfile << "\n\nP_H -> H\n";
		for (i = 0; i < hidden_layer_nodes; i++)
		{
			for (j = 0; j < hidden_layer_nodes; j++)
			{
				outfile << whh[k][i][j];
				if (j != (hidden_layer_nodes - 1))
				{
					outfile << ",";
				}
			}
			outfile << "\n";
		}

		//outfile << "\n\nH -> O\n";
		for (i = 0; i <= hidden_layer_nodes; i++)
		{
			for (j = 0; j < output_nodes; j++)
			{
				outfile << who[k][i][j];
				if (j != (output_nodes - 1))
				{
					outfile << ",";
				}
			}
			outfile << "\n";
		}

		//outfile << "\n\nH\n";
		for (i = 0; i < hidden_layer_nodes; i++)
		{
			outfile << prev_hid[k][i] << "\n";
		}
	}
}

void set_weights()
{
	//cout<< "Hello";
	int i = 0, j = 0, k = 0, l = 0, h = 0;
	while (!weights.eof())
	{
		getline(weights, value);
		//float tmp;
		char seps[] = ",";
		char *token;

		token = strtok(&value[0], seps);
		//std::cout << "\n" << token << "\n";
		while (token != NULL)
		{
			if (curr_rec == 15)
			{
				curr_rec = atoi(token);
			}
			else if (i < (input_nodes + 1))
			{

				if (j < hidden_layer_nodes)
				{
					wih[curr_rec][i][j++] = atof(token);
					if (j == hidden_layer_nodes)
					{
						i++;
						j = 0;
					}
				}
			}
			else if (l < (hidden_layer_nodes))
			{
				if (j < hidden_layer_nodes)
				{
					whh[curr_rec][l][j++] = atof(token);
					if (j == hidden_layer_nodes)
					{
						l++;
						j = 0;
					}
				}
			}
			else if (k < (hidden_layer_nodes + 1))
			{
				if (j < output_nodes)
				{
					who[curr_rec][k][j++] = atof(token);
					if (j == output_nodes)
					{
						k++;
						j = 0;
					}
				}
			}
			else if (h < (hidden_layer_nodes))
			{
				prev_hid[curr_rec][h] = atof(token);
				h++;
				if (h == hidden_layer_nodes)
				{
					curr_rec = 15;
					i = 0; j = 0; k = 0; h = 0; l = 0;
				}
			}

			token = strtok(NULL, ",");
		}
	}
	weights.clear();
}

void write_errfile()
{
	errfile << cur_count++ << "," << tv[0] << "," << op[0] << "," << abs(tv[0] - op[0]) << "\n";
}

void printinputs()
{
	int i;
	cout << "\nInput Nodes:\n";
	for (i = 0; i <= input_nodes; i++)
	{
		cout << "\t" << inp[i];
	}
	/*
	cout << "\nHidden Nodes:\n";
	for (i = 0; i <= hidden_layer_nodes; i++)
	{
	cout << "\t" << hid[i];
	}
	*/
}

void test_network()
{
	//printinputs();
	int i = 0;
	tst = 1;
	if ((op[0] == tv[0]) || (abs(op[0] - tv[0]) <= 0.2))
	{
		tst = 1;
		error_rec[0][curr_rec]++;
	}
	else
	{
		tst = 0;
		error_rec[1][curr_rec]++;
	}

	if (tst == 1)
	{
		crr++;
	}
	else
	{
		err++;
		//printinputs();
	}
}