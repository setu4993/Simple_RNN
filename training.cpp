#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <math.h>
#include <stdlib.h>

using namespace std;

#define input_nodes 14
#define layers 3
#define output_nodes 2
#define hidden_layer_nodes 5

double wih[input_nodes + 1][hidden_layer_nodes];	//weight matrix for (input + bias) -> hidden layer
double who[hidden_layer_nodes + 1][output_nodes];	//weight matrix for (hidden layer + bias) -> output
double inp[input_nodes + 1];	//input layer node values + bias
int tmp[input_nodes];
double hid[hidden_layer_nodes + 1];	//hidden layer node values + bias
double tv[output_nodes];	// target value
double op[output_nodes]; //output values
double alpha = 0.1;	//learning rate
double deltaho[output_nodes];
double deltah[hidden_layer_nodes];
double delWho[hidden_layer_nodes + 1][output_nodes];
double delin[hidden_layer_nodes];
double delWih[input_nodes + 1][hidden_layer_nodes];
long maxEpoch = 100000;

double mse = 0;
double initial_weight = 1;

ifstream file("Training9.csv");
string value;

void create_network();
void clear_values();
void run_network();
void recal_weights();
void next_iter();
void printweights();
void printinputs();

int main()
{
	int j = 0;
	long epoch = 0;
	create_network();
	printweights();
	clear_values();
	//printinputs();
	//file.open("test4.csv", std::ifstream::in);
	for (epoch = 0; epoch < maxEpoch; epoch++)
	{
		/* file.open("test.csv", ifstream::in);
		file.get(); */
		//file.open("test3.csv", std::ifstream::in);


		while (!file.eof())
		{
			getline(file, value);
			next_iter();
			//printweights();
			//printinputs();

			//cout << "\nEpoch " << epoch << " running.";

			run_network();
			/*
			if (epoch == (maxEpoch - 1))
			{
				for (j = 0; j < output_nodes; j++)
				{
					cout << "\n op " << j << "=  " << op[j];
					cout << "\n tv " << j << "=  " << tv[j];
				}
			}
			*/
			recal_weights();
			//printweights();
			clear_values();
			//printinputs();
		}
		file.clear();
		file.seekg(0, file.beg);

		//epoch++;
		
		if ((epoch + 1) % 10000 == 0)
		{
			alpha = 0.9 * alpha;
			cout << "\nEpoch " << (epoch + 1) << " completed.";
			printweights();
		
		}
		
		//file.open("test4.csv", std::ifstream::in);

	}
	file.close();
	getchar();
	return 0;
}

void create_network()
{
	int i, j;
	double tmp;
	//initializing all weights randomly

	for (i = 0; i <= input_nodes; i++)
	{
		for (j = 0; j < hidden_layer_nodes; j++)
		{
			tmp = rand() % 2000000 - 1000000;
			wih[i][j] = tmp / 1000000;
			/*
			if ((i + j) % 2)
			{
			wih[i][j] = (double)(i + j)*(-1) / (input_nodes + hidden_layer_nodes);
			}
			else
			{
			wih[i][j] = (double)(i + j) / (input_nodes + hidden_layer_nodes);
			}
			*/
		}
	}

	for (i = 0; i <= hidden_layer_nodes; i++)
	{
		for (j = 0; j < output_nodes; j++)
		{
			tmp = rand() % 2000000 - 1000000;
			who[i][j] = tmp / 1000000;
		}
	}

	//clear();
}

void clear_values()
{
	//clear all input and hidden node values (biases as 1)
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
	//std::cout << "\n" << token << "\n";
	while (token != NULL)
	{
		tmp[i] = atof(token);
		switch (i)
		{
		case 0:
			inp[i] = tmp[i] - 15; // 1 / (double)(1 + exp(-0.18 * (tmp[i] - 15)));	//date
							 //inp[i] = tmp[i] / (double)(1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i]/8)))) - 1;
			break;
		case 1:
			inp[i] = tmp[i] - 6; // / (1 + exp(-0.18 * (tmp[i] - 15)));	//month
							 //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 3)))) - 1;
			break;
		case 2:
			inp[i] = tmp[i] - 3.5; // / (1 + exp(-0.18 * (tmp[i] - 15)));	//day of the week
							 //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 2)))) - 1;
			break;
		case 3:
			inp[i] = tmp[i] - 12; // / (1 + exp(-0.18 * (tmp[i] - 15)));	//hour
							 //inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 6)))) - 1;
			break;
		case 4:
			inp[i] = tmp[i] - 30;// / (1 + exp(-0.18 * (tmp[i] - 15)));	//minutes
							//inp[i] = tmp[i] / (1.0 + abs(tmp[i]));
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 15)))) - 1;
			break;
		case 5:
			inp[i] = atof(token) - 44.00; //latitude
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 13)))) - 1;
			break;
		case 6:
			inp[i] = atof(token) + 115.00; //longitude
			inp[i] = (2 / (double)(1 + exp(-2 * (inp[i] / 24)))) - 1;
			break;
		case 8:
			switch (tmp[i])
			{
			case 1:
				//inp[7] = 1; //daylight
				inp[7] = (rand() % 25 + 75)/ (double) 100;
				inp[8] = (rand() % 100 - 100) / (double)100;
				break;
			case 2:
			case 6:
				inp[7] = (rand() % 100 - 100) / (double)100;
				//inp[8] = 1; //dark
				inp[8] = (rand() % 25 + 75)/ (double) 100;
				break;
			case 3:
				inp[7] = (rand() % 25)/ (double) 100;
				inp[8] = (rand() % 25 + 50)/ (double) 100;
				//inp[7] = 0.25;
				//inp[8] = 0.75;
				break;
			case 4:
			case 5:
				inp[7] = (rand() % 25 + 25)/ (double) 100;
				inp[8] = (rand() % 25 + 25)/ (double) 100;
				//inp[7] = 0.5;
				//inp[8] = 0.5;
				break;
			default:
				break;
			}
			break;
		case 9: switch (tmp[i])
		{
		case 1:
			//inp[9] = 1;	//clear sky
			inp[9] = (rand() % 25 + 75)/ (double) 100;
			inp[10] = (rand() % 100 - 100)/ (double) 100;
			inp[11] = (rand() % 100 - 100) / (double)100;
			inp[12] = (rand() % 100 - 100)/ (double) 100;
			inp[13] = (rand() % 100 - 100) / (double)100;
			break;
		case 2:
			//inp[10] = 1;	//rain
			inp[10] = (rand() % 25 + 75)/ (double) 100;
			inp[9] = (rand() % 100 - 100)/ (double) 100;
			inp[11] = (rand() % 100 - 100)/ (double) 100;
			inp[12] = (rand() % 100 - 100)/ (double) 100;
			inp[13] = (rand() % 100 - 100)/ (double) 100;
			break;
		case 3:
			//inp[11] = 1;	//Sleet / Hail
			inp[11] = (rand() % 25 + 75)/ (double) 100;
			inp[10] = (rand() % 100 - 100)/ (double) 100;
			inp[9] = (rand() % 100 - 100)/ (double) 100;
			inp[12] = (rand() % 100 - 100)/ (double) 100;
			inp[13] = (rand() % 100 - 100)/ (double) 100;
			break;
		case 12:
			//inp[10] = 0.5;
			//inp[11] = 0.5;	//freezing rain or drizzle
			inp[10] = (rand() % 25 + 25)/ (double) 100;
			inp[11] = (rand() % 25 + 25)/ (double) 100;
			inp[9] = (rand() % 100 - 100)/ (double) 100;
			inp[12] = (rand() % 100 - 100)/ (double) 100;
			inp[13] = (rand() % 100 - 100)/ (double) 100;
			break;
		case 4:
			//inp[11] = 0.5;	//snow
			inp[11] = (rand() % 25 + 25)/ (double) 100;
			inp[10] = (rand() % 100 - 100)/ (double) 100;
			inp[9] = (rand() % 100 - 100)/ (double) 100;
			inp[12] = (rand() % 100 - 100)/ (double) 100;
			inp[13] = (rand() % 100 - 100)/ (double) 100;
			break;
		case 5:
			//inp[12] = 1;	//fog, smog, smoke
			inp[12] = (rand() % 25 + 75)/ (double) 100;
			inp[10] = (rand() % 100 - 100)/ (double) 100;
			inp[11] = (rand() % 100 - 100)/ (double) 100;
			inp[9] = (rand() % 100 - 100)/ (double) 100;
			inp[13] = (rand() % 100 - 100)/ (double) 100;
			break;
		case 10:
			//inp[12] = 0.5;	//cloudy
			inp[12] = (rand() % 25 + 25)/ (double) 100;
			inp[10] = (rand() % 100 - 100)/ (double) 100;
			inp[11] = (rand() % 100 - 100)/ (double) 100;
			inp[9] = (rand() % 100 - 100)/ (double) 100;
			inp[13] = (rand() % 100 - 100)/ (double) 100;
			break;
		case 6:
		case 7:
		case 11:
			//inp[13] = 1;	//Severe Crosswinds / Blowing sand, soil, dirt / Blowing snow
			inp[13] = (rand() % 25 + 75)/ (double) 100;
			inp[10] = (rand() % 100 - 100)/ (double) 100;
			inp[11] = (rand() % 100 - 100)/ (double) 100;
			inp[12] = (rand() % 100 - 100)/ (double) 100;
			inp[9] = (rand() % 100 - 100)/ (double) 100;
			break;
		}
				break;

		case 10:
			if (atof(token) < 0.5)
			{
				tv[0] = 1;	//low
			}
			else
			{
				tv[1] = 1; //high
			}
			//tv = atof(token);
			break;
		default: break;
		}

		token = strtok(NULL, ",");
		i++;
	}
	/*
	for (i = 0; i < input_nodes; i++)
	{
		if (inp[i] == 0)
		{
			temp = rand() % 100 - 100;
			inp[i] = (double)temp / 100;
		}
		else if (inp[i] == 1)
		{
			temp = rand() % 100;
			inp[i] = (double)temp / 100;
		}
	}
	*/
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
			hid[j] = hid[j] + (wih[i][j] * inp[i]);
		}

	}
	for (j = 0; j < hidden_layer_nodes; j++)
	{
		//cout << "\n hid " << hid[j];
		//hid[j] = 1.0000 / (1.0000 + exp(-1 * hid[j]));	//sigmoid
		//hid[j] = hid[j] / (1.0 + abs(hid[j]));	//softsign
												//cout << "\n hid " << hid[j];
		hid[j] = (2.0000 / (1.0000 + exp(-2 * hid[j] / 10))) - 1;
	}


	//calculation of output layer
	for (j = 0; j <= hidden_layer_nodes; j++)
	{
		for (i = 0; i < output_nodes; i++)
		{
			op[i] = op[i] + (who[j][i] * hid[j]);
		}

	}

	for (j = 0; j < output_nodes; j++)
	{
		//op[j] = op[j] / (1.0 + abs(op[j]));	//softsign at the output
		//op[j] = 1.0000 / (1.0000 + exp(-1 * op[j]));	//sigmoid at the output
		
		op[j] = (2.0000 / (1.0000 + exp(-2 * op[j] * 10))) - 1;
		
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
		deltaho[i] = (tv[i] - op[i])*(1 - (((2.0000 / (1.0000 + exp(-2 * op[i] * 10))) - 1)*((2.0000 / (1.0000 + exp(-2 * op[i] * 10))) - 1))); //tanh
		
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

	//calculation of delin i -> h
	for (i = 0; i < hidden_layer_nodes; i++)
	{
		for (j = 0; j < output_nodes; j++)
		{
			delin[i] += deltaho[j] * who[i][j];
		}
	}

	//calculation of delih i -> h, incl bias

	for (j = 0; j < hidden_layer_nodes; j++)
	{
		
		deltah[j] = delin[j] * (1 - (((2.0000 / (1.0000 + exp(-2 * hid[j] / 10))) - 1)*((2.0000 / (1.0000 + exp(-2 * hid[j] / 10))) - 1)));	//tanh
		//deltah[j] = delin[j] * exp(hid[j])/((1 + exp(hid[j])) * (1 + exp(hid[j])));	//sigmoid
		//deltah[j] = delin[j] / ((1 + abs(hid[j])) * (1 + abs(hid[j])));	//softsign
	}

	//delta input, i -> h
	for (i = 0; i <= input_nodes; i++)
	{
		for (j = 0; j < hidden_layer_nodes; j++)
		{
			//delWih[i][j] = alpha * deltah[j] * inp[i];
			wih[i][j] = wih[i][j] + (alpha * deltah[j] * inp[i]);
		}
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
			who[i][j] = who[i][j] + delWho[i][j];
		}
	}
}

void printweights()
{
	int i, j;

	cout << "\nI -> H\n";
	for (i = 0; i <= input_nodes; i++)
	{
		for (j = 0; j < hidden_layer_nodes; j++)
		{
			cout << wih[i][j];
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
			cout << who[i][j];
			if (j != (output_nodes - 1))
			{
				cout << ",";
			}
		}
		cout << "\n";
	}

}

void printinputs()
{
	int i;
	cout << "\nInput Nodes:\n";
	for (i = 0; i <= input_nodes; i++)
	{
		cout << "\t" << inp[i];
	}
	cout << "\nHidden Nodes:\n";
	for (i = 0; i <= hidden_layer_nodes; i++)
	{
		cout << "\t" << hid[i];
	}
}