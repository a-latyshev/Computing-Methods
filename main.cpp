#define ARMA_USE_CXX11

#include <iostream>
#include <armadillo>
#include "MethComp2.h"

using namespace std;
using namespace arma;

int main() {
    double MinEvalue; // like this: Av = lDv , l - MinEvalue
    const double h = 0.01;
    const double delta = 0.0001;

    MethComp2 task(h, delta);
    MinEvalue = task.Lyusternik_m_house();
    task.Relaxation_m_house(MinEvalue);
    return 0;

}