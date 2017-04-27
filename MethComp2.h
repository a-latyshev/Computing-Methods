#ifndef METHCOMP2_H
#define METHCOMP2_H

#include <armadillo>

using namespace std;
using namespace arma;

class MethComp2{
private:
    double h;
    double delta;
    const double a = 1.0;
    const double b = 1.2;
    int N;
    double MaxEvalue;
    double MinEvalue;
    double MaxError = 0.0;
    vec nodes;

    const double f(const double &x, const double &y) const;
    const double fi(const double &x, const double &y) const;
    const double dot_matrix(mat &A, mat &B) const;
    const double find_max(mat &A) const;
    const double find_max_nevyazka(mat &A) const;

public:
    MethComp2();
    MethComp2(const double &h, const double &delta);
    //counting min and max eigen values of Laplase task
    // via Lyusternik method (for square, for "house" domains)
    void Lyusternik_m_square();
    const double Lyusternik_m_house();
    //solving diff equation via relaxation method
    void Relaxation_m_square(double &w);
    void Relaxation_m_house(double &w);
    template <typename T>
    void show(T &obj) {
        cout << endl;
        cout.setf(ios::fixed);
        cout.precision(6);
        obj.raw_print(cout);
        cout.unsetf(ios::fixed);
        cout << endl;
    }
};
#endif
