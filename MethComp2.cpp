#include <armadillo>
#include <cmath>
#include <fstream>
#include "MethComp2.h"

#define house_without_borders (i + j > N/2 && i - j < N/2)
#define house_borders (i + j == N/2 || i - j == N/2 || (i == N && j > N/2) || (i == 0 && j > N/2) || j == N)

using namespace std;
using namespace arma;

MethComp2::MethComp2() {
}

MethComp2::MethComp2(const double &h, const double &delta) {
    this->h = h;
    this->delta = delta;
    N = (int) 1.0 / h;
    nodes = zeros<vec>(N + 1);
    for(int i = 0; i < N + 1; i++){
        nodes(i) = i*h;
    }
}

const double MethComp2::f(const double &x, const double &y) const {
    return 0.2 * exp(x)*cos(y);
}

const double MethComp2::fi(const double &x, const double &y) const {
    return exp(x)*cos(y);
}

const double MethComp2::dot_matrix(mat &A, mat &B) const {
    double a = 0.0;
    for(int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a += A(i,j) * B(i,j);
        }
    }
    return a;
}

const double MethComp2::find_max(mat &A) const {
    double max = 0.0;
    for(int i = 0; i < A.n_rows; i++){
        for(int j = 0; j < A.n_cols; j++){
            if (abs(A(i,j)) > max){
                max = abs(A(i,j));
            }
        }
    }
    return max;
}

const double MethComp2::find_max_nevyazka(mat &A) const {
    double max = 0.0;
    double tmp;
    for (int i = 1; i < A.n_rows - 1; i++) {
        for (int j = 1; j < A.n_cols - 1; j++) {
            if(house_without_borders) {
                tmp = abs(f(nodes(i), nodes(j)) + (a * (A(i - 1, j) - 2 * A(i, j) + A(i + 1, j)) +
                                                   b * (A(i, j - 1) - 2 * A(i, j) + A(i, j + 1))) / (h * h));
                if (tmp > max) {
                    max = tmp;
                }
            }
        }
    }
    return max;
}

void MethComp2::Lyusternik_m_square() {
    cout.setf(ios::fixed);
    cout.precision(8);
    double prevEvalue = 0.0;
    double Evalue = 1000.0;
    double norm_Uprev;
    mat Uprev(N + 1, N + 1, fill::zeros);
    mat U(N + 1, N + 1, fill::zeros);
    int iter = 0;
    for (int i = 1; i < N; i++){
        for(int j = 1; j < N; j++){
            U(i, j) = 1.0;
        }
    }
    U(N - 1, N - 1) = 10.0;

    do {
        Uprev = U;
        for(int i = 1; i < N; i++){
            for(int j = 1; j < N; j++){
                U(i,j) = - a * (Uprev(i - 1, j) - 2 * Uprev(i, j) + Uprev(i + 1, j))
                         - b * (Uprev(i, j - 1) - 2 * Uprev(i, j) + Uprev(i, j + 1));
            }
        }
        norm_Uprev = sqrt(dot_matrix(Uprev, Uprev));
        prevEvalue = Evalue;
        U = U / norm_Uprev;
        Uprev = Uprev / norm_Uprev;
        Evalue = dot_matrix(Uprev, U);
        iter++;
    } while (abs(Evalue - prevEvalue) / prevEvalue >= delta);
    MaxEvalue = Evalue;
    cout << "Max eigen value: " << MaxEvalue / (h * h)  << " # " << iter << endl;

    //Minimum EValue counting:
    iter = 0;
    U = zeros<mat>(N + 1, N + 1);
    for (int i = 1; i < N; i++){
        for(int j = 1; j < N; j++){
            U(i, j) = 1.0;
        }
    }
    U(N - 1, N - 1) = 10.0;
    Evalue = 1000.0;
    do {
        Uprev = U;
        for(int i = 1; i < N - 1; i++){
            for(int j = 1; j < N - 1; j++){
                U(i,j) = MaxEvalue * Uprev(i,j) + a * (Uprev(i - 1, j) - 2 * Uprev(i, j) + Uprev(i + 1, j))
                         + b * (Uprev(i, j - 1) - 2 * Uprev(i, j) + Uprev(i, j + 1));
            }
        }
        norm_Uprev = sqrt(dot_matrix(Uprev, Uprev));
        prevEvalue = Evalue;
        U = U / norm_Uprev;
        Uprev = Uprev / norm_Uprev;
        Evalue = dot_matrix(Uprev, U);
        iter++;
    } while (abs(Evalue - prevEvalue) / abs(MaxEvalue - prevEvalue) >= delta);
    MinEvalue = (MaxEvalue - Evalue) / (h * h);
    cout << "Min eigen value: " << MinEvalue  << " # " << iter << endl;
    cout.unsetf(ios::fixed);
}

const double MethComp2::Lyusternik_m_house() {
    cout.setf(ios::fixed);
    cout.precision(8);
    double prevEvalue = 0.0;
    double Evalue = 1000.0;
    double norm_Uprev;
    mat Uprev(N + 1, N + 1, fill::zeros);
    mat U(N + 1, N + 1, fill::zeros);
    int iter = 0;
      for (int i = 1; i < N; i++) {
          for(int j = 1; j < N; j++) {
              if (house_without_borders) {
                  U(i, j) = 1.0;
              }
          }
      }
      U(N - 1, N - 1) = 10.0;
      do {
          Uprev = U;
          for(int i = 1; i < N; i++) {
              for(int j = 1; j < N; j++) {
                  if (house_without_borders) {
                      U(i,j) = - a * (Uprev(i - 1, j) - 2 * Uprev(i, j) + Uprev(i + 1, j))
                               - b * (Uprev(i, j - 1) - 2 * Uprev(i, j) + Uprev(i, j + 1));
                  }
              }
          }
          norm_Uprev = sqrt(dot_matrix(Uprev, Uprev));
          prevEvalue = Evalue;
          U = U / norm_Uprev;
          Uprev = Uprev / norm_Uprev;
          Evalue = dot_matrix(Uprev, U);
          iter++;
      } while (abs(Evalue - prevEvalue) / prevEvalue >= delta);
      MaxEvalue = Evalue;
      cout << "Max eigen value: " << MaxEvalue / (h * h)  << " Iterations: " << iter << endl;

      //Minimum EValue counting:
      iter = 0;
      U = zeros<mat>(N + 1, N + 1);
      for (int i = 1; i < N; i++) {
          for(int j = 1; j < N; j++) {
              if (house_without_borders) {
                  U(i, j) = 1.0;
              }
          }
      }
      U(N - 1, N - 1) = 10.0;
      Evalue = 1000.0;
      do {
          Uprev = U;
          for(int i = 1; i < N; i++) {
              for(int j = 1; j < N; j++) {
                  if (house_without_borders) {
                      U(i,j) = MaxEvalue * Uprev(i,j) + a * (Uprev(i - 1, j) - 2 * Uprev(i, j) + Uprev(i + 1, j))
                                                      + b * (Uprev(i, j - 1) - 2 * Uprev(i, j) + Uprev(i, j + 1));
                  }
              }
          }
          norm_Uprev = sqrt(dot_matrix(Uprev, Uprev));
          prevEvalue = Evalue;
          U = U / norm_Uprev;
          Uprev = Uprev / norm_Uprev;
          Evalue = dot_matrix(Uprev, U);
          iter++;
      } while (abs(prevEvalue - Evalue) / abs(MaxEvalue - prevEvalue) >= delta);
      MinEvalue = (MaxEvalue - Evalue) / (h * h);
      cout << "Min eigen value: " << MinEvalue  << " Iterations: " << iter << endl;
      cout.unsetf(ios::fixed);
    return MinEvalue;
}

void MethComp2::Relaxation_m_square(double &w) {
    mat U(N + 1, N + 1);
    int iter = 0;
    double tmp;
    U = zeros(N + 1, N + 1);
    for (int i = 0; i < N + 1; i++){
        U(i, 0) = fi(nodes(i), nodes(0));
        U(0, i) = fi(nodes(0), nodes(i));
        U(i, N) = fi(nodes(i), nodes(N));
        U(N, i) = fi(nodes(N), nodes(i));
    }

    do {
        for(int i = 1; i < N; i++){
            for(int j = 1; j < N; j++){
                U(i, j) = (1.0 - w)*U(i, j) +
                          w*(h*h*f(nodes(i), nodes(j)) + a * (U(i - 1, j) + U(i + 1, j)) + b * (U(i, j - 1) + U(i, j + 1)))/(2.0 * (a + b));
            }
        }
        iter++;
    } while (find_max_nevyazka(U) > delta);

    MaxError = 0.0;
    for(int i = 1; i < N; i++){
        for(int j = 1; j < N; j++){
            if(abs(fi(nodes(i), nodes(j)) - U(i, j)) > MaxError) {
                MaxError = abs(fi(nodes(i), nodes(j)) - U(i, j));
            }
        }
    }
    cout.setf(ios::fixed);
    cout.precision(16);
    cout << "Maximum Error: " << MaxError << endl;
    cout << "Nevyazka: " << find_max_nevyazka(U) << endl;
    cout.unsetf(ios::fixed);
    cout << "Iterations: " << iter << endl;

    ofstream fout("/home/andrey/Downloads/MethComp2/forGnuplot.txt");
    fout.precision(6);
    fout.setf(ios::fixed);
    for(int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            fout << "\t" << nodes(i) << "\t" << nodes(j) << "\t" << U(i,j) << endl;
        }
        fout << "\n";
    }
    fout.unsetf(ios::fixed);
    fout.close();
}

void MethComp2::Relaxation_m_house(double &w) {
    cout.setf(ios::fixed);
    cout.precision(16);
    //Optimal parametr for the iterative proccese (Samarskii, Nikolaev)
    w = 4*(a + b)/(2*(a + b) + h*sqrt(w*((4*(a + b) - h*h*w))));
    cout << "Omega Optimal: " << w << endl;
    mat U(N + 1, N + 1);
    int iter = 0;
    U = zeros(N + 1, N + 1);
    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            if (house_borders) {
                U(i, j) = fi(nodes(i), nodes(j));
            }
        }
    }

    do {
        for(int j = 1; j < N; j++){
            for(int i = 1; i < N; i++){
                if (house_without_borders) {
                    U(i, j) = (1.0 - w)*U(i, j) +
                              w*(h * h * f(nodes(i), nodes(j)) + a * (U(i - 1, j) + U(i + 1, j)) + b * (U(i, j - 1) + U(i, j + 1))) / (2.0 * (a + b));
                }
            }
        }
        iter++;
    } while (find_max_nevyazka(U) > delta);

    MaxError = 0.0;
    for(int i = 1; i < N; i++) {
        for(int j = 1; j < N; j++) {
                if(house_without_borders) {
                    if(abs(fi(nodes(i), nodes(j)) - U(i, j)) > MaxError) {
                        MaxError = abs(fi(nodes(i), nodes(j)) - U(i, j));
                }
            }
        }
    }
    cout << "Maximum Error: " << MaxError << endl;
    cout << "Nevyazka: " << find_max_nevyazka(U) << endl;
    cout.unsetf(ios::fixed);
    cout << "Iterations: " << iter << endl;

    ofstream fout("/home/andrey/Downloads/MethComp2/forGnuplot.txt");
    fout.precision(6);
    fout.setf(ios::fixed);
    for(int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            if (house_without_borders) {
                fout << "\t" << nodes(i) << "\t" << nodes(j) << "\t" << U(i,j) << endl;
            }
        }
        fout << "\n";
    }
    fout.unsetf(ios::fixed);
    fout.close();
}