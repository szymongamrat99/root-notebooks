import ROOT
######################## DOUBLE GAUSSIAN WITH A SINGLE MEAN FOR FITTING ########################

ROOT.gInterpreter.Declare("""
#include <cmath>
#include <vector>
#include <Math/Vector4D.h>
#include <ROOT/RVec.hxx>

// ==========================================
// 1. Funkcje fitowania (Double Gaussian)
// ==========================================
double double_gaussian_cpp(double *x, double *par) {
    double xx = x[0];
    double amp = par[0];
    double mean = par[1];
    double sigma1 = par[2];
    double sigma2 = par[3];
    double fraction = par[4];
    
    if (sigma1 <= 0 || sigma2 <= 0) return 1e9;
    
    double gauss1 = std::exp(-0.5 * std::pow((xx - mean) / sigma1, 2)) / (sigma1 * std::sqrt(2 * M_PI));
    double gauss2 = std::exp(-0.5 * std::pow((xx - mean) / sigma2, 2)) / (sigma2 * std::sqrt(2 * M_PI));
    
    return amp * (fraction * gauss1 + (1 - fraction) * gauss2);
}

double single_gauss_comp1(double *x, double *par) {
    double xx = x[0];
    double amp = par[0];
    double mean = par[1];
    double sigma = par[2];
    double fraction = par[3];
    if (sigma <= 0) return 0;
    double g = std::exp(-0.5 * std::pow((xx - mean) / sigma, 2)) / (sigma * std::sqrt(2 * M_PI));
    return amp * fraction * g;
}

double single_gauss_comp2(double *x, double *par) {
    double xx = x[0];
    double amp = par[0];
    double mean = par[1];
    double sigma = par[2];
    double fraction = par[3];
    if (sigma <= 0) return 0;
    double g = std::exp(-0.5 * std::pow((xx - mean) / sigma, 2)) / (sigma * std::sqrt(2 * M_PI));
    return amp * (1.0 - fraction) * g;
}

// Funkcja Crystal Ball (5 parametrów: N, mean, sigma, alpha, n)
// par[0]: Normalizacja (Amplitude / Area)
// par[1]: Średnia (Mean)
// par[2]: Sigma (Resolution)
// par[3]: Alpha (Punkt przejścia Gaussa w ogon potęgowy)
// par[4]: n (Wykładnik potęgi ogona)
double crystal_ball_cpp(double *x, double *par) {
    double xx = x[0];
    double norm  = par[0];
    double mean  = par[1];
    double sigma = par[2];
    double alpha = par[3];
    double n     = par[4];

    if (sigma <= 0) return 1e9;

    double z = (xx - mean) / sigma;
    if (alpha < 0) z = -z;

    double abs_alpha = std::abs(alpha);

    if (z >= -abs_alpha) {
        return norm * std::exp(-0.5 * z * z);
    } else {
        double a = std::pow(n / abs_alpha, n) * std::exp(-0.5 * abs_alpha * abs_alpha);
        double b = n / abs_alpha - abs_alpha;
        return norm * a * std::pow(b - z, -n);
    }
}

// ==========================================
// 2. Obliczanie masy naładowanych kaonów
// ==========================================
double calc_m_K_charged(double px1, double py1, double pz1, double m1,
                        double px2, double py2, double pz2, double m2) {
    double E1 = std::sqrt(px1*px1 + py1*py1 + pz1*pz1 + m1*m1);
    double E2 = std::sqrt(px2*px2 + py2*py2 + pz2*pz2 + m2*m2);
    
    ROOT::Math::PxPyPzEVector p4_1(px1, py1, pz1, E1);
    ROOT::Math::PxPyPzEVector p4_2(px2, py2, pz2, E2);
    
    return (p4_1 + p4_2).M();
}

// ==========================================
// 3. Funkcje pomocnicze (promienie i współrzędne)
// ==========================================
double calc_spherical_radius(const ROOT::RVec<double>& k, const ROOT::RVec<double>& ip) {
    if (k.size() > 8 && ip.size() > 2) {
        return sqrt(pow(k[6] - ip[0], 2) + pow(k[7] - ip[1], 2) + pow(k[8] - ip[2], 2));
    }
    return -999.0;
}
    
double calc_cylindrical_radius(const ROOT::RVec<double>& k, const ROOT::RVec<double>& ip) {
    if (k.size() > 7 && ip.size() > 1) {
        return hypot(k[6] - ip[0], k[7] - ip[1]);
    }
    return -999.0;
}
    
double calc_z_component(const ROOT::RVec<double>& k, const ROOT::RVec<double>& ip) {
    if (k.size() > 8 && ip.size() > 2) {
        return k[8] - ip[2];
    }
    return -999.0;
}

// ==========================================
// 4. Ellipse cut on minv plane
// =========================================
bool ellipse_cut_func(double u, double v)
{
  double u0 = -92.9524, v0 = 7.1644, su = 33.7633, sv = 42.0163, rho = 0.556;
  double n_sigma_cut = 2.5;

  double du = u - u0;
  double dv = v - v0;
  double inv_rho2 = 1.0 / (1.0 - rho * rho);

  double dist2 = inv_rho2 * ((du * du) / (su * su) +
                              (dv * dv) / (sv * sv) -
                              2.0 * rho * (du * dv) / (su * sv));

  return std::sqrt(std::max(0.0, dist2)) > n_sigma_cut;
}

// ==========================================
// 4. Cosine between two vectors
// ==========================================
double cosine_between_vectors(const ROOT::RVec<double>& v1, const ROOT::RVec<double>& v2) {
    if (v1.size() > 2 && v2.size() > 2) {
        double dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
        double mag_v1 = std::sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
        double mag_v2 = std::sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);
        if (mag_v1 > 0 && mag_v2 > 0) {
            return dot_product / (mag_v1 * mag_v2);
        }
    }
    return -999.0;
}

// ==========================================
// 5. Cut on m1 vs m2 plane
// ==========================================
bool m1_vs_m2_cut(const double& m1, const double& m2, const double& cut1, const double& cut2) {
    double u = (m1 + m2) / std::sqrt(2.0);
    double v = (m1 - m2) / std::sqrt(2.0);

    return std::abs(u) < cut1 && std::abs(v) < cut2;
}
""")


def add_charged_kaon_mass(
    df,
    out_col: str,
    m_plus: float,  # Masa dla cząstki DODATNIEJ (+)
    m_minus: float,  # Masa dla cząstki UJEMNEJ (-)
    t1_cols="trk1[0], trk1[1], trk1[2]",
    t2_cols="trk2[0], trk2[1], trk2[2]",
    q1_col="Curv1",  # Kolumna określająca znak ładunku pierwszego śladu
):
    """Oblicza masę niezmienniczą dwóch śladów, przypisując masę m_plus

    do cząstki dodatniej (+) oraz m_minus do ujemnej (-).
    """
    # 1. Obliczamy masę pod hipotezą: trk1 (+), trk2 (-)
    df_temp = df.Define(
        f"_{out_col}_pos1",
        f"calc_m_K_charged({t1_cols}, {m_plus}, {t2_cols}, {m_minus})",
    )

    # 2. Obliczamy masę pod hipotezą: trk1 (-), trk2 (+)
    df_temp = df_temp.Define(
        f"_{out_col}_pos2",
        f"calc_m_K_charged({t1_cols}, {m_minus}, {t2_cols}, {m_plus})",
    )

    # 3. Wybieramy właściwą masę na podstawie znaku ładunku pierwszego śladu
    return df_temp.Define(
        out_col, f"{q1_col} < 0 ? _{out_col}_pos1 : _{out_col}_pos2"
    )