import ROOT

# Utils used for Physics variables

import ROOT

# 1. Jednorazowa deklaracja generycznej funkcji C++ w ROOT
ROOT.gInterpreter.Declare(
    """
#include <Math/Vector4D.h>
#include <cmath>

double calc_m_K_charged(double px1, double py1, double pz1, double m1,
                        double px2, double py2, double pz2, double m2) {
    
    double E1 = std::sqrt(px1*px1 + py1*py1 + pz1*pz1 + m1*m1);
    double E2 = std::sqrt(px2*px2 + py2*py2 + pz2*pz2 + m2*m2);
    
    ROOT::Math::PxPyPzEVector p4_1(px1, py1, pz1, E1);
    ROOT::Math::PxPyPzEVector p4_2(px2, py2, pz2, E2);
    
    return (p4_1 + p4_2).M(); // Zwraca masę m [MeV/c^2]
}
"""
)


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