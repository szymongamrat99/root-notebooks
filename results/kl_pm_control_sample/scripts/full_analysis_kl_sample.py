import glob
import json
import time
import os
import hashlib
import sys
import yaml
from pathlib import Path

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

grandparent_dir = script_dir.parent.parent
sys.path.append(str(grandparent_dir))

import ROOT
import numpy as np

current_dir = Path(__file__).resolve().parent
target_dir = current_dir.parent.parent.parent
sys.path.append(str(target_dir))

import utils.physics_utils as PU
from utils.general_utils import get_valid_files_fast, log_analysis_run, build_cut_expression
from utils.hist_utils import (
    split_variable_group,
    fill_1d_histograms,
    fill_2d_histograms,
    draw_and_save_1d_histograms,
    draw_and_save_2d_histograms,
)

ROOT.EnableImplicitMT(45)
ROOT.gStyle.SetPalette(ROOT.kRainbow)

ROOT.gSystem.Load("/workspace/KLOE/Include/Codes/libLibRec.so")
ROOT.gSystem.Load("/workspace/KLOE/Subanalysis/CPFit/libCPFit.so")
ROOT.gInterpreter.AddIncludePath("/workspace/KLOE/src/Include/Codes/inc")
ROOT.gInterpreter.AddIncludePath("/workspace/KLOE/src/Include/FortranAnalysis/inc")
ROOT.gInterpreter.AddIncludePath("/workspace/KLOE/src/Subanalysis/CPFit/inc")

ROOT.gInterpreter.Declare('#include "/workspace/KLOE/src/Include/FortranAnalysis/inc/fort_common.h"')
ROOT.gInterpreter.Declare('#include "/workspace/KLOE/src/Include/Codes/inc/const.h"')
ROOT.gInterpreter.Declare('#include "/workspace/KLOE/src/Include/Codes/inc/HypothesisCodes.h"')
ROOT.gInterpreter.Declare('#include "/workspace/KLOE/src/Include/Codes/inc/SelectionCodes.h"')
ROOT.gInterpreter.Declare('#include "/workspace/KLOE/src/Include/Codes/inc/ErrorLogs.h"')

ROOT.gInterpreter.Declare('#include "/workspace/KLOE/src/Include/Codes/inc/charged_mom.h"')
ROOT.gInterpreter.Declare('#include "/workspace/KLOE/src/Include/Codes/inc/kloe_class.h"')

ROOT.gInterpreter.Declare('#include "/workspace/KLOE/src/Subanalysis/CPFit/inc/cpfit.hpp"')

ROOT.KLOE.setGlobalStyle()

channName = {int(k): str(v) for k, v in ROOT.KLOE.channName}
channTitle = {str(k): str(v) for k, v in ROOT.KLOE.channTitle}
channColor = {str(k): int(v) for k, v in ROOT.KLOE.channColor}

# Enrichment for control samples
channTitle["hasKsPM"] = "Event includes K_{S}#rightarrow#pi^{+}#pi^{-}"
channTitle["!hasKsPM"] = "Event does not include K_{S}#rightarrow#pi^{+}#pi^{-}"

channColor["hasKsPM"] = ROOT.kRed
channColor["!hasKsPM"] = ROOT.kBlue

print(channName)

error_codes_enum = ROOT.ErrorHandling.ErrorCodes
err_code_dict = {
    int(val): name 
    for name in dir(error_codes_enum) 
    if not name.startswith("_") 
    and isinstance((val := getattr(error_codes_enum, name)), int)  # sprawdza, czy to int
    and 300 <= val <= 314
}

print(err_code_dict)

# Pobierz wszystkie pliki pasujące do wzorca
all_files = glob.glob(
    "/workspace/root_files/hal/ALL_PHYS_FOUR_PI_NoSmearing/*.root"
)
all_files += glob.glob(
    "/workspace/root_files/hal/ALL_PHYS2_FOUR_PI_NoSmearing/*.root"
)
all_files += glob.glob(
    "/workspace/root_files/hal/ALL_PHYS3_FOUR_PI_NoSmearing/*.root"
)

data_files = glob.glob(
  f"/workspace/root_files/hal/DATA_FOUR_PI_NoSmearing/*.root"
)

# Użycie (błyskawiczne!):
all_files = get_valid_files_fast(all_files)
data_files = get_valid_files_fast(data_files)

# Reduction of the file sets for tests
LIMIT = -1

limit_slice = None if LIMIT == -1 else LIMIT
all_files = all_files[:limit_slice]
data_files = data_files[:limit_slice]

# Scaling factor
lumi_data = 1634433.9750 if LIMIT == -1 else len(all_files) * 4.0
lumi_mc = 1444990.0313 if LIMIT == -1 else len(all_files) * 4.0
scaling_factor = lumi_data / lumi_mc
print(f"Lumi factor: {scaling_factor:.4f}")

# Przekaż czystą listę plików bezpośrednio do RDataFrame
df = ROOT.RDataFrame("h1", all_files)
df_data = ROOT.RDataFrame("h1", data_files)

mPi0 = 134.9768 # MeV
mPiCh = 139.57039 # MeV
mK0 = 497.611 # MeV
mPhi = 1019.461 # MeV

pK = np.sqrt((mPhi / 2.0)**2 - mK0**2)

df_Preselection = df.Filter("mctruth == 0 || mctruth == 7 || mctruth == -1").Define("ks_kl_semileptonic", "find_ks_kl_semileptonic(mctruth, ntmc, nvtxmc, vtxmc, mother, pidmc)")

# --- PRZETWARZANIE DANYCH (DATA) ---
df_data = (
    df_data.Define("KchrecKS_5", f"KchrecKS[5] - {mK0}")
    .Define("KchrecKL_5", f"KchrecKL[5] - {mK0}")
    .Define("rho_KS", "calc_cylindrical_radius(KchrecKS, ipKS)")
    .Define("rho_KL", "calc_cylindrical_radius(KchrecKL, ipKL)")
    .Define("rho_KS_MC", "calc_cylindrical_radius(Kchmc, ipmc)")
    .Define("rho_KL_MC", "calc_cylindrical_radius(Knemc, ipmc)")
    .Define("z_KS", "calc_z_component(KchrecKS, ipKS)")
    .Define("z_KL", "calc_z_component(KchrecKL, ipKL)")
    .Define("z_KS_MC", "calc_z_component(Kchmc, ipmc)")
    .Define("z_KL_MC", "calc_z_component(Knemc, ipmc)")
    .Define("R_KS", "calc_spherical_radius(KchrecKS, ipKS)")
    .Define("R_KL", "calc_spherical_radius(KchrecKL, ipKL)")
    .Define("R_KS_MC", "calc_spherical_radius(Kchmc, ipmc)")
    .Define("R_KL_MC", "calc_spherical_radius(Knemc, ipmc)")
    .Define("QmissKS", "sqrt(pow(KchrecKS[3] - KchboostKS[3], 2) + pow(KchrecKS[0] - KchboostKS[0], 2) + pow(KchrecKS[1] - KchboostKS[1], 2) + pow(KchrecKS[2] - KchboostKS[2], 2))")
    .Define("QmissKL", "sqrt(pow(KchrecKL[3] - KchboostKL[3], 2) + pow(KchrecKL[0] - KchboostKL[0], 2) + pow(KchrecKL[1] - KchboostKL[1], 2) + pow(KchrecKL[2] - KchboostKL[2], 2))")
    .Define("neu_clu_num", "neutral_clusters_num(nclu, ntcl, Asscl)")
    ######################### Lorentz transformed variables ############################
    .Define("trk1KS_phiCM", "lorentz_transform(trk1KS, Broots, Bpx, Bpy, Bpz)")
    .Define("trk2KS_phiCM", "lorentz_transform(trk2KS, Broots, Bpx, Bpy, Bpz)")
    .Define("trk1KL_phiCM", "lorentz_transform(trk1KL, Broots, Bpx, Bpy, Bpz)")
    .Define("trk2KL_phiCM", "lorentz_transform(trk2KL, Broots, Bpx, Bpy, Bpz)")
    .Define("KchrecKS_phiCM", "lorentz_transform(KchrecKS, Broots, Bpx, Bpy, Bpz)")
    .Define("KchrecKL_phiCM", "lorentz_transform(KchrecKL, Broots, Bpx, Bpy, Bpz)")
    .Define("mag_KS_phiCM", f"magnitude(KchrecKS_phiCM) - {pK}")
    .Define("mag_KL_phiCM", f"magnitude(KchrecKL_phiCM) - {pK}")
    .Define("KchboostKS_phiCM", "lorentz_transform(KchboostKS, Broots, Bpx, Bpy, Bpz)")
    .Define("KchboostKL_phiCM", "lorentz_transform(KchboostKL, Broots, Bpx, Bpy, Bpz)")
    .Define("KS_vertex", "ROOT::RVec<double>{KchrecKS[6] - ipKS[0], KchrecKS[7] - ipKS[1], KchrecKS[8] - ipKS[2]}")
    .Define("KL_vertex", "ROOT::RVec<double>{KchrecKL[6] - ipKL[0], KchrecKL[7] - ipKL[1], KchrecKL[8] - ipKL[2]}")
    ######################### Variables for Omega-pi0 geometrical cut ############################
    .Define("bhabha_IP", "ROOT::RVec<double>{Bx, By, Bz}")
    .Define("bhabha_IP_changed", "ROOT::RVec<double>{Bx, By, KchrecKL[8]}")
    .Define("rho_00_closest", "calc_cylindrical_radius(KchrecKS, bhabha_IP_changed)")
    .Define("rho_pm_closest", "calc_cylindrical_radius(KchrecKL, bhabha_IP)")
    .Define("rho_omega", "hypot(rho_00_closest, rho_pm_closest)")
    .Define("z_00_closest", "KchrecKS[8] - KchrecKL[8]")
    .Define("z_pm_closest", "KchrecKL[8] - Bz")
    ###############################################################################################
    ######################### Variables for bad tracks geometrical cut ############################
    .Define("rho_pm_00", "hypot(KchrecKL[6] - KchrecKS[6], KchrecKL[7] - KchrecKS[7])")
    .Define("z_pm_00", "KchrecKL[8] - KchrecKS[8]")
    .Define("cos_trk1_K", "cosine_between_vectors(trk1KL, KchrecKL)")
    .Define("cos_trk2_K", "cosine_between_vectors(trk2KL, KchrecKL)")
    ###############################################################################################
    ######################## Delta t ##############################################################
    .Define("Delta_t_fit", "KaonKLTimeCMFitFit - KaonKSTimeCMFitFit")
    .Define("Delta_t_rec", "KaonKLTimeCMBoostBoost - KaonKSTimeCMBoostBoost")
    .Define("Delta_t_MC", "KaonNeTimeCMMC - KaonChTimeCMMC")
    ###############################################################################################
    ######################## Resolutions ##########################################################
    .Define("Delta_t_fit_resolution", "Delta_t_fit - Delta_t_MC")
    .Define("Delta_t_rec_resolution", "Delta_t_rec - Delta_t_MC")
    .Define("trk1KL_px_resolution", "trk1KL[0] - trk1KLmc[0]")
    .Define("trk2KL_px_resolution", "trk2KL[0] - trk2KLmc[0]")
    .Define("trk1KL_py_resolution", "trk1KL[1] - trk1KLmc[1]")
    .Define("trk2KL_py_resolution", "trk2KL[1] - trk2KLmc[1]")
    .Define("trk1KL_pz_resolution", "trk1KL[2] - trk1KLmc[2]")
    .Define("trk2KL_pz_resolution", "trk2KL[2] - trk2KLmc[2]")
    .Define("trk1KS_px_resolution", "trk1KS[0] - trk1KSmc[0]")
    .Define("trk2KS_px_resolution", "trk2KS[0] - trk2KSmc[0]")
    .Define("trk1KS_py_resolution", "trk1KS[1] - trk1KSmc[1]")
    .Define("trk2KS_py_resolution", "trk2KS[1] - trk2KSmc[1]")
    .Define("trk1KS_pz_resolution", "trk1KS[2] - trk1KSmc[2]")
    .Define("trk2KS_pz_resolution", "trk2KS[2] - trk2KSmc[2]")
    .Define("trk1KS_energy_resolution", "trk1KS[3] - trk1KSmc[3]")
    .Define("trk2KS_energy_resolution", "trk2KS[3] - trk2KSmc[3]")
    .Define("trk1KL_energy_resolution", "trk1KL[3] - trk1KLmc[3]")
    .Define("trk2KL_energy_resolution", "trk2KL[3] - trk2KLmc[3]")
    .Define("KchrecKS_px_resolution", "KchrecKS[0] - Kchmc[0]")
    .Define("KchrecKS_py_resolution", "KchrecKS[1] - Kchmc[1]")
    .Define("KchrecKS_pz_resolution", "KchrecKS[2] - Kchmc[2]")
    .Define("KchrecKS_energy_resolution", "KchrecKS[3] - Kchmc[3]")
    .Define("KchrecKS_x_resolution", "KchrecKS[6] - Kchmc[6]")
    .Define("KchrecKS_y_resolution", "KchrecKS[7] - Kchmc[7]")
    .Define("KchrecKS_z_resolution", "KchrecKS[8] - Kchmc[8]")
    .Define("KchrecKL_px_resolution", "KchrecKL[0] - Knemc[0]")
    .Define("KchrecKL_py_resolution", "KchrecKL[1] - Knemc[1]")
    .Define("KchrecKL_pz_resolution", "KchrecKL[2] - Knemc[2]")
    .Define("KchrecKL_energy_resolution", "KchrecKL[3] - Knemc[3]")
    .Define("KchrecKL_x_resolution", "KchrecKL[6] - Knemc[6]")
    .Define("KchrecKL_y_resolution", "KchrecKL[7] - Knemc[7]")
    .Define("KchrecKL_z_resolution", "KchrecKL[8] - Knemc[8]")
    .Define("KchboostKS_px_resolution", "KchboostKS[0] - Kchmc[0]")
    .Define("KchboostKS_py_resolution", "KchboostKS[1] - Kchmc[1]")
    .Define("KchboostKS_pz_resolution", "KchboostKS[2] - Kchmc[2]")
    .Define("KchboostKS_energy_resolution", "KchboostKS[3] - Kchmc[3]")
    .Define("KchboostKL_px_resolution", "KchboostKL[0] - Knemc[0]")
    .Define("KchboostKL_py_resolution", "KchboostKL[1] - Knemc[1]")
    .Define("KchboostKL_pz_resolution", "KchboostKL[2] - Knemc[2]")
    .Define("KchboostKL_energy_resolution", "KchboostKL[3] - Knemc[3]")
    .Define("R_KS_resolution", "R_KS - R_KS_MC")
    .Define("R_KL_resolution", "R_KL - R_KL_MC")
    .Define("rho_KS_resolution", "rho_KS - rho_KS_MC")
    .Define("rho_KL_resolution", "rho_KL - rho_KL_MC")
    .Define("z_KS_resolution", "z_KS - z_KS_MC")
    .Define("z_KL_resolution", "z_KL - z_KL_MC")
    .Define("ip_KS_vs_KL_x", "ipKS[0] - ipKL[0]")
    .Define("ip_KS_vs_KL_y", "ipKS[1] - ipKL[1]")
    .Define("ip_KS_vs_KL_z", "ipKS[2] - ipKL[2]")
    .Define("KchrecKS_5_resolution", f"KchrecKS[5] - {mK0}")
    .Define("KchrecKL_5_resolution", f"KchrecKL[5] - {mK0}")
    .Define("ip_KS_z_resolution", "ipKS[2] - ipmc[2]")
    .Define("ip_KL_z_resolution", "ipKL[2] - ipmc[2]")
    ###############################################################################################
    .Filter("mcflag == 0 && mctruth != -1")
)

# --- PRZETWARZANIE MONTE CARLO (MC) ---
df = (
    df.Define("KchrecKS_5", f"KchrecKS[5] - {mK0}")
    .Define("KchrecKL_5", f"KchrecKL[5] - {mK0}")
    .Define("rho_KS", "calc_cylindrical_radius(KchrecKS, ipKS)")
    .Define("rho_KL", "calc_cylindrical_radius(KchrecKL, ipKL)")
    .Define("rho_KS_MC", "calc_cylindrical_radius(Kchmc, ipmc)")
    .Define("rho_KL_MC", "calc_cylindrical_radius(Knemc, ipmc)")
    .Define("z_KS", "calc_z_component(KchrecKS, ipKS)")
    .Define("z_KL", "calc_z_component(KchrecKL, ipKL)")
    .Define("z_KS_MC", "calc_z_component(Kchmc, ipmc)")
    .Define("z_KL_MC", "calc_z_component(Knemc, ipmc)")
    .Define("R_KS", "calc_spherical_radius(KchrecKS, ipKS)")
    .Define("R_KL", "calc_spherical_radius(KchrecKL, ipKL)")
    .Define("R_KS_MC", "calc_spherical_radius(Kchmc, ipmc)")
    .Define("R_KL_MC", "calc_spherical_radius(Knemc, ipmc)")
    .Define("QmissKS", "sqrt(pow(KchrecKS[3] - KchboostKS[3], 2) + pow(KchrecKS[0] - KchboostKS[0], 2) + pow(KchrecKS[1] - KchboostKS[1], 2) + pow(KchrecKS[2] - KchboostKS[2], 2))")
    .Define("QmissKL", "sqrt(pow(KchrecKL[3] - KchboostKL[3], 2) + pow(KchrecKL[0] - KchboostKL[0], 2) + pow(KchrecKL[1] - KchboostKL[1], 2) + pow(KchrecKL[2] - KchboostKL[2], 2))")
    .Define("neu_clu_num", "neutral_clusters_num(nclu, ntcl, Asscl)")
    ######################### Lorentz transformed variables ############################
    .Define("trk1KS_phiCM", "lorentz_transform(trk1KS, Broots, Bpx, Bpy, Bpz)")
    .Define("trk2KS_phiCM", "lorentz_transform(trk2KS, Broots, Bpx, Bpy, Bpz)")
    .Define("trk1KL_phiCM", "lorentz_transform(trk1KL, Broots, Bpx, Bpy, Bpz)")
    .Define("trk2KL_phiCM", "lorentz_transform(trk2KL, Broots, Bpx, Bpy, Bpz)")
    .Define("KchrecKS_phiCM", "lorentz_transform(KchrecKS, Broots, Bpx, Bpy, Bpz)")
    .Define("KchrecKL_phiCM", "lorentz_transform(KchrecKL, Broots, Bpx, Bpy, Bpz)")
    .Define("mag_KS_phiCM", f"magnitude(KchrecKS_phiCM) - {pK}")
    .Define("mag_KL_phiCM", f"magnitude(KchrecKL_phiCM) - {pK}")
    .Define("KchboostKS_phiCM", "lorentz_transform(KchboostKS, Broots, Bpx, Bpy, Bpz)")
    .Define("KchboostKL_phiCM", "lorentz_transform(KchboostKL, Broots, Bpx, Bpy, Bpz)")
    .Define("KS_vertex", "ROOT::RVec<double>{KchrecKS[6] - ipKS[0], KchrecKS[7] - ipKS[1], KchrecKS[8] - ipKS[2]}")
    .Define("KL_vertex", "ROOT::RVec<double>{KchrecKL[6] - ipKL[0], KchrecKL[7] - ipKL[1], KchrecKL[8] - ipKL[2]}")
    ######################### Variables for Omega-pi0 geometrical cut ############################
    .Define("bhabha_IP", "ROOT::RVec<double>{Bx, By, Bz}")
    .Define("bhabha_IP_changed", "ROOT::RVec<double>{Bx, By, KchrecKL[8]}")
    .Define("rho_00_closest", "calc_cylindrical_radius(KchrecKS, bhabha_IP_changed)")
    .Define("rho_pm_closest", "calc_cylindrical_radius(KchrecKL, bhabha_IP)")
    .Define("rho_omega", "hypot(rho_00_closest, rho_pm_closest)")
    .Define("z_00_closest", "KchrecKS[8] - KchrecKL[8]")
    .Define("z_pm_closest", "KchrecKL[8] - Bz")
    ###############################################################################################
    ######################### Variables for bad tracks geometrical cut ############################
    .Define("rho_pm_00", "hypot(KchrecKL[6] - KchrecKS[6], KchrecKL[7] - KchrecKS[7])")
    .Define("z_pm_00", "KchrecKL[8] - KchrecKS[8]")
    .Define("cos_trk1_K", "cosine_between_vectors(trk1KL, KchrecKL)")
    .Define("cos_trk2_K", "cosine_between_vectors(trk2KL, KchrecKL)")
    ###############################################################################################
    ######################## Delta t ##############################################################
    .Define("Delta_t_fit", "KaonKLTimeCMFitFit - KaonKSTimeCMFitFit")
    .Define("Delta_t_rec", "KaonKLTimeCMBoostBoost - KaonKSTimeCMBoostBoost")
    .Define("Delta_t_MC", "KaonNeTimeCMMC - KaonChTimeCMMC")
    ###############################################################################################
    ######################## Resolutions ##########################################################
    .Define("Delta_t_fit_resolution", "Delta_t_fit - Delta_t_MC")
    .Define("Delta_t_rec_resolution", "Delta_t_rec - Delta_t_MC")
    .Define("trk1KL_px_resolution", "trk1KL[0] - trk1KLmc[0]")
    .Define("trk2KL_px_resolution", "trk2KL[0] - trk2KLmc[0]")
    .Define("trk1KL_py_resolution", "trk1KL[1] - trk1KLmc[1]")
    .Define("trk2KL_py_resolution", "trk2KL[1] - trk2KLmc[1]")
    .Define("trk1KL_pz_resolution", "trk1KL[2] - trk1KLmc[2]")
    .Define("trk2KL_pz_resolution", "trk2KL[2] - trk2KLmc[2]")
    .Define("trk1KS_px_resolution", "trk1KS[0] - trk1KSmc[0]")
    .Define("trk2KS_px_resolution", "trk2KS[0] - trk2KSmc[0]")
    .Define("trk1KS_py_resolution", "trk1KS[1] - trk1KSmc[1]")
    .Define("trk2KS_py_resolution", "trk2KS[1] - trk2KSmc[1]")
    .Define("trk1KS_pz_resolution", "trk1KS[2] - trk1KSmc[2]")
    .Define("trk2KS_pz_resolution", "trk2KS[2] - trk2KSmc[2]")
    .Define("trk1KS_energy_resolution", "trk1KS[3] - trk1KSmc[3]")
    .Define("trk2KS_energy_resolution", "trk2KS[3] - trk2KSmc[3]")
    .Define("trk1KL_energy_resolution", "trk1KL[3] - trk1KLmc[3]")
    .Define("trk2KL_energy_resolution", "trk2KL[3] - trk2KLmc[3]")
    .Define("KchrecKS_px_resolution", "KchrecKS[0] - Kchmc[0]")
    .Define("KchrecKS_py_resolution", "KchrecKS[1] - Kchmc[1]")
    .Define("KchrecKS_pz_resolution", "KchrecKS[2] - Kchmc[2]")
    .Define("KchrecKS_energy_resolution", "KchrecKS[3] - Kchmc[3]")
    .Define("KchrecKS_x_resolution", "KchrecKS[6] - Kchmc[6]")
    .Define("KchrecKS_y_resolution", "KchrecKS[7] - Kchmc[7]")
    .Define("KchrecKS_z_resolution", "KchrecKS[8] - Kchmc[8]")
    .Define("KchrecKL_px_resolution", "KchrecKL[0] - Knemc[0]")
    .Define("KchrecKL_py_resolution", "KchrecKL[1] - Knemc[1]")
    .Define("KchrecKL_pz_resolution", "KchrecKL[2] - Knemc[2]")
    .Define("KchrecKL_energy_resolution", "KchrecKL[3] - Knemc[3]")
    .Define("KchrecKL_x_resolution", "KchrecKL[6] - Knemc[6]")
    .Define("KchrecKL_y_resolution", "KchrecKL[7] - Knemc[7]")
    .Define("KchrecKL_z_resolution", "KchrecKL[8] - Knemc[8]")
    .Define("KchboostKS_px_resolution", "KchboostKS[0] - Kchmc[0]")
    .Define("KchboostKS_py_resolution", "KchboostKS[1] - Kchmc[1]")
    .Define("KchboostKS_pz_resolution", "KchboostKS[2] - Kchmc[2]")
    .Define("KchboostKS_energy_resolution", "KchboostKS[3] - Kchmc[3]")
    .Define("KchboostKL_px_resolution", "KchboostKL[0] - Knemc[0]")
    .Define("KchboostKL_py_resolution", "KchboostKL[1] - Knemc[1]")
    .Define("KchboostKL_pz_resolution", "KchboostKL[2] - Knemc[2]")
    .Define("KchboostKL_energy_resolution", "KchboostKL[3] - Knemc[3]")
    .Define("R_KS_resolution", "R_KS - R_KS_MC")
    .Define("R_KL_resolution", "R_KL - R_KL_MC")
    .Define("rho_KS_resolution", "rho_KS - rho_KS_MC")
    .Define("rho_KL_resolution", "rho_KL - rho_KL_MC")
    .Define("z_KS_resolution", "z_KS - z_KS_MC")
    .Define("z_KL_resolution", "z_KL - z_KL_MC")
    .Define("ip_KS_vs_KL_x", "ipKS[0] - ipKL[0]")
    .Define("ip_KS_vs_KL_y", "ipKS[1] - ipKL[1]")
    .Define("ip_KS_vs_KL_z", "ipKS[2] - ipKL[2]")
    .Define("KchrecKS_5_resolution", f"KchrecKS[5] - {mK0}")
    .Define("KchrecKL_5_resolution", f"KchrecKL[5] - {mK0}")
    .Define("ip_KS_z_resolution", "ipKS[2] - ipmc[2]")
    .Define("ip_KL_z_resolution", "ipKL[2] - ipmc[2]")
    ###############################################################################################    
    .Filter("mcflag == 1 && mctruth != -1")
)

# df Signal
df_Signal = df.Filter(
    f"mcflag == 1 && mctruth == 7"
)

df_Background = df.Filter(
    f"mcflag == 1 && mctruth != 7 && mctruth != -1"
)

base_path = "/workspace/root_notebooks/results"
cache_path = "/workspace/analysis_snapshot"
analysis_dir = "kl_pm_control_sample"

current_date = time.strftime("%Y-%m-%d")

metadata = {
    "saved_errors": "pi+pi-pi+pi-",
    "cutset": "NONE",
    "luminosity_pb_MC": lumi_mc,
    "luminosity_pb_data": lumi_data,
    "scale_factor": scaling_factor,
}

analysis_hash = hashlib.md5(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
metadata["analysis_hash"] = analysis_hash

current_result_dir = f"{base_path}/{analysis_dir}/{analysis_hash}"
cache_path = f"{cache_path}/{analysis_dir}/{analysis_hash}"
config_path = f"{base_path}/{analysis_dir}/config"

os.makedirs(current_result_dir, exist_ok=True)
with open(f"{current_result_dir}/analysis_config.json", "w") as f:
  json.dump(metadata, f, indent=4)

########################## LOAD HISTOGRAM CONFIGURATION ##############################
hist_config_file_path = f"{base_path}/{analysis_dir}/config/histograms_config_1D.yaml"
with open(hist_config_file_path, "r") as f:
    hist_config = yaml.safe_load(f)
######################################################################################

############ LOAD VARIABLES RESOLUTIONS PER CUT HASH #################################
resolutions = {}
list_of_dirs = [
    f for f in os.listdir(config_path) if os.path.isdir(os.path.join(config_path, f))
]

for d in list_of_dirs:
  resolution_file_path = f"{config_path}/{d}/resolutions_config.yaml"
  with open(resolution_file_path, "r") as f:
      resolutions_config = yaml.safe_load(f)
  resolutions[d] = resolutions_config
######################################################################################

try:
  variables_1D = list(hist_config.get("histograms_1d", {}).keys())
  TRUTH_ONLY_VARS_1D = hist_config.get("truth_only_vars_1d", [])
  TRUTH_MCTRUTH_ID_1D = hist_config.get("truth_mctruth_id_1d", 1)

  variables_1D_titles = []
  unit_1D = []
  range_X_1D = {}
  modes = {}
  predefined_resolution = {}

  for var in variables_1D:
      variables_1D_titles.append(hist_config["histograms_1d"][var].get("title", ""))
      unit_1D.append(hist_config["histograms_1d"][var].get("unit", ""))
      modes[var] = list(hist_config["histograms_1d"][var].get("range_x", {}).keys())

      range_X_1D[var] = {}
      for mode in modes[var]:
          range_X_1D[var][mode] = hist_config["histograms_1d"][var].get("range_x", {}).get(mode, [])
      
      predefined_resolution[var] = hist_config["histograms_1d"][var].get("resolution", False)
except Exception as e:
  print(f"Error occurred: {e}")

import math


## Prepare 1D histogram model on levels:
## cut_hash --> mode --> variable

def prepare_hist_model(cut_hash: list, vars: list, vars_titles: list, unit: list, range_X: dict, resolutions: dict) -> dict:
  hist_model = {}

  if isinstance(cut_hash, str):
      cut_hash = [cut_hash]

  for cut in cut_hash:
      hist_model[cut] = {}

      for var in variables_1D:

        sigma = predefined_resolution.get(var, False)
        if not sigma:
          ## Prepare sigma for the variable
          sigma_core = resolutions.get(cut, {}).get('variables', {}).get(var, {}).get('sigma_core', [1.0])[0]
          sigma_tail = resolutions.get(cut, {}).get('variables', {}).get(var, {}).get('sigma_tail', [1.0])[0]

          if "resolution" in var:
            if range_len < 0.5:
              sigma_core = 0.001
              sigma_tail = 0.001
            elif range_len < 3:
              sigma_core = 0.01
              sigma_tail = 0.01
            elif range_len < 50:
              sigma_core = 0.1
              sigma_tail = 0.1
            else:
              sigma_core = 1.0
              sigma_tail = 1.0
            
          sigma = min(sigma_core, sigma_tail)

        for mode in modes[var]:
          if mode not in hist_model[cut]:
              hist_model[cut][mode] = {}

          ## Calculate the range length for the variable in the given mode
          range_X_min, range_X_max = range_X[var].get(mode, [0, 1])
          range_len = range_X_max - range_X_min

          title = f"{vars_titles[vars.index(var)]} [{unit[vars.index(var)]}]" if var in vars else ""

          hist_model[cut][mode][var] = {
              "title": title,
              "range_x": range_X[var].get(mode, []),
              "n_bins": int(range_len / sigma) if sigma > 0 else range_len,
              "resolution": sigma
          }

  return hist_model

tag_cut_list = [
            "abs(KchrecKS_5) <= 5.0",
            "abs(mag_KS_phiCM) <= 8.0"
            ]

########### Fiducial volume Omega-pi0 #########################
fiducial_volume_omega = "rho_pm_closest < 2.0 && rho_00_closest < 1.5 && abs(z_pm_closest) < 2.0 && abs(z_00_closest) < 2.0"
###############################################################
########### Fiducial volume tracks back to back ###############
fiducial_volume_close_tracks = "rho_pm_00 < 2.05 && abs(z_pm_00) < 2.45"
###############################################################

cut_list = ["abs(KchrecKL_5) <= 3 * 1.1",
            f"({fiducial_volume_omega} && rho_pm_closest > 1.0) || !({fiducial_volume_omega})",
            f"({fiducial_volume_close_tracks} && (abs(cos_trk1_K) < 0.8 || abs(cos_trk2_K) < 0.8) ) || !({fiducial_volume_close_tracks})"
            ]

full_cut_list = tag_cut_list + cut_list

print(full_cut_list)

############# FIRST CUT LIST FOR 000 TAG ################
from utils.hist_utils import fill_1D_histogram_simple

division_list = ["channels"]
mc_flags = {}

for division in division_list:
  if division == "tag":
    mc_flags = {"true": "Data", "hasKsPM": "hasKsPM", "!hasKsPM": "!hasKsPM"}
    file_base_name = "tag_ks_pm"
  else:
    mc_flags = channName
    file_base_name = "tag_ks_pm_chann_div"

  # --- GŁÓWNA PĘTLA BUDUJĄCA CIĘCIA KROK PO KROKU ---
  # Możesz zacząć od indeksu -1 (czyli stan "beż żadnych cięć" / nocuts)
  # lub od i = 0 (jeśli chcesz zacząć od razu od pierwszego cięcia).
  for i in range(-1, len(full_cut_list)):

    # Wycinamy podzbiór cięć od początku do bieżącego kroku
    if i == -1:
      current_cut_list = []  # Stan bazowy bez cięć
      step_name = "00_no_cuts"
    else:
      current_cut_list = full_cut_list[: i + 1]
      step_name = f"step_{i+1:02d}"

    print(
        f"\n[INFO] Przetwarzanie kroku: {step_name} (liczba aktywnych cięć:"
        f" {len(current_cut_list)})"
    )

    # Przygotowanie metadanych, hashu i unikalnego katalogu dla tego kroku
    cuts_json_str = json.dumps(current_cut_list, indent=2)
    cut_hash = (
        "nocuts"
        if not current_cut_list
        else hashlib.md5(cuts_json_str.encode("utf-8")).hexdigest()[:8]
    )

    # Możesz opcjonalnie rozbić wyniki na podkatalogi dla każdego kroku, np.:
    # step_result_dir = os.path.join(current_result_dir, step_name)
    step_result_dir = (
        current_result_dir  # Lub zapisywać w jednym miejscu z hashem w nazwie pliku
    )

    total_cut = build_cut_expression(current_cut_list)

    # Rozdzielczości i model histogramu zależą od bieżącego zestawu cięć.
    histo1Dmodel = prepare_hist_model(
        cut_hash,
        variables_1D,
        variables_1D_titles,
        unit_1D,
        range_X_1D,
        resolutions,
    )
    model_for_cut = histo1Dmodel[cut_hash]

    truth_only_vars = set(TRUTH_ONLY_VARS_1D)
    main_indices = [
        index for index, variable in enumerate(variables_1D)
        if variable not in truth_only_vars
    ]
    truth_indices = [
        index for index, variable in enumerate(variables_1D)
        if variable in truth_only_vars
    ]
    vars_1D_main = [variables_1D[index] for index in main_indices]
    titles_1D_main = [variables_1D_titles[index] for index in main_indices]
    vars_1D_truth = [variables_1D[index] for index in truth_indices]
    titles_1D_truth = [variables_1D_titles[index] for index in truth_indices]

    def modes_for_variables(variables):
      return {
          mode: {}
          for mode, variables_by_mode in model_for_cut.items()
          if all(variable in variables_by_mode for variable in variables)
      }

    modes_1D_main = modes_for_variables(vars_1D_main)
    modes_1D_truth = modes_for_variables(vars_1D_truth)

    # Wypełnianie histogramów głównych: Data + kanały MC + suma MC
    hist_1D = fill_1D_histogram_simple(
        df,
        df_data,
        mc_flags,
        histo1Dmodel,
        cut_hash,
        total_cut,
        variables=vars_1D_main,
        scaling_factor=scaling_factor,
        compute_mc_sum=True,
        division=division
    )

    # Wypełnianie histogramów truth (jeśli istnieją)
    hist_1D_truth = {}
    if vars_1D_truth:
      truth_channels_1D = {
          TRUTH_MCTRUTH_ID_1D: channName[TRUTH_MCTRUTH_ID_1D]
      }
      hist_1D_truth = fill_1D_histogram_simple(
          df,
          df_data,
          truth_channels_1D,
          histo1Dmodel,
          cut_hash,
          total_cut,
          variables=vars_1D_truth,
          compute_mc_sum=False,
      )

    # Rysowanie i zapis do dedykowanych plików ROOT / PDF
    # (zakładam, że Twoja funkcja draw_and_save przyjmuje cut_hash i zapisuje pliki)
    root_file_name = f"{file_base_name}_1D_variables.root"
    hist_1D, canvas, legends = draw_and_save_1d_histograms(
        hist_1D,
        vars_1D_main,
        modes_1D_main,
        channColor,
        channTitle,
        step_result_dir,
        cut_hash,
        cuts_json_str,
        current_cut_list,
        root_file_name,
    )

    if vars_1D_truth:
      root_file_name_truth = (
          f"{file_base_name}_1D_truth_variables.root"
      )
      hist_1D_truth, canvas_truth, legends_truth = draw_and_save_1d_histograms(
          hist_1D_truth,
          vars_1D_truth,
          modes_1D_truth,
          channColor,
          channTitle,
          step_result_dir,
          cut_hash,
          cuts_json_str,
          current_cut_list,
          root_file_name_truth,
      )

    # Sprzątanie obiektów Canvas z pamięci ROOT-a
    for canva in canvas:
      canva.Close()
    if vars_1D_truth:
      for canva in canvas_truth:
        canva.Close()

  print(
      "[INFO] Pętla sekwencyjnego nakładania cięć została pomyślnie zakończona!"
)