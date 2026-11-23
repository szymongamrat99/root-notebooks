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

current_dir = Path(__file__).resolve().parent
target_dir = current_dir.parent.parent.parent
sys.path.append(str(target_dir))

import utils.physics_utils as PU
from utils.general_utils import get_valid_files_fast, log_analysis_run, build_cut_expression, histogram_1D_config_creator
from utils.hist_utils import (
    split_variable_group,
    fill_1d_histograms,
    fill_2d_histograms,
    draw_and_save_1d_histograms,
    draw_and_save_2d_histograms,
)

ROOT.EnableImplicitMT(90)
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
channTitle["has000"] = "Event includes K_{S}#rightarrow3#pi^{0}"
channTitle["!has000"] = "Event does not include K_{S}#rightarrow3#pi^{0}"

channColor["has000"] = ROOT.kRed
channColor["!has000"] = ROOT.kBlue

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
    "/workspace/root_files/hal/ALL_PHYS_SEMILEPTONIC_NoSmearing/*.root"
)
all_files += glob.glob(
    "/workspace/root_files/hal/ALL_PHYS2_SEMILEPTONIC_NoSmearing/*.root"
)
all_files += glob.glob(
    "/workspace/root_files/hal/ALL_PHYS3_SEMILEPTONIC_NoSmearing/*.root"
)
all_files += glob.glob(
    "/workspace/root_files/kitt/ALL_PHYS_SEMILEPTONIC_NoSmearing/*.root"
)
all_files += glob.glob(
    "/workspace/root_files/kitt/ALL_PHYS2_SEMILEPTONIC_NoSmearing/*.root"
)
all_files += glob.glob(
    "/workspace/root_files/kitt/ALL_PHYS3_SEMILEPTONIC_NoSmearing/*.root"
)


data_files = glob.glob(
  f"/workspace/root_files/hal/DATA_SEMILEPTONIC_NoSmearing/*.root"
)
data_files += glob.glob(
  f"/workspace/root_files/kitt/DATA_SEMILEPTONIC_NoSmearing/*.root"
)

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

df_Preselection = df.Filter("mctruth == 0 || mctruth == 5 || mctruth == -1").Define("ks_kl_semileptonic", "find_ks_kl_semileptonic(mctruth, ntmc, nvtxmc, vtxmc, mother, pidmc)")

# --- PRZETWARZANIE DANYCH (DATA) ---
df_data = (
    df_data.Define("Kchrec_5", f"Kchrec[5] - {mK0}")
    .Define("Knerec_5", f"Knerec[5] - {mK0}")
    .Define("KnerecSix_5", f"KnerecSix[5] - {mK0}")
    .Define("KnetriKinFit_5", f"KnetriKinFit[5] - {mK0}")
    .Define("MassPi01", f"pi01[5] - {mPi0}")
    .Define("MassPi02", f"pi02[5] - {mPi0}")
    .Define("MassPi01Fit", f"pi01Fit[5] - {mPi0}")
    .Define("MassPi02Fit", f"pi02Fit[5] - {mPi0}")
    .Define("MassPi0NeutralFit1", f"pi0NeutralFit1[5] - {mPi0}")
    .Define("MassPi0NeutralFit2", f"pi0NeutralFit2[5] - {mPi0}")
    .Define("MassPi0PMFit1", f"pi0PMFit1[5] - {mPi0}")
    .Define("MassPi0PMFit2", f"pi0PMFit2[5] - {mPi0}")
    .Define("rho_ch", "calc_cylindrical_radius(Kchrec, ip)")
    .Define("rho_ne", "calc_cylindrical_radius(Knerec, ip)")
    .Define("z_ch", "calc_z_component(Kchrec, ip)")
    .Define("z_ne", "calc_z_component(Knerec, ip)")
    .Define("R_ch", "calc_spherical_radius(Kchrec, ip)")
    .Define("R_ne", "calc_spherical_radius(KnerecSix, ip)")
    .Define("neu_clu_num", "neutralClustersList.size()")
    ######################### Variables for Omega-pi0 geometrical cut ############################
    .Define("bhabha_IP", "ROOT::RVec<double>{Bx, By, Bz}")
    .Define("bhabha_IP_changed", "ROOT::RVec<double>{Bx, By, KchrecClosest[8]}")
    .Define("rho_00_closest", "calc_cylindrical_radius(Knerec, bhabha_IP_changed)")
    .Define("rho_pm_closest", "calc_cylindrical_radius(KchrecClosest, bhabha_IP)")
    .Define("rho_omega", "hypot(rho_00_closest, rho_pm_closest)")
    .Define("z_00_closest", "Knerec[8] - KchrecClosest[8]")
    .Define("z_pm_closest", "KchrecClosest[8] - Bz")
    ###############################################################################################
    ######################### Variables for bad tracks geometrical cut ############################
    .Define("rho_pm_00", "hypot(Kchrec[6] - Knerec[6], Kchrec[7] - Knerec[7])")
    .Define("z_pm_00", "Kchrec[8] - Knerec[8]")
    .Define("cos_trk1_K", "cosine_between_vectors(trk1, Kchrec)")
    .Define("cos_trk2_K", "cosine_between_vectors(trk2, Kchrec)")
    ###############################################################################################
    ######################## Delta t ##############################################################
    .Define("Delta_t_fit", "KaonChTimeCMSignalFit - KaonNeTimeCMSignalFit")
    .Define("Delta_t_MC", "KaonChTimeCMMC - KaonNeTimeCMMC")
    .Define("Delta_t_rec", "KaonChTimeCMBoostLor - KaonNeTimeCMBoostLor")
    .Define("Delta_t_tri_fit", "KaonChTimeCMBoostTriFit - KaonNeTimeCMBoostTriFit")
    ###############################################################################################
    ######################## Resolutions ##########################################################
    .Define("Delta_t_fit_resolution", "Delta_t_fit - Delta_t_MC")
    .Define("Delta_t_rec_resolution", "Delta_t_rec - Delta_t_MC")
    .Define("Delta_t_tri_fit_resolution", "Delta_t_tri_fit - Delta_t_MC")
    .Define("trk1_matched", "get_matched_vector(trk1, trk2, trk1MC, trk2MC)[0]")
    .Define("trk2_matched", "get_matched_vector(trk1, trk2, trk1MC, trk2MC)[1]")
    .Define("trk1_px_resolution", "trk1_matched[0] - trk1MC[0]")
    .Define("trk2_px_resolution", "trk2_matched[0] - trk2MC[0]")
    .Define("trk1_py_resolution", "trk1_matched[1] - trk1MC[1]")
    .Define("trk2_py_resolution", "trk2_matched[1] - trk2MC[1]")
    .Define("trk1_pz_resolution", "trk1_matched[2] - trk1MC[2]")
    .Define("trk2_pz_resolution", "trk2_matched[2] - trk2MC[2]")
    ###############################################################################################
    .Filter("mcflag == 0 && mctruth != -1")
)

# --- PRZETWARZANIE MONTE CARLO (MC) ---
df = (
    df.Define("Kchrec_5", f"Kchrec[5] - {mK0}")
    .Define("Knerec_5", f"Knerec[5] - {mK0}")
    .Define("KnerecSix_5", f"KnerecSix[5] - {mK0}")
    .Define("KnetriKinFit_5", f"KnetriKinFit[5] - {mK0}")
    .Define("MassPi01", f"pi01[5] - {mPi0}")
    .Define("MassPi02", f"pi02[5] - {mPi0}")
    .Define("MassPi01Fit", f"pi01Fit[5] - {mPi0}")
    .Define("MassPi02Fit", f"pi02Fit[5] - {mPi0}")
    .Define("MassPi0NeutralFit1", f"pi0NeutralFit1[5] - {mPi0}")
    .Define("MassPi0NeutralFit2", f"pi0NeutralFit2[5] - {mPi0}")
    .Define("MassPi0PMFit1", f"pi0PMFit1[5] - {mPi0}")
    .Define("MassPi0PMFit2", f"pi0PMFit2[5] - {mPi0}")
    .Define("rho_ch", "calc_cylindrical_radius(Kchrec, ip)")
    .Define("rho_ne", "calc_cylindrical_radius(Knerec, ip)")
    .Define("z_ch", "calc_z_component(Kchrec, ip)")
    .Define("z_ne", "calc_z_component(Knerec, ip)")
    .Define("R_ch", "calc_spherical_radius(Kchrec, ip)")
    .Define("R_ne", "calc_spherical_radius(KnerecSix, ip)")
    .Define("neu_clu_num", "neutralClustersList.size()")
    ######################### Variables for Omega-pi0 geometrical cut ############################
    .Define("bhabha_IP", "ROOT::RVec<double>{Bx, By, Bz}")
    .Define("bhabha_IP_changed", "ROOT::RVec<double>{Bx, By, KchrecClosest[8]}")
    .Define("rho_00_closest", "calc_cylindrical_radius(Knerec, bhabha_IP_changed)")
    .Define("rho_pm_closest", "calc_cylindrical_radius(KchrecClosest, bhabha_IP)")
    .Define("rho_omega", "hypot(rho_00_closest, rho_pm_closest)")
    .Define("z_00_closest", "Knerec[8] - KchrecClosest[8]")
    .Define("z_pm_closest", "KchrecClosest[8] - Bz")
    ###############################################################################################
    ######################### Variables for bad tracks geometrical cut ############################
    .Define("rho_pm_00", "hypot(Kchrec[6] - Knerec[6], Kchrec[7] - Knerec[7])")
    .Define("z_pm_00", "Kchrec[8] - Knerec[8]")
    .Define("cos_trk1_K", "cosine_between_vectors(trk1, Kchrec)")
    .Define("cos_trk2_K", "cosine_between_vectors(trk2, Kchrec)")
    ###############################################################################################
    ######################## Delta t ##############################################################
    .Define("Delta_t_fit", "KaonChTimeCMSignalFit - KaonNeTimeCMSignalFit")
    .Define("Delta_t_MC", "KaonChTimeCMMC - KaonNeTimeCMMC")
    .Define("Delta_t_rec", "KaonChTimeCMBoostLor - KaonNeTimeCMBoostLor")
    .Define("Delta_t_tri_fit", "KaonChTimeCMBoostTriFit - KaonNeTimeCMBoostTriFit")
    ###############################################################################################
    ######################## Resolutions ##########################################################
    .Define("Delta_t_fit_resolution", "Delta_t_fit - Delta_t_MC")
    .Define("Delta_t_rec_resolution", "Delta_t_rec - Delta_t_MC")
    .Define("Delta_t_tri_fit_resolution", "Delta_t_tri_fit - Delta_t_MC")
    .Define("trk1_matched", "get_matched_vector(trk1, trk2, trk1MC, trk2MC)[0]")
    .Define("trk2_matched", "get_matched_vector(trk1, trk2, trk1MC, trk2MC)[1]")
    .Define("trk1_px_resolution", "trk1_matched[0] - trk1MC[0]")
    .Define("trk2_px_resolution", "trk2_matched[0] - trk2MC[0]")
    .Define("trk1_py_resolution", "trk1_matched[1] - trk1MC[1]")
    .Define("trk2_py_resolution", "trk2_matched[1] - trk2MC[1]")
    .Define("trk1_pz_resolution", "trk1_matched[2] - trk1MC[2]")
    .Define("trk2_pz_resolution", "trk2_matched[2] - trk2MC[2]")
    ###############################################################################################    
    .Filter("mcflag == 1 && mctruth != -1")
)

# df Signal
df_3pi0 = df.Filter(
    f"mcflag == 1 && mctruth == 4"
)

df_Background = df.Filter(
    f"mcflag == 1 && mctruth != 4 && mctruth != -1"
)

base_path = "/workspace/root_notebooks/results"
cache_path = "/workspace/analysis_snapshot"
analysis_dir = "ks_pm_control_sample"

current_date = time.strftime("%Y-%m-%d")

metadata = {
    "saved_errors": "Semileptonic",
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

tag_cut_list = ["neu_clu_num >= 6", 
                "bestErrorSixGamma < 2000.0", 
                "abs(Knerec_5 + 153.0798) < 120"]

########### Fiducial volume Omega-pi0 #########################
fiducial_volume_omega = "rho_pm_closest < 2.0 && rho_00_closest < 1.5 && abs(z_pm_closest) < 2.0 && abs(z_00_closest) < 2.0"
###############################################################
########### Fiducial volume tracks back to back ###############
fiducial_volume_close_tracks = "rho_pm_00 < 2.05 && abs(z_pm_00) < 2.45"
###############################################################

cut_list = ["abs(Kchrec_5) <= 3 * 0.8090",
            f"({fiducial_volume_omega} && rho_pm_closest > 1.0) || !({fiducial_volume_omega})",
            f"({fiducial_volume_close_tracks} && (abs(cos_trk1_K) < 0.8 || abs(cos_trk2_K) < 0.8) ) || !({fiducial_volume_close_tracks})"
            ]

full_cut_list = tag_cut_list + cut_list

print(full_cut_list)

############# FIRST CUT LIST FOR 000 TAG ################
from utils.hist_utils import fill_1D_histogram_simple

division_list = ["tag", "channels"]
mc_flags = {}

for division in division_list:
  if division == "tag":
    mc_flags = {"true": "Data", "has000": "has000", "!has000": "!has000"}
    file_base_name = "tag_kl_000"
  else:
    mc_flags = channName
    file_base_name = "tag_kl_000_chann_div"

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