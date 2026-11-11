import ROOT
import ctypes
import json
import math
import os
import operator

from utils.general_utils import log_analysis_run


def split_variable_group(variables, titles, modes, is_truth_only):
  """Dzieli (variables, titles, modes) na grupę "główną" i "truth-only".

  variables/titles: listy wyrównane pozycyjnie (dla 2D - titles to pary).
  modes: dict {mode_name: [zakres_dla_var0, zakres_dla_var1, ...]} - zakresy
    muszą być wyrównane pozycyjnie z `variables`, tak jak w oryginalnym kodzie.
  is_truth_only: zbiór/lista nazw zmiennych do wydzielenia albo funkcja
    callable(var) -> bool (dla 2D zwykle lambda pair: "_".join(pair) in {...}).

  Zwraca ((vars_main, titles_main, modes_main), (vars_truth, titles_truth, modes_truth)).
  """
  if not callable(is_truth_only):
    truth_set = set(is_truth_only)
    is_truth_only = lambda v: v in truth_set

  idx_main = [i for i, v in enumerate(variables) if not is_truth_only(v)]
  idx_truth = [i for i, v in enumerate(variables) if is_truth_only(v)]

  def take(idxs):
    vs = [variables[i] for i in idxs]
    ts = [titles[i] for i in idxs]
    ms = {mode: [ranges[i] for i in idxs] for mode, ranges in modes.items()}
    return vs, ts, ms

  return take(idx_main), take(idx_truth)


def fill_1d_histograms(df, df_data, channels, variables, titles, modes, total_cut,
                        scaling_factor=1.0, compute_mc_sum=True):
  """Wypełnia histogramy 1D dla podanych kanałów i (opcjonalnie) dolicza "MC sum".

  channels: dict {mctruth: chann_name}, analogiczny do globalnego `channName`.
    Kanał o nazwie "Data" jest brany z df_data, pozostałe z df. Wpis o wartości
    "MC sum" jest ignorowany (sumę liczy ta funkcja, jeśli compute_mc_sum=True).
    Dla zmiennych widocznych tylko dla jednego mctruth (np. prawdziwe wartości
    z MC) wystarczy przekazać np. {1: channName[1]} i compute_mc_sum=False -
    wtedy nie powstają ani Data, ani MC sum dla tych zmiennych.

  Zwraca hist_1D: {var: {chann: {mode: hist}}}.
  """
  hist_1D = {}
  for i, var in enumerate(variables):
    hist_1D[var] = {}
    for mctruth, chann in channels.items():
      if chann == "MC sum":
        continue

      hist_1D[var][chann] = {}
      df_draw = df_data if chann == "Data" else df
      title_str = f";{titles[i]};Counts"
      for mode in modes:
        filter_expr = f"mctruth == {mctruth} && {total_cut}"
        hist_1D[var][chann][mode] = df_draw.Filter(filter_expr).Histo1D(
            (f"{var}_{chann}_{mode}", title_str, *modes[mode][i]), var
        )

  ROOT.RDF.RunGraphs([
      hist_1D[var][chann][mode]
      for var in variables for chann in hist_1D[var] for mode in modes
  ])

  if compute_mc_sum:
    mc_channels = [c for c in channels.values() if c not in ("Data", "MC sum")]
    if mc_channels:
      first_mc = mc_channels[0]
      for var in variables:
        hist_1D[var]["MC sum"] = {}
        for mode in modes:
          hist_1D[var][first_mc][mode].Scale(scaling_factor)
          hist_1D[var]["MC sum"][mode] = hist_1D[var][first_mc][mode].Clone()
          for chann in mc_channels[1:]:
            hist_1D[var][chann][mode].Scale(scaling_factor)
            hist_1D[var]["MC sum"][mode].Add(hist_1D[var][chann][mode].GetPtr())

  return hist_1D


def fill_2d_histograms(df, df_data, channels, variables_2d, titles_2d, modes, total_cut,
                        scaling_factor=1.0, compute_mc_sum=True):
  """Odpowiednik fill_1d_histograms dla histogramów 2D.

  variables_2d: lista par [var_x, var_y]. titles_2d: lista par [title_x, title_y].
  modes: dict {mode_name: [(nbinsX, xlow, xhigh, nbinsY, ylow, yhigh), ...]}.

  Zwraca hist_2D: {var_key: {chann: {mode: hist}}}, gdzie var_key = "x_y".
  """
  hist_2D = {}
  for idx, varset in enumerate(variables_2d):
    var_key = "_".join(varset)
    hist_2D[var_key] = {}
    title_str = f";{titles_2d[idx][0]};{titles_2d[idx][1]}"

    for mctruth, chann in channels.items():
      if chann == "MC sum":
        continue

      df_hist = df_data if chann == "Data" else df
      hist_2D[var_key][chann] = {}
      for mode, model_args in modes.items():
        filter_expr = f"mctruth == {mctruth} && {total_cut}"
        hist_2D[var_key][chann][mode] = df_hist.Filter(filter_expr).Histo2D(
            (f"{var_key}_{mode}_{chann}", title_str, *model_args[idx]),
            varset[0], varset[1]
        )

  all_hists = [
      hist_2D[var_key][chann][mode]
      for var_key in hist_2D for chann in hist_2D[var_key] for mode in modes
  ]
  ROOT.RDF.RunGraphs(all_hists)

  if compute_mc_sum:
    mc_channels = [c for c in channels.values() if c not in ("Data", "MC sum")]
    if mc_channels:
      first_mc = mc_channels[0]
      for idx, varset in enumerate(variables_2d):
        var_key = "_".join(varset)
        hist_2D[var_key]["MC sum"] = {}
        for mode in modes:
          hist_2D[var_key][first_mc][mode].Scale(scaling_factor)
          hist_2D[var_key]["MC sum"][mode] = hist_2D[var_key][first_mc][mode].Clone()
          for chann in mc_channels[1:]:
            hist_2D[var_key][chann][mode].Scale(scaling_factor)
            hist_2D[var_key]["MC sum"][mode].Add(hist_2D[var_key][chann][mode].GetPtr())

  return hist_2D


def _build_flag_channels(mc_flags, data_name, data_flag):
  """Zwraca {chann: (is_data, flag_expr)} dla kanałów MC (z flag) i Data."""
  chans = {chann: (False, flag) for chann, flag in mc_flags.items()}
  if data_name:
    chans = {data_name: (True, data_flag), **chans}
  return chans


def _flag_filter(flag, total_cut):
  parts = [p for p in (flag, total_cut) if p]
  return " && ".join(f"({p})" for p in parts) if parts else "true"


def _sum_mc_hists(per_chann, mc_names, scaling_factor, sum_name, sum_hist_name):
  """Skaluje histogramy MC i zapisuje ich sumę jako per_chann[sum_name]."""
  mc_hists = []
  for chann in mc_names:
    h = per_chann[chann]
    h = h.GetPtr() if hasattr(h, "GetPtr") else h
    h.Scale(scaling_factor)
    mc_hists.append(h)
  mc_sum = mc_hists[0].Clone(sum_hist_name)
  for h in mc_hists[1:]:
    mc_sum.Add(h)
  per_chann[sum_name] = mc_sum


def fill_1d_histograms_flags(df, df_data, mc_flags, variables, titles, modes, total_cut,
                              scaling_factor=1.0, compute_mc_sum=True,
                              data_name="Data", data_flag=None, mc_sum_name="MC sum"):
  """Wersja fill_1d_histograms dla próbek kontrolnych rozdzielanych flagami bool.

  Zamiast mctruth używa wyrażeń boolowskich (kolumn RDataFrame), np.
    mc_flags = {"has000": "has000", "no000": "!has000"}
  (klucz = nazwa kanału użyta w legendzie/kolorach/zapisie, wartość = wyrażenie
  filtrujące). Dostajemy po jednym histogramie MC na flagę, histogram danych
  (df_data, opcjonalnie z data_flag; domyślnie tylko total_cut) oraz "MC sum"
  jako sumę przeskalowanych (scaling_factor) histogramów MC.
  data_name=None wyłącza kanał danych.

  Zwraca hist_1D: {var: {chann: {mode: hist}}} - ta sama struktura, którą
  przyjmuje draw_and_save_1d_histograms (kolejność: Data, MC..., MC sum),
  więc wszystko trafia na jeden canvas i do jednego pliku ROOT.
  """
  chans = _build_flag_channels(mc_flags, data_name, data_flag)
  hist_1D = {}
  for i, var in enumerate(variables):
    hist_1D[var] = {}
    title_str = f";{titles[i]};Counts"
    for chann, (is_data, flag) in chans.items():
      df_draw = df_data if is_data else df
      filter_expr = _flag_filter(flag, total_cut)
      hist_1D[var][chann] = {
          mode: df_draw.Filter(filter_expr).Histo1D(
              (f"{var}_{chann}_{mode}", title_str, *modes[mode][i]), var)
          for mode in modes
      }

  ROOT.RDF.RunGraphs([
      hist_1D[var][chann][mode]
      for var in variables for chann in hist_1D[var] for mode in modes
  ])

  if compute_mc_sum and mc_flags:
    for var in variables:
      by_mode = {mode: {c: hist_1D[var][c][mode] for c in mc_flags} for mode in modes}
      for mode in modes:
        _sum_mc_hists(by_mode[mode], list(mc_flags), scaling_factor, mc_sum_name,
                      f"{var}_{mc_sum_name}_{mode}")
        hist_1D[var].setdefault(mc_sum_name, {})[mode] = by_mode[mode][mc_sum_name]

  return hist_1D


def fill_2d_histograms_flags(df, df_data, mc_flags, variables_2d, titles_2d, modes, total_cut,
                              scaling_factor=1.0, compute_mc_sum=True,
                              data_name="Data", data_flag=None, mc_sum_name="MC sum"):
  """Odpowiednik fill_1d_histograms_flags dla histogramów 2D.

  variables_2d: lista par [var_x, var_y]; titles_2d: lista par tytułów;
  modes: {mode: [(nbinsX, xlow, xhigh, nbinsY, ylow, yhigh), ...]}.

  Zwraca hist_2D: {"x_y": {chann: {mode: hist}}} (jak dla draw_and_save_2d_histograms).
  """
  chans = _build_flag_channels(mc_flags, data_name, data_flag)
  hist_2D = {}
  for idx, varset in enumerate(variables_2d):
    var_key = "_".join(varset)
    hist_2D[var_key] = {}
    title_str = f";{titles_2d[idx][0]};{titles_2d[idx][1]}"
    for chann, (is_data, flag) in chans.items():
      df_hist = df_data if is_data else df
      filter_expr = _flag_filter(flag, total_cut)
      hist_2D[var_key][chann] = {
          mode: df_hist.Filter(filter_expr).Histo2D(
              (f"{var_key}_{mode}_{chann}", title_str, *modes[mode][idx]),
              varset[0], varset[1])
          for mode in modes
      }

  ROOT.RDF.RunGraphs([
      hist_2D[k][chann][mode]
      for k in hist_2D for chann in hist_2D[k] for mode in modes
  ])

  if compute_mc_sum and mc_flags:
    for var_key in hist_2D:
      for mode in modes:
        by_chann = {c: hist_2D[var_key][c][mode] for c in mc_flags}
        _sum_mc_hists(by_chann, list(mc_flags), scaling_factor, mc_sum_name,
                      f"{var_key}_{mode}_{mc_sum_name}")
        hist_2D[var_key].setdefault(mc_sum_name, {})[mode] = by_chann[mc_sum_name]

  return hist_2D


def compute_channel_integrals_and_purity(hist_dict, modes,
                                          signal_channel="Signal", mc_sum_channel="MC sum"):
  """Liczy ważoną liczbę zdarzeń (całkę) na kanał oraz czystość próbki po cięciach.

  Całka liczona jest TYLKO z jednego histogramu (pierwszy klucz hist_dict,
  pierwszy tryb z `modes`) metodą Integral() z zakresem obejmującym
  underflow/overflow (h.Integral(0, nbins+1[, 0, nbinsY+1])) - to poprawnie
  uwzględnia zastosowane wagi i jest niezależne od zakresu/binningu
  konkretnego histogramu: dla danego kanału i zestawu cięć ta sama (ważona)
  liczba zdarzeń trafia do jakiegokolwiek histogramu tego kanału, więc
  liczenie więcej niż jednej zmiennej/trybu byłoby zbędnym powtórzeniem tego
  samego wyniku.

  hist_dict: {var: {chann: {mode: hist}}} (hist_1D albo hist_2D - dla 2D var
    to już połączony klucz "x_y", nie trzeba go podawać osobno).
  signal_channel/mc_sum_channel: nazwy kanałów użyte do obliczenia czystości
    jako Integral(signal_channel) / Integral(mc_sum_channel). Jeśli
    którykolwiek z nich nie występuje w hist_dict (np. dla zmiennych
    truth-only, gdzie nie liczy się "MC sum"), czystość nie jest zwracana.

  Zwraca (channel_integrals: {chann: float}, purity: float | None).
  """
  if not hist_dict:
    return {}, None

  first_var = next(iter(hist_dict))
  first_mode = next(iter(modes))
  channels = hist_dict[first_var]

  channel_integrals = {}
  for chann, by_mode in channels.items():
    h = by_mode[first_mode]
    h = h.GetPtr() if hasattr(h, "GetPtr") else h
    if h.InheritsFrom("TH2"):
      integral = h.Integral(0, h.GetNbinsX() + 1, 0, h.GetNbinsY() + 1)
    else:
      integral = h.Integral(0, h.GetNbinsX() + 1)
    channel_integrals[chann] = integral

  purity = None
  if signal_channel in channel_integrals and mc_sum_channel in channel_integrals:
    mc_sum_integral = channel_integrals[mc_sum_channel]
    if mc_sum_integral > 0:
      purity = channel_integrals[signal_channel] / mc_sum_integral

  return channel_integrals, purity


def _write_channel_summary(hash_dir, hist_dict, modes):
  """Zapisuje ChannelIntegrals (JSON) i - jeśli policzalne - Purity do hash_dir."""
  channel_integrals, purity = compute_channel_integrals_and_purity(hist_dict, modes)

  ROOT.TNamed("ChannelIntegrals", json.dumps(channel_integrals, indent=2)).Write(
      "ChannelIntegrals", ROOT.TObject.kOverwrite
  )
  if purity is not None:
    ROOT.TParameter("double")("Purity", purity).Write("Purity", ROOT.TObject.kOverwrite)


def draw_and_save_1d_histograms(hist_1D, variables, modes, channColor, channTitle,
                                 current_result_dir, cut_hash, cuts_json_str, cut_list,
                                 root_file_name, logy_states=(False, True)):
  """Rysuje, zapisuje do pliku ROOT/PDF i loguje przebieg dla histogramów 1D.

  Kolejność rysowanych kanałów wynika z kluczy hist_1D (np. ["Data", MC..., "MC sum"]
  dla pełnego zestawu albo [channName[1]] dla zmiennych truth-only) - nie trzeba
  jej podawać osobno.

  root_file_name to sama nazwa pliku (bez hasha) - ten sam plik ROOT jest
  współdzielony przez różne zestawy cięć. Plik otwierany jest w trybie UPDATE
  (tworzy go, jeśli jeszcze nie istnieje), a dane trafiają do TDirectory
  o nazwie cut_hash wewnątrz pliku, dzięki czemu różne zestawy cięć nie
  nadpisują się nawzajem mimo identycznej nazwy pliku.

  Zwraca (hist_1D, canvas, legends).
  """
  ROOT.TGaxis.SetMaxDigits(4)
  ROOT.gStyle.SetOptStat(0)

  canvas = []
  legends = []
  mode_list = list(modes.keys())
  draw_channels = list(next(iter(hist_1D.values())).keys()) if hist_1D else []

  os.makedirs(current_result_dir, exist_ok=True)
  output_root_file = ROOT.TFile(f"{current_result_dir}/{root_file_name}", "UPDATE")

  hash_dir = output_root_file.Get(cut_hash)
  if not hash_dir:
    hash_dir = output_root_file.mkdir(cut_hash)
  hash_dir.cd()

  ROOT.TNamed("CutList", cuts_json_str).Write("CutList", ROOT.TObject.kOverwrite)
  ROOT.TNamed("CutHash", cut_hash).Write("CutHash", ROOT.TObject.kOverwrite)

  _write_channel_summary(hash_dir, hist_1D, modes)

  ### Get Max Channel per mode per var for scaling purposes
  max_per_mode = {var: {mode: 0 for mode in mode_list} for var in variables}
  for var in variables:
    for mode in mode_list:
      for chann in draw_channels:
        h_curr = hist_1D[var][chann][mode]
        h_curr = h_curr.GetPtr() if hasattr(h_curr, "GetPtr") else h_curr
        max_per_mode[var][mode] = max(max_per_mode[var][mode], h_curr.GetBinContent(h_curr.GetMaximumBin()))

  for logy in logy_states:
    for i, var in enumerate(variables):

      var_dir = hash_dir.Get(var)
      if not var_dir:
        var_dir = hash_dir.mkdir(var)
      var_dir.cd()

      legend = ROOT.TLegend(0.2, 0.7, 0.83, 0.88)
      legend.SetBorderSize(0)
      legend.SetFillStyle(0)  # Przezroczyste tło
      legend.SetNColumns(2)
      legend.SetTextSize(0.03)

      log_suffix = "log" if logy else "lin"
      c_name = f"c_{var}_{i}_{log_suffix}"
      canvas.append(ROOT.TCanvas(c_name, c_name, 800, 400))
      canvas[-1].Divide(len(mode_list), 1)

      for j, mode in enumerate(mode_list):
        canvas[-1].cd(j + 1)
        ROOT.gPad.SetLogy(logy)

        # max(..., 1) zabezpiecza przed zdegenerowanym zakresem osi Y,
        # gdy histogram (np. bez cięć) jest pusty i GetMaximum() == 0.
        y_max = max(max_per_mode[var][mode], 1)

        for k, chann in enumerate(draw_channels):
          h_curr = hist_1D[var][chann][mode]
          h_curr = h_curr.GetPtr() if hasattr(h_curr, "GetPtr") else h_curr
          h_curr.SetLineColor(channColor[chann])

          if not logy:
            h_curr.SetMinimum(0)
            h_curr.GetYaxis().SetRangeUser(0, y_max * 1.2)
            h_curr.Write(f"{var}_{chann}_{mode}", ROOT.TObject.kOverwrite)
          else:
            h_curr.SetMinimum(1)
            h_curr.GetYaxis().SetRangeUser(1, y_max * 100)

          mc_data_draw_style = "PE1" if chann == "Data" else "HIST"

          if k == 0:
            h_curr.Draw(mc_data_draw_style)
          else:
            h_curr.Draw(f"{mc_data_draw_style} SAME")

          if j == 0:
            entry_style = "l" if chann != "Data" else "pe1"
            legend.AddEntry(h_curr, channTitle[chann], entry_style)

        legend.Draw()
        legends.append(legend)

      canvas[-1].Draw()

      os.makedirs(f"{current_result_dir}/img/{cut_hash}", exist_ok=True)
      canvas[-1].SaveAs(f"{current_result_dir}/img/{cut_hash}/{var}_{i}_{log_suffix}_{cut_hash}.pdf")

  output_root_file.Close()

  log_analysis_run(current_result_dir, cut_hash, root_file_name, cut_list)

  return hist_1D, canvas, legends


def draw_and_save_2d_histograms(hist_2D, variables_2d, modes, current_result_dir, cut_hash,
                                 cuts_json_str, cut_list, root_file_name):
  """Rysuje, zapisuje do pliku ROOT/PDF i loguje przebieg dla histogramów 2D.

  Kolejność rysowanych kanałów wynika z kluczy hist_2D - tak samo jak
  w draw_and_save_1d_histograms.

  root_file_name to sama nazwa pliku (bez hasha) - ten sam plik ROOT jest
  współdzielony przez różne zestawy cięć. Plik otwierany jest w trybie UPDATE
  (tworzy go, jeśli jeszcze nie istnieje), a dane trafiają do TDirectory
  o nazwie cut_hash wewnątrz pliku, dzięki czemu różne zestawy cięć nie
  nadpisują się nawzajem mimo identycznej nazwy pliku.

  Zwraca (hist_2D, canva).
  """
  ROOT.TGaxis.SetMaxDigits(6)
  canva = []
  mode_list = list(modes.keys())
  draw_channels = list(next(iter(hist_2D.values())).keys()) if hist_2D else []

  os.makedirs(current_result_dir, exist_ok=True)
  output_root_file = ROOT.TFile(f"{current_result_dir}/{root_file_name}", "UPDATE")

  hash_dir = output_root_file.Get(cut_hash)
  if not hash_dir:
    hash_dir = output_root_file.mkdir(cut_hash)
  hash_dir.cd()

  ROOT.TNamed("CutList", cuts_json_str).Write("CutList", ROOT.TObject.kOverwrite)
  ROOT.TNamed("CutHash", cut_hash).Write("CutHash", ROOT.TObject.kOverwrite)

  _write_channel_summary(hash_dir, hist_2D, modes)

  for i, varset in enumerate(variables_2d):
    var_key = "_".join(varset)
    var_dir = hash_dir.Get(var_key)
    if not var_dir:
      var_dir = hash_dir.mkdir(var_key)
    var_dir.cd()

    for chann in draw_channels:
      canva.append(ROOT.TCanvas(f"c2d_{i}_{chann}", "2D Histograms", 800, 400))
      canva[-1].Divide(len(mode_list), 1)

      for j, mode in enumerate(mode_list):
        canva[-1].cd(j + 1)
        h_curr = hist_2D[var_key][chann][mode]
        h_curr.Draw("COLZ")
        # Nazwa zapisu MUSI być budowana jawnie (nie h_curr.GetName()):
        # "MC sum" powstaje przez Clone() z pierwszego kanału MC, a Clone()
        # bez podania nowej nazwy zachowuje oryginalną nazwę źródła (np.
        # "Signal") - zapis pod h_curr.GetName() nadpisywałby wtedy wpis
        # Signal danymi MC sum zamiast utworzyć dla niego osobny klucz.
        h_curr.Write(f"{var_key}_{mode}_{chann}", ROOT.TObject.kOverwrite)

      canva[-1].Draw()

      os.makedirs(f"{current_result_dir}/img/{cut_hash}", exist_ok=True)
      canva[-1].SaveAs(f"{current_result_dir}/img/{cut_hash}/{var_key}_{chann}_{cut_hash}.pdf")

  output_root_file.Close()

  log_analysis_run(current_result_dir, cut_hash, root_file_name, cut_list)

  return hist_2D, canva


def get_highest_channels(histos):
    """
    Znajduje nazwę najwyższego kanału dla każdego indeksu histogramu (pada).

    Parameters:
    -----------
    histos : dict
        Słownik w formacie {chann_name: [hist1, hist2, ...]} lub {chann_name: hist1}

    Returns:
    --------
    list[str]
        Lista nazw kanałów o największym Y max dla kolejnych padów (pozycji w liście).
        Przykład: ['Data', 'Signal'] -> 'Data' najwyższa na padzie 0, 'Signal' na padzie 1.
    """
    if not histos:
        return []

    # Standaryzacja słownika: klucze na str, wartości na listy
    formatted_dict = {}
    for k, v in histos.items():
        clean_key = str(k).strip()
        formatted_dict[clean_key] = v if isinstance(v, (list, tuple)) else [v]

    # Liczba padów zależy od długości listy pierwszego kanału
    num_pads = len(next(iter(formatted_dict.values())))
    highest_channels = []

    # DLA KAZDEGO PADA (pozycja i):
    for pad_idx in range(num_pads):
        top_chann = None
        max_y = -float('inf')

        for chann, h_list in formatted_dict.items():
            hist_ref = h_list[pad_idx]
            # Obsługa RResultPtr z RDataFrame lub zwykłego TH1
            h = hist_ref.GetValue() if hasattr(hist_ref, "GetValue") else hist_ref
            
            y_max = h.GetMaximum()
            if y_max > max_y:
                max_y = y_max
                top_chann = chann

        highest_channels.append(top_chann)

    return highest_channels

def fit_and_scale_mc(h_data, h_pm, h_nonpm):
    """Funkcja wykonująca TFractionFitter dla dwóch komponentów MC względem

    danych.

    Zwraca przeskalowane kopie histogramów oraz wykres dopasowanej sumy MC.
    """
    mc_array = ROOT.TObjArray(2)

    # Klonujemy histogramy, żeby nie modyfikować surowych danych wejściowych
    h_pm_fit = h_pm.Clone(f"{h_pm.GetName()}_fit")
    h_nonpm_fit = h_nonpm.Clone(f"{h_nonpm.GetName()}_fit")

    mc_array.Add(h_pm_fit)
    mc_array.Add(h_nonpm_fit)

    fitter = ROOT.TFractionFitter(h_data, mc_array)

    # Jeśli zależy Ci na zwolnieniu więzów sumy frakcji (suma != 1):
    # Dostęp do wewnętrznego fitera w ROOT uzyskuje się przez GetFitter():
    # internal_fitter = fitter.GetFitter()
    # Usunięto fitter.UnconstrainFit(), ponieważ nie występuje w tej klasie w ROOT.

    status = fitter.Fit()

    if int(status) == 0:
        # W Pythonie 3 używamy ctypes.c_double zamiast wycofanego ROOT.Double
        val_pm, err_pm = ctypes.c_double(0), ctypes.c_double(0)
        val_nonpm, err_nonpm = ctypes.c_double(0), ctypes.c_double(0)

        fitter.GetResult(0, val_pm, err_pm)
        fitter.GetResult(1, val_nonpm, err_nonpm)

        v_pm = val_pm.value
        v_nonpm = val_nonpm.value

        n_data = h_data.Integral()

        # Przeskalowanie surowych histogramów do dopasowanych zliczeń
        if h_pm_fit.Integral() > 0:
            h_pm_fit.Scale((v_pm * n_data) / h_pm_fit.Integral())
        if h_nonpm_fit.Integral() > 0:
            h_nonpm_fit.Scale((v_nonpm * n_data) / h_nonpm_fit.Integral())

        # Pobranie wykresu dopasowanej sumy MC
        h_sum_fit = fitter.GetPlot()
        return h_pm_fit, h_nonpm_fit, h_sum_fit, (v_pm, v_nonpm)
    else:
        print("TFractionFitter: brak zbieżności fitu!")
        return h_pm_fit, h_nonpm_fit, None, (0.0, 0.0)

def _histogram_model_for_cut(model, cut_hash):
  if cut_hash not in model:
    raise KeyError(f"Brak modelu histogramów dla cut_hash={cut_hash!r}")
  cut_model = model[cut_hash]
  if not cut_model:
    raise ValueError(f"Pusty model histogramów dla cut_hash={cut_hash!r}")
  return cut_model


def _histogram_axis(model_entry, variable):
  try:
    n_bins = operator.index(model_entry["n_bins"])
    x_min, x_max = (float(bound) for bound in model_entry["range_x"])
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError(
        f"Niepoprawny model osi histogramu dla zmiennej {variable!r}"
    ) from error

  if (
      n_bins <= 0
      or not math.isfinite(x_min)
      or not math.isfinite(x_max)
      or x_min >= x_max
  ):
    raise ValueError(
        f"Niepoprawny model osi histogramu dla zmiennej {variable!r}: "
        f"n_bins={n_bins}, range_x={[x_min, x_max]}"
    )
  return n_bins, x_min, x_max


def _model_variables(cut_model):
  return list(dict.fromkeys(
      variable
      for variables_by_mode in cut_model.values()
      for variable in variables_by_mode
  ))


def _model_mc_sum(histograms, channels, scaling_factor, sum_name, hist_name):
  if not channels or any(channel not in histograms for channel in channels):
    return None

  channel_hists = {channel: histograms[channel] for channel in channels}
  _sum_mc_hists(channel_hists, channels, scaling_factor, sum_name, hist_name)
  return channel_hists[sum_name]


def fill_1D_histogram_simple(
    df, df_data, channels, model, cut_hash, total_cut,
    variables=None, scaling_factor=1.0, compute_mc_sum=True, division="channels"
):
  """Wypełnia histogramy 1D używając modelu {cut_hash: {mode: {variable: ...}}}.

  `channels` ma format {mctruth: channel_name}; kanał "Data" korzysta z df_data.
  Jeśli `variables` jest pominięte, funkcja bierze zmienne bezpośrednio z modelu.
  Zwraca {variable: {channel: {mode: histogram}}}.
  """
  cut_model = _histogram_model_for_cut(model, cut_hash)
  variables = _model_variables(cut_model) if variables is None else list(variables)
  unknown_variables = set(variables) - set(_model_variables(cut_model))
  if unknown_variables:
    raise KeyError(f"Zmienne nieobecne w modelu: {sorted(unknown_variables)}")

  hist_1D = {}
  pending = []
  for variable in variables:
    hist_1D[variable] = {}
    for mctruth, channel in channels.items():
      if channel == "MC sum":
        continue

      df_draw = df_data if channel == "Data" else df
      hist_1D[variable][channel] = {}
      for mode, variables_by_mode in cut_model.items():
        entry = variables_by_mode.get(variable)
        if entry is None:
          continue

        n_bins, x_min, x_max = _histogram_axis(entry, variable)
        title = entry.get("title") or variable
        filter = f"mctruth == {mctruth} && {total_cut}" if division == "channels" else f"{mctruth} && {total_cut}"
        histogram = df_draw.Filter(
            filter
        ).Histo1D(
            (f"{variable}_{channel}_{mode}", f";{title};Counts",
             n_bins, x_min, x_max),
            variable,
        )
        hist_1D[variable][channel][mode] = histogram
        pending.append(histogram)

  if pending:
    ROOT.RDF.RunGraphs(pending)

  mc_channels = list(dict.fromkeys(
      channel for channel in channels.values()
      if channel not in ("Data", "MC sum")
  ))
  if compute_mc_sum and mc_channels:
    for variable, per_channel in hist_1D.items():
      per_channel["MC sum"] = {}
      for mode in cut_model:
        if any(mode not in per_channel.get(channel, {}) for channel in mc_channels):
          continue
        summed = _model_mc_sum(
            {channel: per_channel[channel][mode] for channel in mc_channels},
            mc_channels, scaling_factor, "MC sum",
            f"{variable}_MC_sum_{mode}",
        )
        if summed is not None:
          per_channel["MC sum"][mode] = summed

  return hist_1D


def fill_2D_histogram_simple(
    df, df_data, channels, model, cut_hash, variables_2d, total_cut,
    scaling_factor=1.0, compute_mc_sum=True
):
  """Wypełnia histogramy 2D, używając osi zmiennych z modelu 1D.

  `variables_2d` to lista par (x, y). Dla każdej osi liczba binów i zakres
  pochodzą z wpisu zmiennej w modelu wskazanego przez `cut_hash` i `mode`.
  Zwraca {"x_y": {channel: {mode: histogram}}}.
  """
  cut_model = _histogram_model_for_cut(model, cut_hash)
  available_variables = set(_model_variables(cut_model))
  hist_2D = {}
  pending = []

  for variable_pair in variables_2d:
    if len(variable_pair) != 2:
      raise ValueError(f"Zmienna 2D musi być parą (x, y): {variable_pair!r}")
    var_x, var_y = variable_pair
    missing = {var_x, var_y} - available_variables
    if missing:
      raise KeyError(f"Zmienne nieobecne w modelu: {sorted(missing)}")

    var_key = f"{var_x}_{var_y}"
    hist_2D[var_key] = {}
    for mctruth, channel in channels.items():
      if channel == "MC sum":
        continue

      df_draw = df_data if channel == "Data" else df
      hist_2D[var_key][channel] = {}
      for mode, variables_by_mode in cut_model.items():
        entry_x = variables_by_mode.get(var_x)
        entry_y = variables_by_mode.get(var_y)
        if entry_x is None or entry_y is None:
          continue

        n_x, x_min, x_max = _histogram_axis(entry_x, var_x)
        n_y, y_min, y_max = _histogram_axis(entry_y, var_y)
        title_x = entry_x.get("title") or var_x
        title_y = entry_y.get("title") or var_y
        histogram = df_draw.Filter(
            f"mctruth == {mctruth} && {total_cut}"
        ).Histo2D(
            (f"{var_key}_{channel}_{mode}",
             f";{title_x};{title_y}",
             n_x, x_min, x_max, n_y, y_min, y_max),
            var_x, var_y,
        )
        hist_2D[var_key][channel][mode] = histogram
        pending.append(histogram)

  if pending:
    ROOT.RDF.RunGraphs(pending)

  mc_channels = list(dict.fromkeys(
      channel for channel in channels.values()
      if channel not in ("Data", "MC sum")
  ))
  if compute_mc_sum and mc_channels:
    for var_key, per_channel in hist_2D.items():
      per_channel["MC sum"] = {}
      for mode in cut_model:
        if any(mode not in per_channel.get(channel, {}) for channel in mc_channels):
          continue
        summed = _model_mc_sum(
            {channel: per_channel[channel][mode] for channel in mc_channels},
            mc_channels, scaling_factor, "MC sum",
            f"{var_key}_MC_sum_{mode}",
        )
        if summed is not None:
          per_channel["MC sum"][mode] = summed

  return hist_2D


def fill_1d_histograms_simple(
    df, df_data, channels, model, total_cut, scaling_factor=1.0,
    compute_mc_sum=True, cut_hash=None, variables=None
):
  """Zgodnościowa nazwa dla fill_1D_histogram_simple."""
  if cut_hash is None:
    if len(model) != 1:
      raise ValueError("Podaj cut_hash, gdy model zawiera więcej niż jeden hash")
    cut_hash = next(iter(model))
  return fill_1D_histogram_simple(
      df, df_data, channels, model, cut_hash, total_cut, variables,
      scaling_factor, compute_mc_sum,
  )


def fill_2d_histograms_simple(
    df, df_data, channels, model, cut_hash, variables_2d, total_cut,
    scaling_factor=1.0, compute_mc_sum=True
):
  """Zgodnościowa nazwa dla fill_2D_histogram_simple."""
  return fill_2D_histogram_simple(
      df, df_data, channels, model, cut_hash, variables_2d, total_cut,
      scaling_factor, compute_mc_sum,
  )