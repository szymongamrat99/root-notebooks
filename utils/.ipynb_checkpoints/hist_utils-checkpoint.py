import ROOT
import ctypes

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