import yaml

import ROOT
import concurrent.futures
import csv
import os

def check_single_file(f):
    """Sprawdza pojedynczy plik na osobnym wątku."""
    try:
        # kError ignoruje niepotrzebne ostrzeżenia w konsoli
        ROOT.gErrorIgnoreLevel = ROOT.kError
        tf = ROOT.TFile.Open(f, "READ")
        if tf and not tf.IsZombie():
            tree = tf.Get("h1")
            if tree and not tree.IsZombie():
                tf.Close()
                return f
        if tf:
            tf.Close()
    except Exception:
        pass
    return None


def get_valid_files_fast(file_list):
    """Sprawdza całą listę plików równolegle na wielu rdzeniach CPU."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        results = list(executor.map(check_single_file, file_list))
    # Zwraca tylko poprawne pliki
    valid = [f for f in results if f is not None]
    print(
        f"Przetestowano {len(file_list)} plików. Poprawnych: {len(valid)}, Odrzuconych: {len(file_list) - len(valid)}"
    )
    return valid


def split_root_file(path, tree_name="h1", max_bytes=50 * 1024**3, columns=None):
    """Dzieli plik ROOT z drzewem na części o rozmiarze <= ~max_bytes.

    RDataFrame::Snapshot nie respektuje TTree::SetMaxTreeSize, więc podział robimy
    po zapisie: średni rozmiar zdarzenia = rozmiar pliku / liczba wpisów, z tego
    liczba wpisów na część, a części (path z końcówką _0, _1, ...) zapisywane są
    jednym przebiegiem po pliku źródłowym. Plik źródłowy jest usuwany po
    pomyślnym podziale. Jeśli plik mieści się w limicie, zostaje bez zmian.

    Zwraca listę ścieżek (do wczytania np. ROOT.RDataFrame(tree_name, lista)).
    """
    size = os.path.getsize(path)
    if size <= max_bytes:
        return [path]

    n_entries = ROOT.RDataFrame(tree_name, path).Count().GetValue()
    per_part = max(1, int(n_entries * max_bytes / size * 0.95))  # 5% zapasu
    n_parts = -(-n_entries // per_part)

    base, ext = os.path.splitext(path)
    df = ROOT.RDataFrame(tree_name, path)
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fLazy = True
    cols = list(columns) if columns else [str(c) for c in df.GetColumnNames()]

    parts, snaps = [], []
    for k in range(n_parts):
        part_path = f"{base}_{k}{ext}"
        parts.append(part_path)
        lo, hi = k * per_part, (k + 1) * per_part
        snaps.append(
            df.Filter(f"rdfentry_ >= {lo}ULL && rdfentry_ < {hi}ULL")
              .Snapshot(tree_name, part_path, cols, opts)
        )
    ROOT.RDF.RunGraphs(snaps)

    os.remove(path)
    return parts


def build_cut_expression(cut_list):
  """Łączy listę cięć w jedno wyrażenie logiczne dla RDataFrame.Filter().

  Dla pustej listy zwraca "true", dzięki czemu Filter() nigdy nie dostaje
  pustego stringa ani samych nawiasów "()" (co ROOT zgłasza jako błąd
  składni) - pozwala to sprawdzić histogramy/cut-flow bez żadnych cięć.
  """
  if not cut_list:
    return "true"
  return " && ".join(f"({cut})" for cut in cut_list)


def _load_registry(manifest_file):
  """Wczytuje istniejący rejestr jako dict {hash: {"cuts_description": str, "filenames": set}}.

  Obsługuje też stary format pliku (kolumna "filename" w liczbie pojedynczej,
  możliwe wielokrotne wiersze dla tego samego hasha) - grupuje je po hashu
  i scala nazwy plików, co migruje rejestr do nowego formatu przy pierwszym
  kolejnym zapisie.
  """
  registry = {}
  if not os.path.isfile(manifest_file):
    return registry

  with open(manifest_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    filename_field = "filenames" if "filenames" in (reader.fieldnames or []) else "filename"
    for row in reader:
      h = row["hash"]
      entry = registry.setdefault(h, {"cuts_description": "", "filenames": set()})
      if row.get("cuts_description"):
        entry["cuts_description"] = row["cuts_description"]
      for name in row.get(filename_field, "").split(";"):
        name = name.strip()
        if name:
          entry["filenames"].add(name)

  return registry


def log_analysis_run(result_dir, cut_hash, output_filename, cut_list):
  """Zapisuje/aktualizuje powiązanie hash <-> lista cięć w rejestrze CSV.

  Rejestr trzyma JEDEN wiersz na unikalny cut_hash (deduplikacja) - kolejne
  wywołania z tym samym hashem (np. osobne pliki 1D/2D/truth dla tego samego
  zestawu cięć albo ponowne uruchomienie notebooka) tylko dopisują nazwę
  pliku do istniejącego wpisu zamiast duplikować wiersz.
  """
  # Upewniamy się, że katalog istnieje
  os.makedirs(result_dir, exist_ok=True)

  manifest_file = os.path.join(result_dir, "analysis_registry.csv")
  registry = _load_registry(manifest_file)

  cuts_description = " && ".join(cut_list) if cut_list else "(brak cięć)"
  entry = registry.setdefault(cut_hash, {"cuts_description": cuts_description, "filenames": set()})
  entry["cuts_description"] = cuts_description
  entry["filenames"].add(output_filename)

  fieldnames = ["hash", "filenames", "cuts_description"]
  with open(manifest_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for h, data in registry.items():
      writer.writerow({
          "hash": h,
          "filenames": ";".join(sorted(data["filenames"])),
          "cuts_description": data["cuts_description"],
      })

  print(
      f"[INFO] Zaktualizowano wpis dla hashu {cut_hash} w rejestrze w"
      f" {result_dir}/analysis_registry.csv"
  )

# Yaml config
# --- 2. Scalanie danych w strukturę wygodną do pliku konfiguracyjnego ---

def histogram_1D_config_creator(variables_1D: list, variables_1D_titles: list, unit_1D: list, range_X_1D: list[tuple], TRUTH_MCTRUTH_ID_1D: int, TRUTH_ONLY_VARS_1D: list, output_filename: str = "histograms_config.yaml"):
  import yaml
  config_data = {
      "truth_mctruth_id_1d": TRUTH_MCTRUTH_ID_1D,
      "truth_only_vars_1d": list(TRUTH_ONLY_VARS_1D),
      "histograms_1d": {}
  }

  for name, title, unit, rng in zip(variables_1D, variables_1D_titles, unit_1D, range_X_1D):
      hist_entry = {
          "title": title,
          "unit": unit,
          # Używamy jawnej konwersji na listę, ale yaml.dump potrafi zapisać małe listy w jednej linii,
          # jeśli użyjemy tricku z niestandardowym reprezentantem lub dopiszemy to w odpowiedni sposób.
          "range_x": list(rng)
      }
      config_data["histograms_1d"][name] = hist_entry

  # --- 3. Zapis do pliku YAML ---
  with open(output_filename, "w", encoding="utf-8") as f:
      # allow_unicode=True sprawia, że symbole ROOTa (np. greckie litery) zapiszą się czytelnie
      # sort_keys=False zachowuje kolejność kluczy
      yaml.dump(config_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

  print(f"Pomyślnie wygenerowano plik konfiguracyjny YAML: {output_filename}")

def save_resolutions_to_yaml(fit_results_dict: dict, output_filename: str = "resolutions_config.yaml"):
    """
    Zapisuje wyznaczone parametry rozdzielczości i fitów do osobnego pliku YAML.
    
    Oczekiwany format słownika fit_results_dict:
    {
        "NazwaZmiennej": {
            "sigma_core": (val, err),
            "sigma_tail": (val, err),
            "mean": (val, err),
            "amplitude_ratio": (val, err),
            "chi2_ndf": float_val,
            "fit_status": int_val
        },
        ...
    }
    """
    config_data = {
        "variables": fit_results_dict
    }

    # Zapis do pliku YAML
    with open(output_filename, "w", encoding="utf-8") as f:
        yaml.dump(
            config_data, 
            f, 
            allow_unicode=True, 
            sort_keys=False, 
            default_flow_style=None # Pozwala na ładne formatowanie list [wartość, niepewność] w jednej linii
        )

    print(f"Pomyślnie zrzucono wyniki rozdzielczości do pliku: {output_filename}")