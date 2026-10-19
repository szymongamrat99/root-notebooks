import ROOT
import concurrent.futures

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