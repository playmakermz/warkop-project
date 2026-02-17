# Script Awal pembuatan : 25-08-2025
# Refactored for readability and maintainability.
# Algorithm dan behavior 100% sama dengan versi original.

import os
import time
from math import ceil, log
from datetime import datetime

import numpy as np
import pandas as pd
import multiprocessing as mp

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     KONFIGURASI — Cukup modifikasi di sini!                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CONFIG = {
    # Nama game untuk label di log dan tampilan
    "game_name": "System 02 - HSR Scharacter",

    # Probabilitas jackpot per pull
    #   - 0.0006 untuk Character
    #   - 0.0008 untuk Light Cone
    "probability": 0.0006,

    # Minimum percobaan (target jarak) sebelum masuk fase lambat
    "min_percobaan": 21_500,

    # Batch size besar untuk fase cepat (automatic pull)
    "batch_size_fast": 600,

    # Batch size kecil untuk fase lambat & pull interaktif
    #   (tidak boleh lebih dari 10 pada mode tertentu)
    "batch_size_slow": 300,

    # Metode pull:  "NO" = Normal,  "MP" = Multiprocess
    "pull_method": "NO",

    # Confidence level target untuk berhenti (0.999 = 99.9%, 0.9999 = 99.99%)
    "confidence_target": 0.999,

    # Interval (detik) antara setiap progress log di layar
    "log_interval": 10,

    # Batch size untuk single interactive pull (opsi 1)
    "batch_size_single": 1,

    # Aktifkan opsi 3 (automatic pull) di menu interaktif
    "enable_auto_pull_menu": True,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              UTILITY FUNCTIONS                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def clear_screen():
    """Bersihkan layar terminal (Windows / macOS / Linux)."""
    os.system("cls" if os.name == "nt" else "clear")


def log_to_file(message: str, filename: str = "jackpot.txt"):
    """Tulis satu baris ke file log."""
    with open(filename, "a") as f:
        f.write(message)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          PREDIKSI MLE (Geometric)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def predict_next_jackpot_mle(jackpot_distances: list, confidence_target: float, jarak_jackpot: int = 0) -> dict | None:
    """
    Prediksi jarak pull sampai jackpot berikutnya menggunakan MLE
    pada distribusi geometric, berdasarkan data jarak jackpot sebelumnya.

    Returns dict berisi semua prediksi percentile, atau None jika data kosong.
    """
    data = [int(k) for k in jackpot_distances if isinstance(k, (int, np.integer)) and k > 0]
    if not data:
        print("❌ Data jackpot kosong, tidak bisa prediksi.")
        return None

    # MLE estimate: p_hat = 1 / rata-rata jarak
    mean_k = float(np.mean(data))
    p_hat = 1.0 / mean_k

    # Helper: hitung percentile dari geometric distribution
    def percentile_pulls(confidence: float) -> int:
        return ceil(log(1 - confidence) / log(1 - p_hat))

    # Hitung semua level prediksi
    preds = {
        "p_hat":       p_hat,
        "mean_pred":   int(round(mean_k)),
        "median_pred": percentile_pulls(0.50),
        "p90_pred":    percentile_pulls(0.90),
        "p95_pred":    percentile_pulls(0.95),
        "p98_pred":    percentile_pulls(0.98),
        "p99_pred":    percentile_pulls(0.99),
        "p999_pred":   percentile_pulls(0.999),
        "p100_pred":   percentile_pulls(confidence_target),  # target utama
        "p101_pred":   percentile_pulls(0.99999),
        "p102_pred":   percentile_pulls(0.99999),
    }

    # Tampilkan ringkasan
    print("\n🎯 Prediksi Jackpot Berikutnya (MLE):")
    print(f"- p̂ (peluang jackpot per pull): {p_hat:.6f} ({p_hat * 100:.4f}%)")
    print(f"- Rata-rata pulls sampai jackpot berikutnya : {preds['mean_pred']:,}")
    print(f"- Median pulls (50% kasus)                : {preds['median_pred']:,}")
    print(f"- 90% kemungkinan ≤                        : {preds['p90_pred']:,}")
    print(f"- 95% kemungkinan ≤                        : {preds['p95_pred']:,}")
    print(f"- 98% kemungkinan ≤                        : {preds['p98_pred']:,}")
    print(f"- 99% kemungkinan ≤                        : {preds['p99_pred']:,}")
    print(f"- 99.09% kemungkinan ≤                       : {preds['p999_pred']:,}")
    print(f"- 99.99% kemungkinan ≤                    ------>    : {preds['p100_pred']:,}")
    print(f"- 99.999% kemungkinan ≤                       : {preds['p101_pred']:,}  ---- Kurang : + {jarak_jackpot - preds['p101_pred']}")
    print(f"- 99.9999% kemungkinan ≤                       : {preds['p102_pred']:,}  ---- Kurang : + {jarak_jackpot - preds['p102_pred']}")
    print(f"p100 : {preds['p100_pred']:,}")

    return preds


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          NUMBA JIT (Opsional)                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if NUMBA_AVAILABLE:
    @jit
    def simulate_batches_numba(prob, batch_size, target, start_streak):
        """
        Fast JIT-compiled loop: pull dalam batch sampai streak >= target.
        Returns (total_pulls_done, list_jackpot_distances, final_streak).
        """
        pulls_done = 0
        jackpots = []
        streak = 0

        while True:
            if streak >= (target - 10):
                break

            pulls = np.random.random(size=batch_size)
            hits = np.where(pulls < prob)[0]

            if len(hits) > 0:
                first_hit = hits[0] + 1
                pulls_done += first_hit
                streak += first_hit
                jackpots.append(streak)
                streak = 0
            else:
                pulls_done += batch_size
                streak += batch_size

        return pulls_done, jackpots, streak


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        MULTIPROCESSING WORKER                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _mp_worker(args):
    """
    Worker process: simulasi pull dan kirim hasil jackpot via Queue.
    Berhenti ketika streak >= stop_value.
    """
    prob, batch_size, seed, stop_value, queue = args
    np.random.seed(seed)

    pulls_done = 0
    jackpots = []
    streak = 0

    while True:
        if stop_value.value > 0 and streak >= stop_value.value:
            break

        vals = np.random.random(batch_size)
        hits = np.where(vals < prob)[0]

        if len(hits) > 0:
            first_hit = hits[0] + 1
            streak += first_hit
            pulls_done += first_hit
            jackpots.append(streak)
            streak = 0
            queue.put((pulls_done, jackpots.copy()))
            jackpots.clear()
        else:
            streak += batch_size
            pulls_done += batch_size

    queue.put((pulls_done, jackpots))
    queue.put(None)  # sinyal selesai


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            GACHA SIMULATOR CLASS                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class GachaSimulator:
    """
    Encapsulasi semua state simulasi gacha.
    Menggantikan semua global variable dari versi original.
    """

    def __init__(self, config: dict):
        # ── Konfigurasi (read-only setelah init) ──
        self.game_name        = config["game_name"]
        self.prob             = config["probability"]
        self.target           = config["min_percobaan"]       # nilai_N
        self.batch_fast       = config["batch_size_fast"]     # batch_size
        self.batch_slow       = config["batch_size_slow"]     # a_little_batch_size
        self.batch_single     = config["batch_size_single"]   # little_batch_size (1)
        self.pull_method      = config["pull_method"]
        self.confidence       = config["confidence_target"]   # pp100
        self.log_interval     = config["log_interval"]
        self.enable_auto_menu = config["enable_auto_pull_menu"]

        # ── State yang berubah selama simulasi ──
        self._reset_state()

    # ──────────────────────────────────────────────────────────────────────────
    #  State Management
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_state(self):
        """Reset semua variabel simulasi ke kondisi awal."""
        self.total_pulls           = 0      # keseluruhan pull
        self.total_jackpot         = 0      # banyak jackpot yang didapatkan
        self.jarak_jackpot         = 0      # jarak percobaan menuju jackpot saat ini
        self.total_jackpot_terakhir = 0     # jarak jackpot sebelumnya
        self.jackpot_list          = [0]    # semua jarak jackpot yang tercatat
        self.new_pull              = 0      # counter pull baru (di-reset tiap jackpot)
        self.on_pull               = 0      # on-going pull counter
        self.loop_terakhir         = True   # flag: lanjutkan fase lambat
        self.loop_bagian_dua       = True   # flag: ulangi keseluruhan simulasi
        self.ii_terakhir           = 0      # iterasi di fase lambat
        self.bukti                 = 0      # counter bukti loop tambahan

        # Hasil prediksi MLE (diisi saat predict dipanggil)
        self.p100_pred = 0

    def _reset_for_retry(self):
        """
        Reset state untuk siklus baru (ketika jackpot terjadi di fase lambat).
        Menyimpan jackpot_list lama untuk prediksi sebelum di-clear.
        """
        old_jackpot_list = self.jackpot_list.copy()

        self.total_pulls            = 0
        self.total_jackpot          = 0
        self.jarak_jackpot          = 0
        self.total_jackpot_terakhir = 0
        self.jackpot_list           = [0]
        self.new_pull               = 0
        self.ii_terakhir            = 0
        self.bukti                  = 0
        self.loop_terakhir          = True

        clear_screen()
        print("Semua variabel telah direset.")
        predict_next_jackpot_mle(old_jackpot_list, self.confidence)
        if old_jackpot_list:
            print(f" Jakpot tertinggi adalah : {max(old_jackpot_list)}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Core Pull — Satu fungsi menggantikan satu_pull, satu_pull_lima, a_satu_pull
    # ──────────────────────────────────────────────────────────────────────────

    def _do_pull(self, batch_size: int, target_info: int | None = None) -> bool:
        """
        Lakukan satu batch pull.

        Args:
            batch_size:   Berapa banyak random number di-generate per batch.
            target_info:  Jika diberikan, tampilkan progress menuju target ini.
                          (digunakan oleh a_satu_pull di versi original)

        Returns:
            True  = tidak ada jackpot (lanjut)
            False = jackpot ditemukan (misi gagal / loop berhenti)
        """
        pulls = np.random.random(size=batch_size)
        hits = np.where(pulls < self.prob)[0]

        if len(hits) > 0:
            # ── JACKPOT ditemukan ──
            first_hit = hits[0] + 1
            self.jarak_jackpot += first_hit
            self.total_pulls += first_hit
            self.total_jackpot += 1
            self.total_jackpot_terakhir = self.jarak_jackpot
            self.jackpot_list.append(self.jarak_jackpot)
            self.jarak_jackpot = 0
            self.on_pull = 0

            # Tampilkan info
            print(f"\n ====>  jackpot jackpot didapatkan {self.total_pulls}  <====")
            print(f"====>  Informasi New Pull         {self.new_pull}  <====")
            if target_info is not None:
                remaining = target_info - self.new_pull
                print(f"====>  Informasi Total pull         {self.total_pulls:,}  <====")
                print(f"====> Menuju p99 {target_info} , seharusnya kurang {remaining} percobaan lagi!")
                print(f"\033[31m =================== Jackpot didapatkan. Loop Diulang! ===================== \033[0m")
            else:
                print(f"\033[31m =================== Jackpot didapatkan. Misi Gagal! ===================== \033[0m")

            # Catat kegagalan ke file
            log_to_file(
                f"\n ------> Kegagalan pada simulasi ke: {self.new_pull} , "
                f"pull baru: {self.new_pull} , Total pull: {self.total_jackpot_terakhir}"
            )

            # Reset pull counter
            self.new_pull = 0
            self.loop_terakhir = False

            return False  # jackpot = berhenti

        else:
            # ── Tidak ada jackpot ──
            self.jarak_jackpot += batch_size
            self.total_pulls += batch_size
            self.new_pull += batch_size
            self.on_pull += batch_size
            print(f"kamu tidak beruntung, total pull: {self.total_pulls:,}")

            return True  # lanjut

    # ──────────────────────────────────────────────────────────────────────────
    #  Menu Options
    # ──────────────────────────────────────────────────────────────────────────

    def pull_satu(self):
        """Opsi 1: Pull 1 kali (batch_size = 1)."""
        self._do_pull(self.batch_single)

    def pull_sepuluh(self):
        """Opsi 2: Pull 10 kali berturut-turut."""
        for i in range(10):
            print(f"Pull ke : {i + 1}: ", end="")
            self._do_pull(self.batch_single)

    def pull_manual(self, jumlah: int):
        """Opsi 4: Pull sebanyak input manual."""
        for i in range(jumlah):
            print(f"pull Manual ke : {i + 1}. Jarak Jackpot : {self.jarak_jackpot:,}", end="")
            self._do_pull(self.batch_single)

    def pull_kontinu(self):
        """Opsi 5: Pull terus sampai jackpot ditemukan."""
        i = 0
        while True:
            print(f"Pull ke: {i + 1}: ", end="")
            still_going = self._do_pull(self.batch_single)
            i += 1
            if not still_going:
                print("======== Loop berhenti. Silahkan gacha real time!! ==========")
                break

    # ──────────────────────────────────────────────────────────────────────────
    #  Fase Cepat — Automatic Pull (Normal)
    # ──────────────────────────────────────────────────────────────────────────

    def _automatic_pull_fast_phase(self):
        """
        FASE 1 (Cepat): Pull dalam batch besar sampai jarak terpanjang
        mendekati target (nilai_N - 10).
        """
        last_log = time.time()

        if NUMBA_AVAILABLE:
            # ── Numba JIT: satu panggilan cepat ──
            pulls, jackpots, final_streak = simulate_batches_numba(
                self.prob, self.batch_fast, self.target, self.jarak_jackpot
            )
            print("FFFFFFFFFFFFFFFFFFFFFFFFF ===================================================== FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
            print(f"\n Informasi sebelum berpindah ke loop lambat. nilai_N : {self.target} - 10  dan jarak_jackpot : {self.jarak_jackpot}")
            self.total_pulls += pulls
            if jackpots:
                self.jackpot_list.extend(jackpots)
                self.total_jackpot += len(jackpots)
                self.total_jackpot_terakhir = jackpots[-1]
            self.jarak_jackpot = final_streak
        else:
            # ── Fallback: loop Python biasa ──
            while True:
                if self.total_jackpot_terakhir >= (self.target - 10):
                    self.jarak_jackpot = self.total_jackpot_terakhir
                    log_to_file(
                        f"\n Informasi sebelum berpindah ke loop lambat. "
                        f"nilai_N : {self.target}  dan total_jackpot_terakhir : {self.total_jackpot_terakhir}"
                    )
                    print("FFFFFFFFFFFFFFFFFFFFFFFFF ===================================================== FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
                    print(f"\n Informasi sebelum berpindah ke loop lambat. nilai_N : {self.target}  dan total_jackpot_terakhir : {self.total_jackpot_terakhir}")
                    break

                pulls_arr = np.random.random(size=self.batch_fast)
                hits = np.where(pulls_arr < self.prob)[0]

                if len(hits) > 0:
                    first_hit = hits[0] + 1
                    self.jarak_jackpot += first_hit
                    self.total_pulls += first_hit
                    self.total_jackpot += 1
                    self.total_jackpot_terakhir = self.jarak_jackpot
                    print(f"jarak jackpot = {self.jarak_jackpot}  | Nilai N: {self.target}")
                    self.jackpot_list.append(self.jarak_jackpot)
                    self.jarak_jackpot = 0
                else:
                    self.jarak_jackpot += self.batch_fast
                    self.total_pulls += self.batch_fast

                # Progress log
                if time.time() - last_log >= self.log_interval:
                    clear_screen()
                    print("=====================>  Fast Pull System  <====================\n")
                    print(f"kamu tidak beruntung, Pull sebelum jackpot: {self.total_jackpot_terakhir:,}")
                    print(f"Target jarak adalah : {(self.target - 10)}")
                    print(f"Jackpot tertinggi : {max(self.jackpot_list):,}")
                    print(f"Total pull : {self.total_pulls:,}")
                    print(f"Total jackpot : {self.total_jackpot:,}")
                    print(f"Array List JackPot : {sorted(self.jackpot_list, reverse=True)[:5]}")
                    last_log = time.time()

    def _print_phase_stats(self):
        """Tampilkan statistik distribusi jackpot setelah fase cepat."""
        print(f"Total pull       : {self.total_pulls:,}")
        print(f"Total jackpot    : {self.total_jackpot:,}")
        print(f"Jackpot tertinggi: {max(self.jackpot_list)}")
        print(f"jarak jackpot terakhir: {self.total_jackpot_terakhir:,}")
        print(f"Informasi on-going pull: {self.jarak_jackpot}")

        df = pd.DataFrame(self.jackpot_list, columns=["Jarak Jackpot"])
        print("\n📊 Distribusi Jackpot:")
        print(df.describe())
        print("\n📊 Frekuensi Jarak Jackpot:")
        print(df["Jarak Jackpot"].value_counts().head(10))

        self.jackpot_list.sort(reverse=True)

        try:
            modus = df["Jarak Jackpot"].mode()[0]
            print(f"\nModus Jackpot: {modus}")
            print("=========> Catatan jackpot telah ditulis ")
        except Exception:
            print("\nModus tidak dapat dihitung (data terlalu unik).")

    # ──────────────────────────────────────────────────────────────────────────
    #  Fase Lambat — Pull kecil sampai confidence target tercapai
    # ──────────────────────────────────────────────────────────────────────────

    def _automatic_pull_slow_phase(self):
        """
        FASE 2 (Lambat): Pull dalam batch kecil.
        Berhenti jika:
          - Mencapai p100_pred tanpa jackpot  → SUKSES (real world pull!)
          - Jackpot terjadi                   → GAGAL  (ulangi dari awal)
        """
        self.ii_terakhir = 0
        self.bukti = 0

        while self.loop_terakhir:
            if self.ii_terakhir >= (self.p100_pred - 10):
                # ── SUKSES: streak cukup panjang, tidak ada jackpot ──
                self.enable_auto_menu = False
                print(f"\033[34m ===================== Belum Jackpot ======================= \033[0m")
                print("loop berakhir")
                print(f"\033[34m ===================== Semua loop selesai  ======================= \033[0m")
                print(f"\033[32m ====================> Real World Pull Now! <================= \033[0m")
                print(f"\n ==========================> Game : {self.game_name} <=========================")
                self.loop_terakhir = False
                self.loop_bagian_dua = False
                self.new_pull = 0
                break
            else:
                # ── Lakukan pull kecil (jika jackpot → _do_pull returns False → loop_terakhir=False) ──
                self._do_pull(self.batch_slow, target_info=self.p100_pred)
                self.bukti += 1
                self.ii_terakhir += self.batch_slow

    # ──────────────────────────────────────────────────────────────────────────
    #  Automatic Pull — Gabungan Fase Cepat + Lambat
    # ──────────────────────────────────────────────────────────────────────────

    def automatic_pull(self):
        """Opsi 3: Automatic pull = fase cepat → prediksi MLE → fase lambat."""
        start_time = time.time()

        # FASE 1: Pull cepat sampai mendekati target
        self._automatic_pull_fast_phase()

        # Tampilkan statistik fase cepat
        self._print_phase_stats()

        # Hitung prediksi MLE dari data yang terkumpul
        preds = predict_next_jackpot_mle(self.jackpot_list, self.confidence, self.jarak_jackpot)
        if preds:
            self.p100_pred = preds["p100_pred"]

        # FASE 2: Pull lambat sampai confidence target atau jackpot
        self._automatic_pull_slow_phase()

    # ──────────────────────────────────────────────────────────────────────────
    #  Automatic Pull — Multiprocessing
    # ──────────────────────────────────────────────────────────────────────────

    def automatic_pull_mp(self, workers: int | None = None):
        """Versi parallel dari automatic_pull() menggunakan multiprocessing."""
        start_time = time.time()
        cores = workers or max(1, mp.cpu_count() - 1)
        print(f"⚡ Starting multiprocessing with {cores} processes")

        manager = mp.Manager()
        stop_value = manager.Value("i", 0)
        queue = manager.Queue()

        seeds = [int(time.time()) + i for i in range(cores)]
        args = [(self.prob, self.batch_fast, s, stop_value, queue) for s in seeds]
        pool = mp.Pool(cores, initializer=np.random.seed)

        for a in args:
            pool.apply_async(_mp_worker, (a,))

        last_log = time.time()
        active_workers = cores

        while active_workers > 0:
            item = queue.get()
            if item is None:
                active_workers -= 1
                continue

            pulls_done, new_jacks = item
            self.total_pulls += pulls_done
            self.total_jackpot += len(new_jacks)
            if new_jacks:
                self.jackpot_list.extend(new_jacks)
                self.total_jackpot_terakhir = max(self.total_jackpot_terakhir, new_jacks[-1])
                self.jarak_jackpot = 0
            else:
                self.jarak_jackpot += pulls_done

            # Set stop target setelah data cukup
            if len(self.jackpot_list) >= 5 and stop_value.value == 0:
                preds = predict_next_jackpot_mle(self.jackpot_list, self.confidence)
                if preds:
                    stop_value.value = preds["p100_pred"] - 10
                    print(f"🛑 Target set to {stop_value.value} based on p100_pred")

            # Progress log
            if time.time() - last_log >= self.log_interval:
                clear_screen()
                print("=====================>  Fast Pull System (MP)  <====================\n")
                print(f"kamu tidak beruntung, Pull sebelum jackpot: {self.total_jackpot_terakhir:,}")
                if stop_value.value:
                    print(f"Target jarak adalah : {stop_value.value}")
                print(f"Jackpot tertinggi : {max(self.jackpot_list) if self.jackpot_list else 0:,}")
                print(f"Total pull : {self.total_pulls:,}")
                print(f"Total jackpot : {self.total_jackpot:,}")
                print(f"Array List JackPot : {sorted(self.jackpot_list, reverse=True)[:5]}")
                last_log = time.time()

            if stop_value.value and self.ii_terakhir >= stop_value.value:
                break

        pool.close()
        pool.join()

        elapsed = time.time() - start_time
        print(f"\n✅ Multiprocessing finished in {elapsed:.2f}s ({self.total_pulls / elapsed:,.0f} pulls/sec)")

        # Statistik akhir
        df = pd.DataFrame(self.jackpot_list, columns=["Jarak Jackpot"])
        print("\n📊 Distribusi Jackpot:")
        print(df.describe())
        print("\n📊 Frekuensi Jarak Jackpot:")
        print(df["Jarak Jackpot"].value_counts().head(10))

        self.jackpot_list.sort(reverse=True)
        try:
            modus = df["Jarak Jackpot"].mode()[0]
            print(f"\nModus Jackpot: {modus}")
        except Exception:
            print("\nModus tidak dapat dihitung (data terlalu unik).")

        self.loop_terakhir = False
        self.loop_bagian_dua = False

    # ──────────────────────────────────────────────────────────────────────────
    #  Run Simulation (Langsung)
    # ──────────────────────────────────────────────────────────────────────────

    def run_auto_simulation(self):
        """Jalankan simulasi otomatis, ulangi jika jackpot terjadi di fase lambat."""
        while self.loop_bagian_dua:
            self._reset_for_retry()
            if self.pull_method == "NO":
                self.automatic_pull()
            elif self.pull_method == "MP":
                self.automatic_pull_mp()

        print("\033[93m =============================== Semua loop selesai ===================================== \033[0m")
        print("\033[93m =============================== Realword Pull      ===================================== \033[0m")

    # ──────────────────────────────────────────────────────────────────────────
    #  Interactive Menu
    # ──────────────────────────────────────────────────────────────────────────

    def interactive_menu(self):
        """Menu interaktif setelah simulasi otomatis selesai."""
        while True:
            print(f"\n================= Simulasi Gacha V2 =====================")
            print(f"\n================= Game Name: {self.game_name} =====================")
            print(f"Nilai Pull baru                : {self.new_pull}")
            print(f"Banyak percobaan yang dilakukan sekarang untuk menuju jackpot : {self.jarak_jackpot}")
            print(f"Bentuk on going setelah mengikuti panduan prediksi : {self.on_pull}")
            print(f"banyak percobaan untukk jackpot sebelumnya  : {self.total_jackpot_terakhir}")
            print("1. Pull 1 kali")
            print("2. Pull 10 kali")
            print("3. Automatic pull fast" if self.enable_auto_menu else ".")
            print("4. Manual pull input")
            print("5. pull continu()")

            choice = input("Pilih opsi: ")

            if choice == "1":
                self.pull_satu()
            elif choice == "2":
                self.pull_sepuluh()
            elif choice == "3":
                if self.enable_auto_menu:
                    self.run_auto_simulation()
                else:
                    print("03 empty")
            elif choice == "4":
                jumlah = input("Pilih berapa banyak pull: ")
                log_to_file(
                    f"\n Ini nilai percobaan : {jumlah} , "
                    f"Nilai pull baru: {self.new_pull} , Total pull: {self.jarak_jackpot}"
                )
                self.pull_manual(int(jumlah))
            elif choice == "5":
                self.pull_kontinu()
            else:
                print("Opsi tidak valid, coba lagi.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                   MAIN                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    sim = GachaSimulator(CONFIG)

    # Tulis timestamp ke log file
    formatted_time = datetime.now().strftime("%H:%M:%S")
    log_to_file(f"\n ============  Game Name : {sim.game_name} Time: {formatted_time} ============= \n")

    # Jalankan simulasi otomatis
    sim.run_auto_simulation()

    # Masuk ke menu interaktif
    sim.interactive_menu()


if __name__ == "__main__":
    main()
