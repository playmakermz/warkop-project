## Catatan: Setup Python 3.10.6 di Linux

---

### 1. Pindah ke Bash Shell
```bash
bash
```

---

### 2. Install pyenv
```bash
curl https://pyenv.run | bash
```

Setelah install, tambahkan pyenv ke bash config:
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

Reload config:
```bash
source ~/.bashrc
```

---

### 3. Install Python 3.10.6
```bash
pyenv install 3.10.6
```

Set ke project folder kamu:
```bash
pyenv local 3.10.6
```

Verifikasi:
```bash
python --version  # harus tampil Python 3.10.6
```

---

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

Jika ada package PyTorch dengan CUDA, install terpisah:
```bash
# Cek GPU dulu
nvidia-smi

# Install torch sesuai CUDA version kamu
pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Jika tidak ada. hapus terlebih dahulu pytorch di txt

---

### 5. Jalankan Script
```bash
python your_script.py
```

---

> 💡 **Tips:** Setiap kali buka terminal baru, ketik `bash` dulu sebelum jalankan script, karena default shell kamu masih Fish.

Mau saya buatkan ini jadi file `.md` atau `.txt` untuk disimpan?
