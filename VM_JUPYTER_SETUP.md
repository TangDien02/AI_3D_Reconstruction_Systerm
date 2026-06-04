# Google Cloud VM Jupyter Setup

Huong dan nay dung cho Google Cloud Compute Engine VM co GPU Tesla T4,
chay JupyterLab qua ngrok de thay the Colab.

## 1. Mo SSH vao VM

Vao Google Cloud Console:

```text
Compute Engine -> VM instances -> dong VM cua ban -> SSH
```

Neu da tat het tab SSH thi chi can bam lai nut `SSH`.

Sau khi vao thanh cong, prompt se co dang:

```bash
pminhchien2006@instance-20260603-015731:~$
```

## 2. Kiem tra GPU

Chay:

```bash
nvidia-smi
```

Neu thay `Tesla T4` la GPU da san sang.

Neu `nvidia-smi: command not found`, cai driver Debian 12:

```bash
sudo sed -i 's/^Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources
sudo apt update
sudo apt install -y linux-headers-$(uname -r) dkms build-essential
sudo apt install -y nvidia-driver firmware-misc-nonfree
sudo reboot
```

Sau reboot, SSH lai va chay:

```bash
nvidia-smi
```

## 3. Vao workspace va virtualenv

```bash
cd ~/work
source venv/bin/activate
```

Neu `~/work/venv` chua ton tai:

```bash
sudo apt update
sudo apt install -y git wget curl tmux unzip python3-pip python3-venv build-essential

mkdir -p ~/work
cd ~/work
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

## 4. Cai PyTorch CUDA neu chua co

```bash
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Test:

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")
PY
```

Ket qua dung:

```text
CUDA: True
GPU: Tesla T4
```

## 5. Cai JupyterLab neu chua co

```bash
pip install --no-cache-dir jupyterlab
```

## 6. Chay JupyterLab trong tmux

Nen chay Jupyter trong `tmux` de tat tab SSH khong lam mat server.

Tao session moi:

```bash
tmux new -s jupyter
```

Neu bao `duplicate session: jupyter`, vao lai session cu:

```bash
tmux attach -t jupyter
```

Trong tmux, chay:

```bash
cd ~/work
source venv/bin/activate
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Jupyter se in ra URL co token, dang:

```text
http://127.0.0.1:8888/lab?token=...
```

Giu lai token nay.

De thoat khoi tmux ma Jupyter van chay:

```text
Ctrl+B
D
```

## 7. Chay ngrok cho Jupyter port 8888

Mo mot tab SSH khac vao VM.

Neu da co ngrok:

```bash
ngrok http 8888
```

Neu chua co ngrok:

```bash
cd ~/work
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/
```

Neu can add token ngrok:

```bash
ngrok config add-authtoken YOUR_NGROK_TOKEN
```

Sau do:

```bash
ngrok http 8888
```

Ngrok se in ra:

```text
Forwarding https://xxxx.ngrok-free.dev -> http://localhost:8888
```

Mo tren Windows:

```text
https://xxxx.ngrok-free.dev/lab?token=TOKEN_CUA_JUPYTER
```

Vi du neu domain cua ban la:

```text
https://knoll-richness-shield.ngrok-free.dev
```

thi URL se la:

```text
https://knoll-richness-shield.ngrok-free.dev/lab?token=TOKEN_CUA_JUPYTER
```

## 8. Neu da tat het SSH tab

Lam lai:

1. Bam `SSH` trong Google Cloud Console.
2. Kiem tra session:

```bash
tmux ls
```

3. Neu thay `jupyter`, vao lai:

```bash
tmux attach -t jupyter
```

4. Neu khong thay `jupyter`, tao lai:

```bash
tmux new -s jupyter
cd ~/work
source venv/bin/activate
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

5. Mo tab SSH thu hai va chay lai:

```bash
ngrok http 8888
```

6. Mo link ngrok tren Windows kem token Jupyter.

## 9. Upload notebook len VM

Trong JupyterLab:

```text
Sidebar trai -> nut Upload -> chon file .ipynb
```

Thu muc hien tai nen la:

```text
/home/pminhchien2006/work
```

Notebook VM da tao trong project local:

```text
notebook/Hunyuan_workflow_vm_jupyter_full_ascii.ipynb
```

Upload file do len JupyterLab roi chay tu tren xuong.

## 10. Chay Hunyuan worker port 8010

Jupyter port 8888 chi de mo notebook.

Neu notebook can expose Hunyuan VM worker cho Windows backend, mo them mot tab
SSH va chay:

```bash
ngrok http 8010
```

Sau do paste URL port 8010 vao cell notebook khi duoc hoi.

Luu y:

- `ngrok http 8888` la cho JupyterLab.
- `ngrok http 8010` la cho Hunyuan worker.
- Hai port nay khac nhau.

## 11. Token HF va ngrok

Neu da luu token trong `~/.bashrc`:

```bash
nano ~/.bashrc
```

Them:

```bash
export HF_TOKEN="hf_xxx"
export NGROK_TOKEN="xxx"
```

Sau khi sua file, can load lai:

```bash
source ~/.bashrc
```

Neu Jupyter da chay truoc khi sua `~/.bashrc`, restart Jupyter de notebook doc duoc token.

Trong notebook, kiem tra:

```python
import os
print("HF_TOKEN:", bool(os.environ.get("HF_TOKEN")))
print("NGROK_TOKEN:", bool(os.environ.get("NGROK_TOKEN")))
```

## 12. Luu y Spot VM

Spot VM co the bi Google tat bat cu luc nao.

Nen:

- Chay viec dai trong `tmux`.
- Luu checkpoint thuong xuyen.
- Khong de file quan trong chi nam tren VM neu VM/disk co the bi xoa.
- Nen backup code len GitHub, output/checkpoint len Cloud Storage hoac noi luu ben vung.

## 13. Lenh nhanh moi lan bat lai VM

Tab SSH 1:

```bash
tmux new -s jupyter
cd ~/work
source venv/bin/activate
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Detach:

```text
Ctrl+B, roi D
```

Tab SSH 2:

```bash
ngrok http 8888
```

Mo tren Windows:

```text
https://YOUR_NGROK_DOMAIN/lab?token=YOUR_JUPYTER_TOKEN
```

