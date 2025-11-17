1. Download model manually
````bash
mkdir -p /workspace/models
mkdir /workspace/output
````

2. Secure copy manually
```bash
scp -P 47022 -r ./app root@89.37.121.214:/workspace/
scp -P 47022 ./requirements.txt root@89.37.121.214:/workspace/
ssh -p 47022 root@89.37.121.214 -L 8080:localhost:8080
```

3. Run script manually
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
git clone https://huggingface.co/Qwen/Qwen-Image-Edit-2509 /workspace/models/Qwen/Qwen-Image-Edit-2509
python3.11 -m app.main
```
