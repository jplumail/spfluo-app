### Install

```bash
export AUTH_TOKEN=$(cat git_token.txt)
```
##### GPU
```bash
pip install -r requirements-gpu.txt
```
On windows : install pytorch !! see pytorch docs
##### CPU only
```bash
pip install -r requirements-cpu.txt
```
- On linux: install pytorch !! see pytorch docs

### Update
```bash
pip install --upgrade -r requirements-cpu.txt
```