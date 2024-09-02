Use like this:
```python
from spfluo import data
iso_data = data.generated_isotropic()
```

To generate the data yourself, call the `__main__`:
```bash
python spfluo/data/generate_data.py
```

If you want to add some data, add it. Then generate the registry file and upload the archive.

To generate the registry file, call `make_registry.py`:
```bash
python spfluo/data/make_registry.py
```

Set unistra username/password and upload the files to seafile:
```bash
UNISTRA_USERNAME= UNISTRA_PASSWORD= python spfluo/data/upload.py
```