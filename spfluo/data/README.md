Use like this:
```python
from spfluo import data
iso_data = data.generated_isotropic()
```

To generate the data yourself, call the `__main__`:
```bash
python -m spfluo.data --get_path
```

To generate the registry file, call `pooch`:
```bash
python -c "import pooch; pooch.make_registry('spfluo/data/', 'spfluo/data/registry.txt')"
```