## Web demo for tranX Semantic Parsing Toolkit

This folder contains a web demo for `tranX`, see [here](http://moto.clab.cs.cmu.edu:8081) for an online demo. 
 
```bash
source activate py3torch3cuda9
PYTHONPATH=../ python app.py --config_file data/release/config.json
```

Note: the Django semantic parser only works under Python 2. To host a demo for Django:
 
```bash
source activate py2torch3cuda9
PYTHONPATH=../ python app.py --config_file data/release/config_py2.json
```

## Thanks

    * Data pre-processing scripts located under `datasets.(geo|atis).data_process` is authored by [Li Dong](http://homepages.inf.ed.ac.uk/s1478528/)
    * The parse tree visualizer is from [tree-viewer](https://github.com/christos-c/tree-viewer) (by [christos](https://github.com/christos))
    
## Reference

```
Li Dong and Mirella Lapata
Language to Logical Form with Neural Attention, ACL 2016
```