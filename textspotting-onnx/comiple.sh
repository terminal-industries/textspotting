 rm -rf build/
 pip uninstall DeepSolo
 rm -rf ./dist/*.*
 rm -rf ./DeepSolo.egg-info/
 python3 setup.py sdist bdist_wheel
 pip install dist/*.whl