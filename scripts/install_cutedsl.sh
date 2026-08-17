python -m pip uninstall -y \
  cutlass \
  nvidia-cutlass \
  nvidia-cutlass-dsl \
  nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-cu13 \
  nvidia-cudnn-frontend

python -m pip install -U pip setuptools wheel

python -m pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.7.0"
python -m pip install --no-cache-dir "nvidia-cudnn-frontend[cutedsl]==1.27.0"

