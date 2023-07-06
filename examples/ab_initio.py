import os


particles_dir = "examples/data/centriole_thibaut/c1"
output_dir = "examples/output_ab_initio_highLR"
psf_path = "examples/data/centriole_thibaut/PSF_6_c1_resized_ratio_2.tif"

os.system(f"rm -r {output_dir}")
os.system(
    f"python -m spfluo.ab_initio_reconstruction --particles_dir {particles_dir} --output_dir {output_dir} --psf_path {psf_path} --gpu cucim --interp_order 3 --N_iter_max 100 --eps -1 --lr 1"
)
