import os
import shutil

output_prefix = os.path.abspath(os.path.join(
    os.getcwd(), os.pardir, os.pardir, 'big_images', 'Figure_3_temp'))

src_dir = os.path.join(output_prefix, '2_mp4_figs')
dst_dir = os.path.abspath(os.path.join(output_prefix, os.pardir, 'figure_3'))
if os.path.isdir(dst_dir): shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)
