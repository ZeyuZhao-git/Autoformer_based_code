import tarfile
import torch
import io

tar_file_path = "/home/zzy/AutoFormer/out_put/checkpoint-1.pth.tar"
def get_tarfile_checkpoint_keys_content(tar_file_path):
    with tarfile.open(tar_file_path, 'r') as tar:
        pth_files = [file for file in tar.getmembers() if file.name.endswith('.pth')]
        if len(pth_files) == 0:
            print("No .pth files found in the tar file.")
            return
        pth_file = pth_files[0]  # 假设只存在一个.pth文件，如果有多个.pth文件，可以根据需求进行处理
        
        # 解压缩.pth文件到内存中
        pth_file_content = tar.extractfile(pth_file).read()
        
        # 加载.pth文件
        checkpoint = torch.load(io.BytesIO(pth_file_content))
        
        # 输出文件中的键和内容
        print(f"Keys in .pth file '{pth_file.name}':")
        keys = checkpoint.keys()
        for key in keys:
            print(key)
        
        print(f"\nContent of .pth file '{pth_file.name}':")
        print(pth_file_content.decode())

get_tarfile_checkpoint_keys_content(tar_file_path)