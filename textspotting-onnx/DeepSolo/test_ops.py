import torch
torch.ops.load_library('/usr/local/lib/python3.9/dist-packages/DeepSolo/ext/_C.cpython-39-x86_64-linux-gnu.so')
tensor1 = torch.randn(1, 76500, 8, 32).cuda()  # Float tensor
tensor2 = torch.randint(0, 100, (4, 2), dtype=torch.long).cuda()  # Long tensor
tensor3 = torch.arange(4, dtype=torch.long).cuda()  # Long tensor
tensor4 = torch.randn(1, 2500, 8, 4, 4, 2).cuda()  # Float tensor
tensor5 = torch.randn(1, 2500, 8, 4, 4).cuda()  # Float tensor
output = torch.ops.my_ops.ms_deform_attn_forward_onnx(tensor1,tensor2,tensor3,tensor4,tensor5,64)

print(output)
