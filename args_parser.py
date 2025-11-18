import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-arch', type=str, default='AWDNet',
                        choices=[  # these four models are used for ablation experiments
                            'SpatCNN', 'SpecCNN',
                            'SpatRNET', 'SpecRNET',
                            # the proposed method
                            'SSRNET','MCT',
                            # these five models are used for comparison experiments
                            'SSFCNN', 'ConSSFCNN',
                            'TFNet', 'ResTFNet',
                            'MSDCNN',
                            'AWDNet',
                            # 只有小波变换，和小波变换残差lr/hsi
                            #  'GMCT2','GMCT2_lr',
                            # 单独只有分组多尺度卷积的跨注意力机制
                            #  'CYH_GCSE_lr','CYH_GCSE_msi',
                            # 冻结小波变换的参数，然后训练分组卷积跨尺度注意力
                            #  'GMCTUnFrozen',
                            # 小波变换不动参数，分别输入的是hsi/msi
                            #  'GMCT_hsi', 'GMCT_msi',
                        ])

    parser.add_argument('-root', type=str, default='./data')
    parser.add_argument('-dataset', type=str, default='Urban',
                        choices=['PaviaU', 'Botswana', 'KSC', 'Urban', 'Pavia', 'IndianP', 'Washington','MUUFL_HSI','salinas_corrected','Houston_HSI'])
    parser.add_argument('--scale_ratio', type=float, default=4)
    parser.add_argument('--n_bands', type=int, default=0)
    parser.add_argument('--n_select_bands', type=int, default=5)

    parser.add_argument('--model_path', type=str,
                        default='./checkpoints/dataset_arch.pkl',
                        help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='./data/dataset/train',
                        help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='./data/dataset/val',
                        help='directory for resized images')

    # learning settingl
    parser.add_argument('--n_epochs', type=int, default=10000,
                        help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=128)

    # 🔽 新增：中间融合权重路径 + 冻结开关
    # parser.add_argument('--mid_ckpt', type=str, default='./checkpoints/PaviaU_GMCT2.pkl',
    #                     help='path to middle-fusion checkpoint to load (e.g., PaviaU_GMCT2.pkl)')
    # parser.add_argument('--freeze_mid', action='store_true',
    #                     help='freeze middle-fusion (up to final_out) and finetune only top-right attention')

    # 迭代相关\
    # parser.add_argument('--T', type=int, default=1, help='GMCT2 迭代次数；=1 时等价于原模型')
    # parser.add_argument('--no_share', action='store_true', help='关闭权重共享（默认共享）')
    # parser.add_argument('--sensor_kernel', type=int, default=5, help='回投影高斯核大小（奇数）')
    # parser.add_argument('--sensor_sigma', type=float, default=1.0, help='回投影高斯核 sigma')

    # python main.py -arch GMCT2 -dataset PaviaU --T 1 不启用迭代 /python main.py -arch GMCT2_lr -dataset PaviaU --T 1 不启用迭代
    # python main.py -arch GMCT2 -dataset PaviaU --T 3 启用 3 轮迭代 + 共享权重： /python main.py -arch GMCT2_lr -dataset PaviaU --T 3
    # python main.py -arch GMCT2 -dataset PaviaU --T 3 --no_share 启用 3 轮迭代 + 每轮独立权重 / python main.py -arch GMCT2_lr -dataset PaviaU --T 3 --no_share

    # GCSE_lr/msi,输入的注意力机制是 Y=lr
    # parser.add_argument('--attn_guidance', type=str, default='lr', choices=['lr', 'msi'],
    #                     help='Attention guidance source: lr for LR-HSI, msi for HR-MSI guidance')
    # python main.py --attn_guidance lr
    # python main.py --attn_guidance msi

    args = parser.parse_args()
    return args
