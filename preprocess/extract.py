import torch as th
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from audio_loader import AudioLoader
from torch.utils.data import DataLoader
# from extract.preprocessing import Preprocessing
# from extract.random_sequence_shuffler import RandomSequenceSampler
# from args import MODEL_DIR
import argparse
import clip
from einops import rearrange

cuda_device_number = '0'
has_cuda = th.cuda.is_available()
device = th.device('cuda:'+cuda_device_number if has_cuda else 'cpu')
# print('has_cuda', has_cuda)
print('GPU', cuda_device_number)


parser = argparse.ArgumentParser(description="Easy audio feature extractor")

parser.add_argument(
    "--path",
    type=str,
    help="the path of audio files",
)
parser.add_argument(
    "--output",
    type=str,
    help="the output path",
)
parser.add_argument(
    "--sample_rate",
    type=int,
    default=16000,
    help="the sampling frequency of the input audio data",
)
parser.add_argument(
    "--num_mel_bins",
    type=int,
    default=128,
    help="the number of mel bins to use in the filterbank",
)
parser.add_argument(
    "--targetlength",
    type=int,
    default=2240,
    help="the target length (10 seconds)",
)
parser.add_argument(
    "--frame_shift",
    type=float,
    default=10,
    help="the time shift between consecutive frames in milliseconds",
)
parser.add_argument(
    "--audio_mean",
    type=float,
    help="the audio mean",
)
parser.add_argument(
    "--audio_std",
    type=float,
    help="the audio std",
)
parser.add_argument(
    "--num_decoding_thread",
    type=int,
    default=3,
    help="number of parallel threads for video decoding",
)
parser.add_argument(
    "--model_dir",
    type=str,
    default="./pretrained",
    help="the path of the clip model",
)
parser.add_argument(
    "--feature_dim", type=int, default=768, help="output audio feature dimension"
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="batch size for extraction"
)
parser.add_argument(
    "--half_precision",
    type=int,
    default=1,
    help="whether to output half precision float or not",
)
parser.add_argument(
    "--l2_normalize",
    type=int,
    default=0,
    help="whether to l2 normalize the output feature",
)
args = parser.parse_args()

dataset = AudioLoader(
    args.path,
    args.output,
    args.sample_rate,
    args.num_mel_bins,
    args.frame_shift,
    args.targetlength,
    args.audio_mean,
    args.audio_std,
)
n_dataset = len(dataset)
# print(dataset[5]["audio"].size(),dataset[5]["input"])
# sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread
)
# # calculate mean and std
# mean = []
# std = []
# for idx, data in enumerate(loader):
#     audio_spec = data['audio'].to(device)       
#     b,t,c,w,h = audio_spec.shape
#     audio_spec =  rearrange(audio_spec, 'b t c h w -> (b t c) (h w)')
#     local_mean = th.mean(audio_spec, dim=-1)
#     local_std = th.std(audio_spec, dim=-1)
#     mean.append(local_mean)
#     std.append(local_std)
# print('mean: ',th.hstack(mean).mean().item(),'std: ',th.hstack(std).mean().item())

# for k, data in enumerate(loader):
#     input_file = data["input"][0]
#     output_file = data["output"][0]
#     video_file = data["audio"]
#     print(video_file.shape)
#     if len(data["audio"].shape) > 3:
#         print(
#             "Computing features of audio {}/{}: {}".format(
#                 k + 1, n_dataset, input_file
#             )
#         )
#         video = data["audio"].squeeze()
#         if len(video.shape) == 4:
#             n_chunk = len(video)
#     print(video_file.shape, input_file, output_file, n_chunk)


# preprocess = Preprocessing()
model, _ = clip.load("ViT-L/14", download_root=args.model_dir)

# print(model)
model.eval()
model = model.cuda()

with th.no_grad():
    for k, data in enumerate(loader):
        input_file = data["input"][0]
        output_file = data["output"][0]
        print(input_file, output_file)
        if len(data["audio"].shape) > 3:
            print(
                "Computing features of audio {}/{}: {}".format(
                    k + 1, n_dataset, input_file
                )
            )
            audio = data["audio"].squeeze()
            if len(audio.shape) == 4:
                n_chunk = len(audio)
                features = th.cuda.FloatTensor(n_chunk, args.feature_dim).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in tqdm(range(n_iter)):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    audio_batch = audio[min_ind:max_ind].cuda()
                    batch_features = model.encode_image(audio_batch)
                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)
                    features[min_ind:max_ind] = batch_features
                features = features.cpu().numpy()
                if args.half_precision:
                    features = features.astype("float16")
                np.save(output_file, features)
        else:
            print("Audio {} already processed.".format(input_file))