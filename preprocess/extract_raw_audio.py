import os
from moviepy.editor import VideoFileClip
import argparse

'''
Function: Extract audio files(.wav) from videos.
'''

def get_audio_wav(video_path, output_path, audio_name):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(os.path.join(output_path, audio_name), fps=16000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='msrvtt', type=str, required=True,
      help='extract raw audio.')
    
    args = parser.parse_args() 

    dataset_path = os.path.join(f'../datasets/{args.dataset}/MSRVTT/videos/all')
    output_path =  os.path.join(f'./data/{args.dataset}/audio')

    video_list = os.listdir(dataset_path)
    # count =0
    for video in video_list:
        if not video.startswith('video'):
            continue
        # count+=1
        video_path = os.path.join(dataset_path, video)
        audio_name = str("{:06d}".format(int(video[5:-4])))+ '.wav'
        # print(audio_name)
    # print(count)
        try:
            get_audio_wav(video_path, output_path, audio_name)
            # print("finish video id: " + audio_name)
        except:
            print(f'cannot extract{audio_name} from {video_path}')

        # # audio_list = os.listdir(os.path.join(video_pth,dir))
        # for audio_id in audio_list:
        #     name = os.path.join(video_pth, dir,audio_id)
        #     audio_name = audio_id[:-4] + '.wav'
        #     try:
        #         get_audio_wav(name, save_pth, audio_name)
        #         print("finish video id: " + audio_name)
        #     except:
        #         print("cannot load ", name)

    print("\n------------------------------ end -------------------------------\n")