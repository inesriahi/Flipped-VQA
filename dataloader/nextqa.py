from typing import Any, Dict, List, Tuple
import torch
from .base_dataset import BaseDataset
import pandas as pd
import os

class NextQA(BaseDataset):
    def __init__(self, args: Any = None, tokenizer: Any = None, split: str = 'train') -> None:
        super().__init__(args, tokenizer, split)
        self.data = pd.read_csv(f'./data/nextqa/{split}.csv')#[:200] # TODO: Remove later
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 5
        self.qtype_mapping = {'CH': 1, 'CW': 2, 'TN': 3, 'TC': 4, 'TP': 5, 'DL': 6, 'DC': 7, 'DO': 8}
        folder_path_audio = '/scratch/project_462000189/ines/Flipped-VQA/data/nextqa/audio_features_imagebind_10_frames' # loaded shape later will be (10, 1024)
        if args.audio_merge == "attention": # need only one dim feature for each audio
            folder_path_audio = '/scratch/project_462000189/ines/Flipped-VQA/data/nextqa/audio_features_imagebind' # loaded shape later will be (1, 1024)
        
        self.video_features = torch.load(f'./data/{args.dataset}/video_features/clipvitl14.pth')
        self.audio_features = torch.load(os.path.join(folder_path_audio, "features", "imagebind.pth"))
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx: int) -> Dict[str, str]:
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = [self.data[f'a{i}'].values[idx] for i in range(self.num_options)]

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text

    def _get_video(self, video_id: str) -> Tuple[torch.Tensor, int]:
        """
        Retrieves and processes a video tensor based on the given video_id.

        Input:
        - video_id (str): A string identifier for the video. The function looks up 
        this ID in the `self.features` dictionary to retrieve the corresponding 
        video tensor.

        Output:
        - Tuple containing:
            - video (torch.Tensor): A tensor representing the video. Its shape depends 
            on the condition:
                - If `video_id` is not in `self.features`, the shape is [1, self.features_dim].
                - If `video_id` is found, but the number of features is more than `self.max_feats`, 
                it's downsampled to [self.max_feats, self.features_dim].
                - If the number of features is less than `self.max_feats`, it's padded with zeros 
                to [self.max_feats, self.features_dim].
                - If the number of features is exactly `self.max_feats`, the shape is 
                [self.max_feats, self.features_dim].
            - video_len (int): The length of the video in terms of the number of features, 
            which will be either the original length of the video (if less than or equal to 
            `self.max_feats`) or `self.max_feats` if the original length exceeds `self.max_feats`.

        The function assumes `self.features_dim` and `self.max_feats` are predefined attributes 
        of the class instance.
        """
        if video_id not in self.video_features:
            print(video_id, "video not found!")
            video = torch.zeros(1, self.features_dim) # (1, features_dim)
        else:
            video = self.video_features[video_id].float() # (n_frames, features_dim)
        
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled) # (max_feats, features_dim)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], dim=0)
        else:
            video_len = self.max_feats
        return video, video_len # shape of video [self.max_feats, self.features_dim].
    
    def _get_audio(self, audio_id: str) -> Tuple[torch.Tensor, int]:
        """
        Retrieves and processes a audio tensor based on the given audio_id.

        Input:
        - audio_id (str): A string identifier for the audio. The function looks up 
        this ID in the `self.audio_features` dictionary to retrieve the corresponding 
        audio tensor.

        Output:
        - Tuple containing:
            - audio (torch.Tensor): A tensor representing the audio. Its shape depends 
            on the condition:
                - If `audio_id` is not in `self.audio_features`, the shape is [1, self.features_dim].
                - If `audio_id` is found, but the number of features is more than `self.max_feats`, 
                it's downsampled to [self.max_feats, self.features_dim].
                - If the number of features is less than `self.max_feats`, it's padded with zeros 
                to [self.max_feats, self.features_dim].
                - If the number of features is exactly `self.max_feats`, the shape is 
                [self.max_feats, self.features_dim].
            - audio_len (int): The length of the audio in terms of the number of features, 
            which will be either the original length of the audio (if less than or equal to 
            `self.max_feats`) or `self.max_feats` if the original length exceeds `self.max_feats`.

        The function assumes `self.features_dim` and `self.max_feats` are predefined attributes 
        of the class instance.
        """
        if audio_id not in self.audio_features:
            print(audio_id, "audio not found!")
            audio = torch.zeros(1, self.audio_features_dim) # (1, audio_features_dim)
        else:
            audio = self.audio_features[audio_id].float() # (n_frames, audio_features_dim)
        if len(audio) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(audio[(j * len(audio)) // self.max_feats])
            audio = torch.stack(sampled) # (max_feats, audio_features_dim)
            audio_len = self.max_feats
        elif len(audio) < self.max_feats and self.args.audio_merge != "attention": # because in attention case, the shape is already (1, 1024)
            audio_len = len(audio)
            audio = torch.cat([audio, torch.zeros(self.max_feats - audio_len, self.audio_features_dim)], dim=0)
        else:
            audio_len = self.max_feats
        return audio, audio_len # shape of audio [self.max_feats, self.audio_features_dim].

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vid = self.data['video'].values[idx]
        qtype = self.qtype_mapping[self.data['type'].values[idx]]
        answer:int = self.data['answer'].values[idx]
        text = self._get_text(idx) # The answer itself is not included yet
        text_id, label, video_start, video_index, label_mask, prefix_index = self._get_text_token(text, answer, options=text["options"])
        if self.args.audio_only:
            audio, audio_len = self._get_audio(f'{vid}')         
        elif self.args.audio:
            video, video_len = self._get_video(f'{vid}') # shape of video [self.max_feats, self.features_dim].
            audio, audio_len = self._get_audio(f'{vid}')
        else:
            video, video_len = self._get_video(f'{vid}') # shape of video [self.max_feats, self.features_dim].

        # shuffle video frames order for qav task
        # shuffeld_frame_indecies = torch.randperm(self.max_feats) # permutes from 0 to self.max_feats
        # shuffled_video_frames = video.clone()
        # shuffled_video_frames = shuffled_video_frames[shuffeld_frame_indecies]
        # start_frame_index_label = (label['qav'][0] == 0).nonzero(as_tuple=True)[0].item() # label['qav'] is of shape [1,128]
        # label['qav'][:,start_frame_index_label:start_frame_index_label+self.max_feats] = shuffeld_frame_indecies
        # if self.args.debug:
        #     print("Label QAV after shuffling video frames:", label['qav'])
        if self.args.audio and self.args.audio_only:
            if self.args.debug:
                print("Shape of audio:", audio.shape)
            return {"vid": vid, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                    "video_index": video_index, "audio":audio, "audio_len": audio_len, "label_mask": label_mask, 
                    "qid": idx, "answer": answer, "qtype": qtype, "prefix_index": prefix_index}
        elif self.args.audio and not self.args.audio_only:
            if self.args.debug:
                print("Shape of audio:", audio.shape, "Shape of video:", video.shape)
            return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, 
                    "label": label, "video_start": video_start, "video_index": video_index, "audio":audio,
                    "audio_len": audio_len, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype, 
                    "prefix_index": prefix_index}
        else:
            if self.args.debug:
                print("Shape of video:", video.shape)
            return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                    "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype, "prefix_index": prefix_index}

    def __len__(self):
        return len(self.data)