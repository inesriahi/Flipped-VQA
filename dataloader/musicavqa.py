import os
from typing import Any, Dict, List, Tuple
import torch
from .base_dataset import BaseDataset
import json
import re

class MusicAVQA(BaseDataset):
    def __init__(self, args: Any = None, tokenizer: Any = None, split: str = 'train') -> None:
        super().__init__(args, tokenizer, split)
        file_path = f'./data/musicavqa/avqa-{split}.json'
        with open(file_path, 'r') as f:
            self.data = json.load(f)

        # Your specified folder path
        folder_path_video = '/scratch/project_462000189/ines/Flipped-VQA/data/musicavqa/video'

        folder_path_audio = '/scratch/project_462000189/ines/Flipped-VQA/data/musicavqa/audio_features_imagebind_10_frames' # loaded shape later will be (10, 1024)
        if args.audio_merge == "attention": # need only one dim feature for each audio
            folder_path_audio = '/scratch/project_462000189/ines/Flipped-VQA/data/musicavqa/audio_features_imagebind' # loaded shape later will be (1, 1024)
            
        print(f"Number of rows before removing nan rows in {file_path}: {len(self.data)}")

        self.video_features = torch.load(os.path.join(folder_path_video, "clipvitl14.pth"))
        self.audio_features = torch.load(os.path.join(folder_path_audio, "features", "imagebind.pth"))
        self.answer_mapping = {0: '(A)'}
        self.num_options = 1
        self.qtype_mapping = {
            'Audio_Temporal': 1,
            'Audio_Existential': 2,
            'Audio_Comparative': 3,
            'Audio_Location': 4,
            'Audio_Counting': 5,
            'Visual_Temporal': 6,
            'Visual_Existential': 7,
            'Visual_Comparative': 8,
            'Visual_Location': 9,
            'Visual_Counting': 10,
            'Audio-Visual_Temporal': 11,
            'Audio-Visual_Existential': 12,
            'Audio-Visual_Comparative': 13,
            'Audio-Visual_Location': 14,
            'Audio-Visual_Counting': 15
        }

        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx: int, question:str, options: list) -> Dict[str, str]:
        question = question.capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

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
        this ID in the `self.video_features` dictionary to retrieve the corresponding 
        video tensor.

        Output:
        - Tuple containing:
            - video (torch.Tensor): A tensor representing the video. Its shape depends 
            on the condition:
                - If `video_id` is not in `self.video_features`, the shape is [1, self.features_dim].
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
            audio = torch.zeros(1, self.features_dim) # (1, features_dim)
        else:
            audio = self.audio_features[audio_id].float() # (n_frames, features_dim)
        if len(audio) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(audio[(j * len(audio)) // self.max_feats])
            audio = torch.stack(sampled) # (max_feats, features_dim)
            audio_len = self.max_feats
        elif len(audio) < self.max_feats and self.args.audio_merge != "attention": # because in attention case, the shape is already (1, 1024)
            audio_len = len(audio)
            audio = torch.cat([audio, torch.zeros(self.max_feats - audio_len, self.features_dim)], dim=0)
        else:
            audio_len = self.max_feats
        return audio, audio_len # shape of audio [self.max_feats, self.features_dim].

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        vid = item["video_id"]
        question_content = item["question_content"]
        answer_content = item.get("anser", "")  # Using 'get' to avoid KeyError if 'anser' is misspelled or missing

        # Convert the string representation of a list into an actual list
        templ_values = eval(item["templ_values"])
        # Replace placeholders in the question content
        placeholders = re.findall(r"<(.*?)>", question_content)
        for i, placeholder in enumerate(placeholders):
            if i < len(templ_values):
                question_content = question_content.replace(f"<{placeholder}>", templ_values[i], 1)

        type_item = eval(item["type"])
        qtype = self.qtype_mapping[type_item[0] + "_" + type_item[1]]
        answer = 0
        text = self._get_text(idx, question = question_content,options=[answer_content]) # The answer itself is not included yet
        text_id, label, video_start, video_index, label_mask, prefix_index = self._get_text_token(text, answer, options=text['options'])
        if self.args.audio_only:
            audio, audio_len = self._get_audio(f'{vid}')         
        elif self.args.audio:
            video, video_len = self._get_video(f'{vid}') # shape of video [self.max_feats, self.features_dim].
            audio, audio_len = self._get_audio(f'{vid}')
        else:
            video, video_len = self._get_video(f'{vid}') # shape of video [self.max_feats, self.features_dim].

        if self.args.audio and self.args.audio_only:
            return {"vid": vid, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                    "video_index": video_index, "audio":audio, "audio_len": audio_len, "label_mask": label_mask, 
                    "qid": idx, "answer": answer, "qtype": qtype, "prefix_index": prefix_index}
        elif self.args.audio and not self.args.audio_only:
            return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, 
                    "video_start": video_start, "video_index": video_index, "audio":audio, "audio_len": audio_len,
                    "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype, "prefix_index": prefix_index}
        else:
            return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, 
                    "video_start": video_start, "video_index": video_index, "label_mask": label_mask, "qid": idx,
                    "answer": answer, "qtype": qtype, "prefix_index": prefix_index}

    def __len__(self):
        return len(self.data)