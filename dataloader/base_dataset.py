import torch
from torch.utils.data import Dataset
import copy
import sys

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.max_feats = args.max_feats
        self.features_dim = 768
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.split = split
    
    def _get_padding_id(self, text_id):
        # This method takes a batch of tokenized text (text_id) and ensures that each sequence in the batch is padded to the maximum sequence length (self.max_seq_len). 
        # Sequences longer than the maximum are truncated, and sequences shorter than the maximum are padded with -1.
        padding_text_id = torch.zeros((len(text_id), self.max_seq_len), dtype=torch.int64) - 1
        for i, tid in enumerate(text_id):
            padding = self.max_seq_len - len(tid)
            if padding >= 0:
                padding_text_id[i, :len(tid)] = tid
            else:
                padding_text_id[i] = tid[:self.max_seq_len]
                print('max sequence length overflow')
        return padding_text_id
    
    def _get_text_token(self, text: str, answer: int):
        verbose_shapes = False
        # vqa_id, vqa_prefix_index, vqa_video_start: These are the token IDs, the index at which the answer starts, and the position in the token sequence where the video segment begins for the VQA task.
        vqa_id, vqa_prefix_index, vqa_video_start = self.tokenizer.encode_vqa(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        # vaq_id, vaq_prefix_index, vaq_video_start: Similar to VQA, but for the VAQ task.
        vaq_id, vaq_prefix_index, vaq_video_start = self.tokenizer.encode_vaq(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        # qav_id, qav_prefix_index: For the QAV task, which involves ordering video frames, the qav_id are the token IDs, and qav_prefix_index is the position where the video frames are expected to start in the token sequence.
        qav_id, qav_prefix_index = self.tokenizer.encode_qav(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        
        # The method then converts each list of IDs (vqa_id, vaq_id, qav_id) into PyTorch tensors. 
        vqa_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vqa_id]
        vaq_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vaq_id]
        qav_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in qav_id]

        # Print shapes of the converted tensors
        if verbose_shapes:
            print("Shapes of original tensors:")
            print("VQA ID shape:", [v.shape for v in vqa_id])
            print("VAQ ID shape:", [v.shape for v in vaq_id])
            print("QAV ID shape:", [v.shape for v in qav_id])
        
        # Each list of token IDs is then padded to the maximum sequence length (self.max_seq_len) using the _get_padding_id method. This ensures that all sequences are of uniform length, which is a requirement for batching in PyTorch. 
        # If the sequence is longer than self.max_seq_len, it gets truncated, otherwise, it gets padded with -1.
        vqa_padding_text_id = self._get_padding_id(vqa_id)
        vaq_padding_text_id = self._get_padding_id(vaq_id)
        qav_padding_text_id = self._get_padding_id(qav_id)

        if verbose_shapes:
            print("\nShapes after padding:")
            print("VQA Padding Text ID shape:", vqa_padding_text_id.shape)
            print("VAQ Padding Text ID shape:", vaq_padding_text_id.shape)
            print("QAV Padding Text ID shape:", qav_padding_text_id.shape)

        # label
        # Creating Labels and Masks: After padding, the method creates labels for each task by copying the padded text IDs. It then masks parts of the sequence that aren't relevant for training
        vqa_label = copy.deepcopy(vqa_padding_text_id)
        # For VQA, everything before vqa_prefix_index is masked (set to -1),
        vqa_label[:, :vqa_prefix_index] = -1 # Set the tokens before the prefix index to -1 (ignore them during loss calculation)
        vqa_label_mask = vqa_label.ge(0) # Create a mask where positive values are set to True (valid for loss calculation)
        vqa_label[~vqa_label_mask] = 0 # Set the masked out tokens to 0
        vqa_label_mask = vqa_label_mask.float() # Convert the mask to float type for later operations
        
        # Repeat the process for creating labels and masks for the VAQ task.
        vaq_label = copy.deepcopy(vaq_padding_text_id)
        vaq_label[:, :vaq_prefix_index] = -1
        vaq_label_mask = vaq_label.ge(0)
        vaq_label[~vaq_label_mask] = 0
        vaq_label_mask = vaq_label_mask.float()
        
        # For the QAV task, initialize all labels to -1 and set a range for video frame ordering.
        qav_label = torch.ones_like(qav_padding_text_id) * -1
        qav_label[:, qav_prefix_index:qav_prefix_index+self.max_feats] = torch.arange(self.max_feats)
        qav_label_mask = torch.zeros_like(qav_padding_text_id)
        qav_label_mask[:, qav_prefix_index] = 1 # Only the start index is valid for the QAV task
        qav_label_mask = qav_label_mask.float()
                
        # text mask
        # Create text masks for each task to identify valid tokens versus padding.
        vqa_text_mask = vqa_padding_text_id.ge(0)
        vqa_padding_text_id[~vqa_text_mask] = 0
        vaq_text_mask = vaq_padding_text_id.ge(0)
        vaq_padding_text_id[~vaq_text_mask] = 0
        qav_text_mask = qav_padding_text_id.ge(0)
        qav_padding_text_id[~qav_text_mask] = 0
        
        # video index
        # Create a range of indices for the video features in each task.
        vqa_video_index = torch.arange(vqa_prefix_index, vqa_prefix_index + self.max_feats) # Is there a problem here? I think vqa_prefix_index here for example holds the the start index of the answer not of the video
        vaq_video_index = torch.arange(vaq_prefix_index, vaq_prefix_index + self.max_feats) # and here as well, that vaq_prefix_index is the start index of the question
        qav_video_index = torch.arange(qav_prefix_index, qav_prefix_index + self.max_feats) # but here only this works, as qav_prefix_index holds the start index of the video

        if verbose_shapes:
            # Print shapes of labels and masks
            print("\nShapes of labels and masks:")
            print("VQA Label shape:", vqa_label.shape)
            print("VQA Label Mask shape:", vqa_label_mask.shape)
            print("VAQ Label shape:", vaq_label.shape)
            print("VAQ Label Mask shape:", vaq_label_mask.shape)
            print("QAV Label shape:", qav_label.shape)
            print("QAV Label Mask shape:", qav_label_mask.shape)
        
        text_id = {
            'vqa': vqa_padding_text_id, 
            'vaq': vaq_padding_text_id, 
            'qav': qav_padding_text_id
        }
        label = {
            'vqa': vqa_label, 
            'vaq': vaq_label, 
            'qav': qav_label
        }
        video_start = {
            'vqa': vqa_video_start, 
            'vaq': vaq_video_start, 
            'qav': qav_prefix_index
        }
        video_index = {
            'vqa': vqa_video_index, 
            'vaq': vaq_video_index, 
            'qav': qav_video_index
        }
        label_mask = {
            'vqa': vqa_label_mask, 
            'vaq': vaq_label_mask, 
            'qav': qav_label_mask
        }

        if verbose_shapes:
            # Print a concise overview of the first two examples of each
            print("\nConcise overview of the first two examples:")
            for key in text_id:
                print(f"{key.upper()} - Text ID (first 3 examples):", text_id[key][:3])
                print(f"{key.upper()} - Label (first 3 examples):", label[key][:3])
                print(f"{key.upper()} - Video Start:", video_start[key])
                print(f"{key.upper()} - Video Index (first 3 examples):", video_index[key][:3])
                print(f"{key.upper()} - Label Mask (first 3 examples):", label_mask[key][:3])
                print()

        return text_id, label, video_start, video_index, label_mask