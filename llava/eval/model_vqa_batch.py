import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from functools import partial

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class QuestionDataset(Dataset):
    def __init__(self, questions, image_folder, image_processor,
                 mm_use_im_start_end, conv, tokenizer):
        self.questions = questions
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.mm_use_im_start_end = mm_use_im_start_end
        self.conv_template = conv
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        question_id = self.questions[index]["question_id"]
        image_file = self.questions[index]["image"]
        question_text = self.questions[index]["text"]
        image = Image.open(os.path.join(self.image_folder, image_file))
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        if self.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_text
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + question_text
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids, image, question_id, question_text


def pad_sequence_left_with_mask(sequences, batch_first=True, padding_value=0.0):
    """
    Pad a list of variable length Tensors on the left side with `padding_value`
    and create an attention mask for the padded sequences.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): If True, output will be in B x T x * format,
                                      else T x B x * format. Default: False.
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor: Padded tensor of shape (B, T, *) if batch_first is True,
                or (T, B, *) otherwise.
        Tensor: Attention mask of the same shape, with 1s in non-padded positions
                and 0s in padded positions.
    """
    # 检查sequences是否为空
    if not sequences:
        raise ValueError("The list of sequences should not be empty.")

    # 计算最长序列长度
    max_length = max(sequence.size(0) for sequence in sequences)

    # 进行左侧填充并创建attention mask
    padded_sequences = []
    attention_masks = []
    for sequence in sequences:
        pad_size = max_length - sequence.size(0)
        padded_sequence = F.pad(sequence, (pad_size, 0), "constant", padding_value)
        padded_sequences.append(padded_sequence)

        # 创建并添加对应的attention mask
        attention_mask = torch.ones_like(sequence, dtype=torch.int64)
        attention_mask = F.pad(attention_mask, (pad_size, 0), "constant", 0)
        attention_masks.append(attention_mask)

    # 如果batch_first为True，则调整批次维度的顺序
    if batch_first:
        padded_sequences = torch.stack(padded_sequences, dim=0)
        attention_mask = torch.stack(attention_masks, dim=0)
    else:
        padded_sequences = torch.stack(padded_sequences, dim=1)
        attention_mask = torch.stack(attention_masks, dim=1)

    return padded_sequences, attention_mask


def collate_fn(batch, padding_value=0):
    input_ids, images, question_ids, question_texts = list(zip(*batch))
    input_ids, attention_mask = pad_sequence_left_with_mask(input_ids, padding_value=padding_value)
    images = torch.stack(images, dim=0)
    return input_ids, attention_mask, images, question_ids, question_texts


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    conv = conv_templates[args.conv_mode]
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    dataset = QuestionDataset(questions, args.image_folder, image_processor, model.config.mm_use_im_start_end,
                              conv, tokenizer)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=16, pin_memory=True,
                            drop_last=False, collate_fn=partial(collate_fn, padding_value=tokenizer.pad_token_id))
    
    for input_ids, attention_mask, images, question_ids, question_texts in tqdm(dataloader):
        input_ids, attention_mask, images = input_ids.to(model.device), attention_mask.to(model.device), images.half().to(model.device)
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask = attention_mask, 
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        outputs = [o.strip() for o in outputs]
        outputs = [o[:-len(stop_str)] if o.endswith(stop_str) else o for o in outputs]
        outputs = [o.strip() for o in outputs]

        for question_id, prompt, text in zip(question_ids, question_texts, outputs):
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": question_id,
                                       "prompt": prompt,
                                       "text": text,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
            ans_file.flush()
    
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    eval_model(args)
