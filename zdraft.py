import torch
import torch.nn.functional as F
from langchain.chains import RetrievalQA
from llava.model.builder import load_pretrained_model
from transformers import ResNetModel, ResNetBackbone
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from transformers import LlamaTokenizer


def pad_sequence_left_with_mask(sequences, batch_first=False, padding_value=0.0):
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


tokenizer, model, image_processor, context_len = load_pretrained_model('checkpoints/llava-v1.5-7b', None, 'llava-v1.5-7b')

input_text = ['<image>\nThis is the first sentence<image>.', '<image>\nThis is the second sentence.\n<image>']
image_files = [['/medical-data/zsxm/public_dataset/image-caption/ARCH/books_set/images/00046229-7247-4d0e-bac7-dedbd207c4e5.png',
                '/medical-data/zsxm/public_dataset/image-caption/ARCH/books_set/images/00046229-7247-4d0e-bac7-dedbd207c4e5.png'],
               ['/medical-data/zsxm/public_dataset/image-caption/ARCH/books_set/images/00046229-7247-4d0e-bac7-dedbd207c4e5.png',
                '/medical-data/zsxm/public_dataset/image-caption/ARCH/books_set/images/00046229-7247-4d0e-bac7-dedbd207c4e5.png']]

input_ids = [tokenizer_image_token(s, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for s in input_text]


input_ids, attention_mask = pad_sequence_left_with_mask(
    input_ids,
    batch_first=True,
    padding_value=tokenizer.pad_token_id)
input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
print('input_ids.shape:', input_ids.shape)


image_lists = [torch.cat([image_processor.preprocess(Image.open(f), return_tensors='pt')['pixel_values'] for f in files]).half().cuda() for files in image_files]
image_lists = torch.stack(image_lists)
print('image_lists.shape:', image_lists.shape)
print('image_lists[1].shape:', image_lists[1].shape)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        attention_mask = attention_mask,
        images=image_lists,
        do_sample=True,
        temperature=0.2,
        top_p=None,
        num_beams=1,
        max_new_tokens=1024,
        use_cache=True)
    output_str_list = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
    print(output_str_list)