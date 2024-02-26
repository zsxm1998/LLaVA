import os
import shutil
import math
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm 
import json

import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#nltk.download('punkt')

import torch
from angle_emb import AnglE, Prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


batch_size = 8
answer_file = 'playground/patho_data/eval/quilt1m_val_answer_llava_QuiltNet-B-16_vicuna7b-base.jsonl'


question_file = 'playground/patho_data/eval/quilt1m_val.jsonl'

peft_model_id = 'SeanLee97/angle-llama-7b-nli-v2'
config = PeftConfig.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).bfloat16()
model = PeftModel.from_pretrained(model, peft_model_id).cuda()
model.eval()

def decorate_text(text: str):
    return Prompts.A.format(text=text)

@torch.inference_mode()
def calc_cosine(a, b):
    fa = model(**tokenizer([decorate_text(a)], return_tensors='pt'), output_hidden_states=True).hidden_states[-1][0, -1].float().cpu()
    fb = model(**tokenizer([decorate_text(b)], return_tensors='pt'), output_hidden_states=True).hidden_states[-1][0, -1].float().cpu()
    cosine_similarity = torch.dot(fa, fb) / (torch.norm(fa) * torch.norm(fb))
    return cosine_similarity.item()

# close-ended question
is_histology_prompts = [
    'Is this image a pathology or histology specimen?',
    'Determine if this picture is a pathological or histological slide.',
    'Identify if this image depicts a pathology specimen or a histology sample.',
    'Judge whether the provided image is a pathological or histological section.',
    'Assess if this picture represents a specimen from pathology or histology.',
    'Clarify if this image is a slide typical of pathology or histology.',
    'Evaluate if the image shown is a cut section from pathology or histology.',
    'Examine whether this picture is a specimen of pathological or histological nature.',
    'Confirm if the displayed image is a slide from the field of pathology or histology.',
    'Decide if this image is a pathological or histological sample.',
    'Analyze if this picture shows a section typical of pathology or histology.',
    'Verify whether the image is a specimen from a pathology or histology study.',
    'Ascertain if the given image represents a pathological or histological slide.',
    'Establish if this image is a section from a pathology or histology sample.',
    'Deduce if this photo is a typical representation of a pathology or histology specimen.',
    'Classify if the image before you is a cut section relevant to pathology or histology.',
    'Is this image a part of a pathology whole slide image (WSI)?',
    'Determine if this picture represents a section of a pathology WSI.',
    'Can you confirm whether this image is a segment of a pathology WSI?',
    'Assess if this image is a fragment from a pathology whole slide scan.',
    'Identify if this picture is an extract from a pathological WSI.',
    'Judge whether the provided image is a portion of a pathology WSI.',
    'Evaluate if this image shows a piece of a pathological whole slide imaging.',
    'Clarify if this image is a slice of a pathology WSI.',
    'Examine if the displayed image is a part of a pathological whole slide scan.',
    'Verify if this image represents a segment of a pathology whole slide image.',
    'Ascertain if this picture is an element of a pathology WSI.',
    'Decide if the given image is a fraction of a pathology whole slide image.',
    'Analyze if this image constitutes a part of a pathological WSI.',
    'Establish if this photo is a section of a pathology whole slide imaging.',
    'Confirm if the image shown is a piece of a pathology WSI.',
    'Inquire if this image is a part of a pathology whole slide image.',
]
tissue_type_human_prompts = [
    'What type of tissue is depicted in this pathology slide?',
    'Identify the tissue type presented in this histological image.',
    'Can you specify the type of tissue shown in this pathology specimen?',
    'Determine the tissue category in this histopathological slide.',
    'What kind of tissue is visible in this pathology image?',
    'Describe the tissue type seen in this histology specimen.',
    'Classify the type of tissue depicted in this pathological slide.',
    'Which tissue type is evident in this histological sample?',
    'Reveal the type of tissue presented in this pathology image.',
    'What tissue type does this histopathology slide represent?',
    'Analyze and state the tissue type in this pathology section.',
    'Examine the image and identify the tissue type in this histological section.',
    'Discern the type of tissue shown in this pathology photograph.',
    'What tissue category is demonstrated in this histology slide?',
    'Specify the type of tissue visible in this pathological sample.',
    'Clarify the tissue type depicted in this histopathological image.',
    'Indicate the tissue type observed in this pathology slide.',
    'Describe the type of tissue represented in this histological image.',
]
top_three_tissue_types_human_prompts = [
    'What are the top three possible tissue types depicted in this pathology image?',
    'Identify the top three tissue categories present in this histological slide.',
    'List the three most likely tissue types shown in this pathology specimen.',
    'Determine the top three tissue types visible in this histopathological image.',
    'Can you specify the three most probable tissue types in this pathology slide?',
    'Name the top three tissue categories demonstrated in this histology sample.',
    'Rank the three most evident tissue types in this pathological image.',
    'Which three tissue types are most likely represented in this histological section?',
    'Provide the top three tissue categories based on the pathology of this image.',
    'Assess and list the three most prominent tissue types in this histopathology slide.',
    'Describe the three most significant tissue types visible in this pathology specimen.',
    'Enumerate the top three tissue types observed in this histological image.',
    'Highlight the three most distinguishable tissue types in this pathology slide.',
    'Summarize the three leading tissue categories shown in this pathological sample.',
    'Identify and discuss the three primary tissue types in this histopathology image.',
    'Deduce the top three tissue types from the histological features in this picture.',
]
magnification_level_prompts = [
    'What is the magnification level of this pathology image?',
    'Can you specify the magnification scale used in this histological slide?',
    'Identify the magnification rate of this pathology specimen.',
    'Determine the enlargement level in this histopathological image.',
    'At what magnification is this pathology slide viewed?',
    'Please state the magnification degree of this histological image.',
    'What level of zoom is applied to this pathological specimen?',
    'Clarify the scale of magnification for this histopathology slide.',
    'Indicate the magnification ratio for this pathology image.',
    'Reveal the level of enlargement in this histological section.',
    'Assess and report the magnification setting of this pathology slide.',
    'Describe the magnification intensity of this histopathological sample.',
    'What is the zoom level for this pathological slide?',
    'Quantify the magnification factor in this histology image.',
    'Provide the details on the magnification used in this pathology specimen.',
    'How much is this histopathological image magnified?',
]

# open-ended question
detailed_description_prompts = [
    'Describe the following image in detail.',
    'Provide a detailed description of the given image.',
    'Give an elaborate explanation of the image you see.',
    'Share a comprehensive rundown of the presented image.',
    'Offer a thorough analysis of the image.',
    'Explain the various aspects of the image before you.',
    'Clarify the contents of the displayed image with great detail.',
    'Characterize the image using a well-detailed description.',
    'Break down the elements of the image in a detailed manner.',
    'Walk through the important details of the image.',
    'Portray the image with a rich, descriptive narrative.',
    'Narrate the contents of the image with precision.',
    'Analyze the image in a comprehensive and detailed manner.',
    'Illustrate the image through a descriptive explanation.',
    'Examine the image closely and share its details.',
    'Write an exhaustive depiction of the given image.',
]
brief_description_prompts = [
    'Describe the image concisely.',
    'Provide a brief description of the given image.',
    'Offer a succinct explanation of the picture presented.',
    'Summarize the visual content of the image.',
    'Give a short and clear explanation of the subsequent image.',
    'Share a concise interpretation of the image provided.',
    'Present a compact description of the photo’s key features.',
    'Relay a brief, clear account of the picture shown.',
    'Render a clear and concise summary of the photo.',
    'Write a terse but informative summary of the picture.',
    'Create a compact narrative representing the image presented.',
]
detailed_description_pathology_prompts = [
    'Describe the following image in detail.',
    'Provide a detailed description of the given image.',
    'Give an elaborate explanation of the image you see.',
    'Share a comprehensive rundown of the presented image.',
    'Offer a thorough analysis of the image.',
    'Explain the various aspects of the image before you.',
    'Clarify the contents of the displayed image with great detail.',
    'Characterize the image using a well-detailed description.',
    'Break down the elements of the image in a detailed manner.',
    'Walk through the important details of the image.',
    'Portray the image with a rich, descriptive narrative.',
    'Narrate the contents of the image with precision.',
    'Analyze the image in a comprehensive and detailed manner.',
    'Illustrate the image through a descriptive explanation.',
    'Examine the image closely and share its details.',
    'Write an exhaustive depiction of the given image.',
    
    'Describe the pathology-related details in this image.',
    'Provide a comprehensive analysis of the pathological aspects of this image.',
    'Give an in-depth explanation of the histological features in this image.',
    'Share a detailed breakdown of the pathology content in the displayed image.',
    'Offer a thorough description of the histological elements present in this image.',
    'Explain all the pathology-related aspects of the image before you.',
    'Clarify the pathological findings depicted in this image with great detail.',
    'Characterize the image focusing on its histological aspects.',
    'Break down the pathology details of the image in a detailed manner.',
    'Walk through the important histological details of the image.',
    'Portray the image with a focus on its pathological narrative.',
    'Narrate the contents of the image with precision, highlighting its histology.',
    'Analyze the image in a comprehensive manner, focusing on pathology.',
    'Illustrate the image through a descriptive explanation of its histological features.',
    'Examine the image closely and share details relevant to pathology.',
    'Write an exhaustive depiction of the histological aspects of the given image.',
    'Detail the pathological characteristics observed in this image.',
    'Summarize the histology-focused aspects of this image.',
    'Present a detailed overview of the pathology findings in the image.',
    'Elaborate on the histological components visible in this image.',
    'Discuss the pathology-related content of the image in detail.',
    'Convey the significance of the histological features in this image.',
    'Outline the detailed pathology information present in the image.',
    'Recount the specific histological details observed in the image.',
    'Give a meticulous description of the pathology aspects of this image.',
    'Interpret the histological significance of the features in this image.',
    'Analyze the pathology elements in this image in great detail.',
    'Illustrate the key histological details in the image with thorough explanations.',

    'What diagnosis can be made from the pathological features in this image?',
    'Based on the histopathological characteristics, what is the likely diagnosis?',
    'Determine the diagnosis from the histology displayed in this image.',
    'Identify the possible diagnosis from the pathological evidence in this picture.',
    'What medical condition does the histopathology of this image suggest?',
    'Analyze the histological features and provide a potential diagnosis.',
    'Using the pathology shown, what diagnosis would you infer?',
    'Examine the histopathological elements and state the probable diagnosis.',
    'What is the diagnosis derived from the pathology findings in this image?',
    'Assess the histopathological data and suggest a diagnosis.',
    'What condition is indicated by the pathological aspects of this image?',
    'Interpret the histological details to offer a diagnosis.',
    'From the pathology observed, what medical conclusion can be drawn?',
    'What ailment do the histopathological patterns in this image indicate?',
    'Considering the pathology in this image, what is the diagnosis?',
    'Diagnose the condition based on the histopathology evident in this picture.',
    'What disease is reflected in the pathological traits of this image?',
    'Given the histopathological presentation, what diagnosis is suggested?',
]
brief_description_pathology_prompts = [
    'Describe the image concisely.',
    'Provide a brief description of the given image.',
    'Offer a succinct explanation of the picture presented.',
    'Summarize the visual content of the image.',
    'Give a short and clear explanation of the subsequent image.',
    'Share a concise interpretation of the image provided.',
    'Present a compact description of the photo’s key features.',
    'Relay a brief, clear account of the picture shown.',
    'Render a clear and concise summary of the photo.',
    'Write a terse but informative summary of the picture.',
    'Create a compact narrative representing the image presented.',
    
    'Briefly describe the pathology elements in this image.',
    'Provide a concise overview of this pathological specimen.',
    'Offer a succinct summary of the histological features in this image.',
    'Summarize the key pathology aspects of this image.',
    'Give a brief explanation of the histological details in this picture.',
    'Share a compact interpretation of the pathology content in this image.',
    'Present a quick description of the major histological elements of this specimen.',
    'Relay a short account of the pathological findings in this image.',
    'Render a concise summary of the histological aspects of this photo.',
    'Write a brief but informative overview of the pathology in this picture.',
    'Create a quick narrative highlighting the histology in this image.',
    'Explain the pathology seen in this image in a few words.',
    'Give a rapid description of the histological layout in this image.',
    'Share a brief interpretation of the pathology evident in this specimen.',
    'Outline the main histological characteristics of this image.',
    'Quickly go over the pathology highlights in this picture.',
    'Describe the key pathological features of this image succinctly.',
    'Provide a short summary of the histological aspects in this image.',
    'Offer a quick glance at the pathological nature of this specimen.',
    'Summarize the histological and pathological findings in this image briefly.',
    'Give a concise account of the pathology depicted in this image.',
    'Share a brief overview of the histological composition of this picture.',
    'Present a compact look at the pathology in this image.',
    'Relay a brief snapshot of the histological elements in this photo.',

    'What does the pathology in this image briefly suggest?',
    'Can you summarize the diagnosis from the histopathology here?',
    'Give a quick assessment of the pathological condition in this image.',
    'What is the likely diagnosis from the histological details in this picture?',
    'Briefly, what do the pathological features indicate?',
    'In a few words, what does the histopathology of this image reveal?',
    'Offer a concise diagnosis based on the pathology observed here.',
    'What diagnosis is suggested by the histology in this image?',
    'Summarize the medical condition indicated by the pathology in this photo.',
    'State a brief diagnostic interpretation of the histopathological findings.',
    'What can be inferred about the condition from the pathology in this image?',
    'Provide a quick diagnosis based on the histological evidence.',
    'What is the histopathological conclusion for this image in brief?',
    'Briefly, what condition does the pathology of this image point to?',
    'Give a short explanation of the disease indicated by the histopathology.',
    'What medical inference can be drawn from the pathology here?',
    'Quickly identify the diagnosis from the pathological traits in this image.',
    'What key diagnosis does the histology of this picture suggest?',
]
roi_prompts = [
    'What are the regions of interest (ROIs) in this pathology image?',
    'Can you identify any significant areas in this histological slide?',
    'Highlight the key regions of interest in this pathology specimen.',
    'Point out the noteworthy areas in this histopathological image.',
    'Which areas should be focused on in this pathology slide?',
    'Describe any areas of interest in this histological image.',
    'Indicate the important regions in this pathological specimen.',
    'Are there any notable areas in this histopathology slide?',
    'What parts of this pathology image are worth paying attention to?',
    'Locate the regions of interest in this histological section.',
    'Explain the significant areas in this pathology image.',
    'Identify the key areas in this histopathological specimen.',
    'Show the areas of interest in this pathology slide.',
    'Mark the significant regions in this histology image.',
    'Which sections of this pathology specimen are most relevant?',
    'Spotlight the focal areas in this histopathological slide.',
]


question_list = []
with open(question_file, 'r') as file:
    for line in file:
        question_list.append(json.loads(line))

answer_list = []
with open(answer_file, 'r') as file:
    for line in file:
        answer_list.append(json.loads(line))


df = pd.read_csv('/c22073/datasets/Quilt-1M/quilt_1M_lookup_clean.csv')
df = df[df['split'] == 'val']
print(df.shape)

val_conv_file = 'playground/patho_data/pretrain/quilt1m_val.json'
with open(val_conv_file, 'r') as file:
    conv_list = json.load(file)

for a, b in zip(df.itertuples(), conv_list):
    if a.image_path != b['image']:
        print(a, b)


question2conv = []
for i, conv in enumerate(conv_list):
    for ques in conv['conversations'][::2]:
        assert ques['from'] == 'human'
        question2conv.append(i)


def get_first_sentence(text):
    # 首先尝试使用NLTK的句子分割器
    sentences = sent_tokenize(text)
    first_sentence = sentences[0] if sentences else None
    # 如果没有找到，使用换行符尝试分割
    if not first_sentence:
        lines = text.split('\n')
        first_sentence = lines[0] if lines else None
    if first_sentence:
        first_sentence = first_sentence.lower()
    return first_sentence

def is_affirmative_or_negative(sentence):
    # 定义一些基本的否定关键词
    negative_keywords = ['no', 'not', 'never', 'cannot', "n't"]
    # 检查否定关键词
    for word in negative_keywords:
        if word in sentence:
            return 'negative'
    return 'affirmative'

def extract_magnification(text):
    # 构建正则表达式来匹配各种放大倍数表达方式
    # 匹配像 "40x", "40 times", "4000 percent" 等格式
    pattern = r'(\d+(?:\.\d+)?)(?:\s*-?times|x| fold|-fold| percent)?'

    # 在文本中搜索匹配项
    matches = re.findall(pattern, text, re.IGNORECASE)

    for match in matches:
        number = float(match)
        
        # 如果匹配的是百分比形式，需要转换
        if 'percent' in text:
            number = number / 100

        # 返回匹配的第一个数字，可以根据需要调整逻辑
        return number

    return None  # 如果没有匹配项，则返回None

def evaluate_open_ended_qa(model_answer, reference_answers):
    model_answer_tokens = model_answer.split()
    reference_tokens = [answer.split() for answer in reference_answers]
    smoothie = SmoothingFunction().method1  # 使用平滑函数
    bleu_score = sentence_bleu(reference_tokens, model_answer_tokens, smoothing_function=smoothie)
    return bleu_score

is_histology_c, is_histology_t = 0, 0
tissue_type_c, tissue_type_t = 0, 0
magnification_level_c, magnification_level_t = 0, 0
bleu_description_list = []
bleu_roi_list = []
description_true, description_pred = [], []
roi_true, roi_pred = [], []

for answer in tqdm(answer_list):
    id = answer['question_id']
    question = answer['prompt']
    anstext = answer['text']
    row = df.iloc[question2conv[id]]
    if question in is_histology_prompts:
        is_histology_c += 1
        if not anstext:
            continue
        gt = row.not_histology == 0 #是病理图则为True，否则为False
        pr = is_affirmative_or_negative(get_first_sentence(anstext)) == 'affirmative'
        if gt == pr:
            is_histology_t += 1
    elif question in tissue_type_human_prompts+top_three_tissue_types_human_prompts:
        tissue_type_c += 1
        if not anstext:
            continue
        gt = ast.literal_eval(row.pathology)[0].lower()
        if gt in get_first_sentence(anstext):
            tissue_type_t += 1
    elif question in magnification_level_prompts:
        magnification_level_c += 1
        if not anstext:
            continue
        gt = int(row.magnification)
        gt_list = [[0,10],[10,20],[20,float('inf')]]
        pr = extract_magnification(get_first_sentence(anstext))
        if pr is not None and (gt_list[gt][0] < pr <= gt_list[gt][1]):
            magnification_level_t += 1
    elif question in detailed_description_prompts+brief_description_prompts+detailed_description_pathology_prompts+brief_description_pathology_prompts:
        if not anstext:
            continue
        description_true.append(row.caption)
        description_pred.append(anstext)
        bleu_description_list.append(evaluate_open_ended_qa(anstext, [row.caption]))
        #cosine_description_list = calc_cosine(anstext, row.caption)
    elif question in roi_prompts:
        if not anstext or isinstance(row.roi_text, float):
            continue
        roi_true.append(row.roi_text)
        roi_pred.append(anstext)
        bleu_roi_list.append(evaluate_open_ended_qa(anstext, [row.roi_text]))
        #cosine_roi_list = calc_cosine(anstext, row.roi_text)
    else:
        raise ValueError(f'Question not in any prompt list: {question}')

print(os.path.basename(answer_file))
print('is_histology accuracy:', f'{is_histology_t/is_histology_c*100:.2f}%', 'is_histology number:', is_histology_c)
print('tissue_type accuracy:', f'{tissue_type_t/tissue_type_c*100:.2f}%', 'tissue_type number:', tissue_type_c)
print('magnification_level accuracy:', f'{magnification_level_t/magnification_level_c*100:.2f}%', 'magnification_level number:', magnification_level_c)
print('description mean BLEU score:', f'{np.array(bleu_description_list).mean().item():.4f}', 'description number:', len(bleu_description_list))
print('ROI mean BLEU score:', f'{np.array(bleu_roi_list).mean().item():.4f}', 'ROI number:', len(bleu_roi_list))


#计算预测和真实的余弦相似度
with torch.inference_mode():
    true_list, pred_list = [], []
    for i in tqdm(range(0, len(description_true), batch_size), leave=False):
        true_strs = description_true[i:i+batch_size]
        pred_strs = description_pred[i:i+batch_size]
        
        tok = tokenizer([decorate_text(s) for s in true_strs], return_tensors='pt', padding=True)
        res = model(input_ids=tok['input_ids'].cuda(), attention_mask=tok['attention_mask'].cuda(), output_hidden_states=True)
        true_list.append(res.hidden_states[-1][:, -1].cpu())
        del tok, res
        torch.cuda.empty_cache()
        
        tok = tokenizer([decorate_text(s) for s in pred_strs], return_tensors='pt', padding=True)
        res = model(input_ids=tok['input_ids'].cuda(), attention_mask=tok['attention_mask'].cuda(), output_hidden_states=True)
        pred_list.append(res.hidden_states[-1][:, -1].cpu())
        del tok, res
        torch.cuda.empty_cache()
        
    true_list = torch.cat(true_list, dim=0).cuda()
    pred_list = torch.cat(pred_list, dim=0).cuda()
    cos_sim = torch.nn.functional.cosine_similarity(true_list, pred_list, dim=1)
    print('description cos similarity:', cos_sim.mean().item())

    true_list, pred_list = [], []
    for i in tqdm(range(0, len(roi_true), batch_size), leave=False):
        true_strs = roi_true[i:i+batch_size]
        pred_strs = roi_pred[i:i+batch_size]
        
        tok = tokenizer([decorate_text(s) for s in true_strs], return_tensors='pt', padding=True)
        res = model(input_ids=tok['input_ids'].cuda(), attention_mask=tok['attention_mask'].cuda(), output_hidden_states=True)
        true_list.append(res.hidden_states[-1][:, -1].cpu())
        del tok, res
        torch.cuda.empty_cache()
        
        tok = tokenizer([decorate_text(s) for s in pred_strs], return_tensors='pt', padding=True)
        res = model(input_ids=tok['input_ids'].cuda(), attention_mask=tok['attention_mask'].cuda(), output_hidden_states=True)
        pred_list.append(res.hidden_states[-1][:, -1].cpu())
        del tok, res
        torch.cuda.empty_cache()
        
    true_list = torch.cat(true_list, dim=0).cuda()
    pred_list = torch.cat(pred_list, dim=0).cuda()
    cos_sim = torch.nn.functional.cosine_similarity(true_list, pred_list, dim=1)
    print('roi cos similarity:', cos_sim.mean().item())