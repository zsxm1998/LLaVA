from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


model_path = "./checkpoints/llava-v1.5-7b"
prompt = "这是一张什么图片，用中文回答"
image_file = "/medical-data/zsxm/public_dataset/image-caption/Quilt-1M/images/_1DgD-Au5RE_image_2ba38918-a770-4e75-847a-b55b0c376847.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0.5,
    "top_p": 0.5,
    "num_beams": 1,
    "max_new_tokens": 1024
})()

eval_model(args)

#CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/llava-v1.5-7b --load-8bit

#CUDA_VISIBLE_DEVICES=0,1,2 python llava/eval/model_vqa_batch.py --model-path checkpoints/zpatho_0pretrain/llava_QuiltNet-B-16 --model-base checkpoints/vicuna-7b-v1.5 --image-folder /medical-data/zsxm/public_dataset/image-caption/Quilt-1M/images --question-file playground/patho_data/eval/quilt1m_val.jsonl --answers-file playground/patho_data/eval/quilt1m_val_answer_QuiltNet_patho_pretrain_batch.jsonl --conv-mode patho_pretrain --batch_size 32