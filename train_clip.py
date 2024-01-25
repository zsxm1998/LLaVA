import os
import argparse

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor, CLIPPreTrainedModel
from transformers.utils import ModelOutput
from torchvision.transforms import CenterCrop,Compose,Lambda,Normalize,RandomHorizontalFlip,RandomResizedCrop,Resize,ToTensor
from sklearn.metrics import accuracy_score, classification_report


class PatchSet(Dataset):
    def __init__(self, image_dir, label_dir, key, transform):
        self.image_dir, self.label_dir = image_dir, label_dir
        self.key, self.transform = key, transform
        image_list = os.listdir(image_dir)
        self.data = []
        if len(image_list) > 50000:
            for image in image_list:
                self.data.append((image, os.path.splitext(image)[0]+'.pth'))
        else:
            for image in image_list:
                label = torch.load(os.path.join(label_dir, os.path.splitext(image)[0]+'.pth'))[key]
                self.data.append((image, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, label = self.data[index]
        if isinstance(label, str):
            label = torch.load(os.path.join(self.label_dir, os.path.splitext(image)[0]+'.pth'))[self.key]
        image = self.transform(Image.open(os.path.join(self.image_dir, image)))
        return dict(pixel_values=image, labels=label.to(torch.long))
    

class ClipForVisionTokenClassification(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    base_model_prefix = "vision_tower"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CLIPVisionModel):
            module.gradient_checkpointing = value

    def __init__(self, config, num_classes=2, select_layer=-2):
        super().__init__(config)
        self.config.num_classes = num_classes
        self.config.select_layer = select_layer
        self.vision_tower = CLIPVisionModel(config)
        self.vision_tower.requires_grad_(False)
        self.head = nn.Linear(config.hidden_size, num_classes)
        self.post_init()

    def forward(self, pixel_values, labels=None, return_dict=True):
        image_forward_outs = self.vision_tower(pixel_values, output_hidden_states=True, return_dict=True)
        image_features = image_forward_outs.hidden_states[self.config.select_layer][:, 1:] #[B, L, D]
        logits = self.head(image_features) #[B, L, num_classes]
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.transpose(1,2), labels)
        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return ModelOutput(loss=loss, logits=logits)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.stack([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def preprocess_logits(logits, labels):
    return logits.argmax(dim=-1)

def compute_metrics(p):
    assert p.label_ids.shape == p.predictions.shape, f'p.predictions: {p.predictions.shape}, p.label_ids: {p.label_ids.shape}'
    print(f'{classification_report(y_true=p.label_ids.reshape(-1), y_pred=p.predictions.reshape(-1), digits=4)}')
    return {'accuracy': accuracy_score(y_true=p.label_ids.reshape(-1), y_pred=p.predictions.reshape(-1))}


def train(args):
    model = ClipForVisionTokenClassification.from_pretrained(args.model)

    image_processor = CLIPImageProcessor.from_pretrained(args.model)
    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )
    train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])
    val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])
    train_dataset = PatchSet(os.path.join(args.image_dir, 'train'),
                             os.path.join(args.label_dir, 'train'),
                             (model.config.image_size // model.config.patch_size) ** 2,
                             train_transforms)
    val_dataset = PatchSet(os.path.join(args.image_dir, 'val'),
                           os.path.join(args.label_dir, 'val'),
                           (model.config.image_size // model.config.patch_size) ** 2,
                           val_transforms)
    
    training_args = TrainingArguments(
        output_dir=args.output,
        logging_dir=os.path.join(args.output, 'logs'),
        report_to='tensorboard',
        evaluation_strategy='epoch',
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size//torch.cuda.device_count(),
        per_device_eval_batch_size=args.batch_size//torch.cuda.device_count(),
        learning_rate=1e-3,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        data_collator=collate_fn,
        preprocess_logits_for_metrics=preprocess_logits,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        # checkpoint = None
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default='/medical-data/zsxm/datasets/liverWSI/edge08/images')
    parser.add_argument("--label_dir", type=str, default='/medical-data/zsxm/datasets/liverWSI/edge08/labels')
    parser.add_argument("--model", type=str, default="/medical-data/zsxm/codes/LLaVA/checkpoints/clip-vit-large-patch14-336")
    parser.add_argument("--output", type=str, default="./checkpoints/ztestCLIP/clip-vit-large-patch14-336_edge08_epoch30")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()


    train(args)
