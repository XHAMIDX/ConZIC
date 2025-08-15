import torch
import requests
from torch import nn
from PIL import Image
import sys
import os

# Add AlphaCLIP to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'AlphaCLIP'))

class CLIP(nn.Module):
    def __init__(self, model_name):
        super(CLIP, self).__init__()
        # model name: e.g. ViT-B/32 for AlphaCLIP
        print ('Initializing AlphaCLIP model...')
        
        try:
            from alpha_clip.alpha_clip import load
            # Use the specified model name or default to ViT-B/32
            if model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"]:
                self.model_name = model_name
            else:
                print(f"Model {model_name} not found, using default ViT-B/32")
                self.model_name = "ViT-B/32"
            
            # Load model on CPU first, will be moved to device later
            self.model, self.preprocess = load(self.model_name, device="cpu")
            self.model.eval()
            
            # For now, use full image as mask (no masking)
            self.use_masking = False
            
            self.cuda_has_been_checked = False
            print (f'AlphaCLIP model {self.model_name} initialized.')
        except ImportError as e:
            print(f"Error importing AlphaCLIP: {e}")
            print("Please ensure AlphaCLIP is properly installed and accessible")
            raise

    def to(self, device):
        """Move the model to the specified device"""
        self.model = self.model.to(device)
        return self

    def check_cuda(self):
        self.cuda_available = next(self.model.parameters()).is_cuda
        self.device = next(self.model.parameters()).get_device()
        if self.cuda_available:
            print ('Cuda is available.')
            print ('Device is {}'.format(self.device))
        else:
            print ('Cuda is not available.')
            print ('Device is {}'.format(self.device))

    @torch.no_grad()
    def compute_image_representation_from_image_path(self, image_path):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        # image_path: the path of the image
        image = Image.open(image_path)
        return self.compute_image_representation_from_image_instance(image)

    def compute_image_representation_from_image_instance(self, image):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image using AlphaCLIP's preprocess function
        image_tensor = self.preprocess(image).unsqueeze(0)
        
        if self.cuda_available:
            image_tensor = image_tensor.cuda(self.device)
        
        # Create full mask (all ones) for alpha parameter - using full image as mask
        # AlphaCLIP expects alpha to be the same size as the image
        batch_size, channels, height, width = image_tensor.shape
        alpha = torch.ones(batch_size, height, width, device=image_tensor.device)
        
        # Get image embeddings from AlphaCLIP with alpha mask
        image_embeds = self.model.encode_image(image_tensor, alpha)
        
        return image_embeds

    def compute_text_representation(self, text_list):
        """
        IMPORTANT: This method is NOT used by ConZIC for text generation.
        ConZIC uses BERT for text generation, AlphaCLIP is only used for vision.
        This method is kept for compatibility but should not be used.
        """
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        
        # Use AlphaCLIP's tokenize function
        from alpha_clip.alpha_clip import tokenize
        
        # Tokenize text
        text_tokens = tokenize(text_list)
        
        if self.cuda_available:
            text_tokens = text_tokens.cuda(self.device)
        
        # Get text embeddings from AlphaCLIP
        text_embeds = self.model.encode_text(text_tokens)
        
        return text_embeds

    def compute_image_text_similarity_via_embeddings(self, image_embeds, text_embeds):
        '''
            image_embeds: batch x embed_dim
            text_embeds: batch x len(text_list) x embed_dim
        '''
        text_embeds = text_embeds.view(image_embeds.shape[0], -1, text_embeds.shape[-1])
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds.unsqueeze(-1)
        
        # Compute similarity (AlphaCLIP uses cosine similarity by default)
        logits_per_text = torch.matmul(text_embeds, image_embeds)
        logits_per_image = logits_per_text.squeeze(-1)
        
        # CRITICAL FIX: Normalize CLIP scores to be comparable to BERT probabilities
        # Original CLIP scores are very small (0.003), making them irrelevant in final scoring
        # Normalize to range [0, 1] to match BERT probability scale
        clip_scores_normalized = (logits_per_image - logits_per_image.min()) / (logits_per_image.max() - logits_per_image.min() + 1e-8)
        
        return clip_scores_normalized, logits_per_image

    def compute_image_text_similarity_via_raw_text(self, image_embeds, text_list):
        """
        CRITICAL FIX: Use AlphaCLIP's text encoder for similarity scoring
        This maintains the vision-language alignment while keeping BERT for generation
        """
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        
        # Use AlphaCLIP's tokenize function for CLIP-compatible text processing
        from alpha_clip.alpha_clip import tokenize
        
        # Tokenize text using AlphaCLIP tokenizer
        text_tokens = tokenize(text_list)
        
        if self.cuda_available:
            text_tokens = text_tokens.cuda(self.device)
        
        # Get text embeddings from AlphaCLIP for similarity scoring
        text_embeds = self.model.encode_text(text_tokens)
        
        # Get normalized similarity scores
        similarity_scores, raw_scores = self.compute_image_text_similarity_via_embeddings(image_embeds, text_embeds)
        
        return similarity_scores, raw_scores

    ### -------------------- functions for building index ---------------------- ###
    def compute_batch_index_image_features(self, image_list):
        '''
            # list of image instances
        '''
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        
        # Process each image individually for now
        # Could be optimized for batch processing
        image_embeds_list = []
        for image in image_list:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0)
            if self.cuda_available:
                image_tensor = image_tensor.cuda(self.device)
            
            # Create full mask (all ones) for alpha parameter
            batch_size, channels, height, width = image_tensor.shape
            alpha = torch.ones(batch_size, height, width, device=image_tensor.device)
            
            image_embeds = self.model.encode_image(image_tensor, alpha)
            image_embeds_list.append(image_embeds)
        
        # Concatenate all embeddings
        image_embeds = torch.cat(image_embeds_list, dim=0)
        return image_embeds # len(image_list) x embed_dim

    def compute_batch_index_text_representation(self, text_list):
        if not self.cuda_has_been_checked:
            self.check_cuda()
            self.cuda_has_been_checked = True
        else:
            pass
        
        # Use AlphaCLIP's tokenize function
        from alpha_clip.alpha_clip import tokenize
        
        # Tokenize text
        text_tokens = tokenize(text_list)
        
        if self.cuda_available:
            text_tokens = text_tokens.cuda(self.device)
        
        # Get text embeddings from AlphaCLIP
        text_embeds = self.model.encode_text(text_tokens)
        
        return text_embeds

