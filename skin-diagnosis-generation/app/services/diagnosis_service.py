"""
Core diagnosis service for MedGemma 4B skin diagnosis
"""

import os
import json
import asyncio
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import torch
from PIL import Image

from app.config import settings
from .language_prompts import LanguagePromptManager

logger = logging.getLogger(__name__)

class DiagnosisService:
    """Main service for handling skin diagnosis using MedGemma 4B"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.config = None
        self.is_initialized = False
        self.language_manager = LanguagePromptManager()
        
    async def initialize(self):
        """Initialize the diagnosis service"""
        try:
            logger.info("Initializing diagnosis service...")
            
            # Load configuration
            await self._load_config()
            
            # Setup environment
            await self._setup_environment()
            
            # Load model
            await self._load_model()
            
            # Create directories
            await self._create_directories()
            
            self.is_initialized = True
            logger.info("Diagnosis service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize diagnosis service: {e}")
            raise
    
    async def _load_config(self):
        """Load configuration from JSON file"""
        try:
            config_path = settings.CONFIG_FILE
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
            else:
                # Use default configuration
                self.config = self._get_default_config()
                logger.warning(f"Configuration file {config_path} not found, using defaults")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "patient": {
                "name": "Patient",
                "age": 38,
                "gender": "Unknown",
                "classification": "Skin Condition",
                "history": "No medical history provided"
            },
            "model": {
                "model_id": "google/medgemma-4b-it",
                "local_model_path": "./models/medgemma-4b-it",
                "use_local": False
            },
            "output": {
                "reports_directory": settings.REPORTS_DIR,
                "diagnosis_filename_template": "diagnosis_report_{patient_name}.md"
            }
        }
    
    async def _setup_environment(self):
        """Setup environment variables and torch settings"""
        env_config = self.config.get('environment', {})
        
        # Set environment variables
        if env_config.get('torch_compile_disable'):
            os.environ["TORCH_COMPILE_DISABLE"] = str(env_config['torch_compile_disable'])
        if env_config.get('pytorch_disable_cuda_compilation'):
            os.environ["PYTORCH_DISABLE_CUDA_COMPILATION"] = str(env_config['pytorch_disable_cuda_compilation'])
        
        # Configure torch settings
        if env_config.get('torch_dynamo_disable'):
            torch._dynamo.config.disable = env_config['torch_dynamo_disable']
        if env_config.get('torch_dynamo_suppress_errors'):
            torch._dynamo.config.suppress_errors = env_config['torch_dynamo_suppress_errors']
        if env_config.get('enable_flash_sdp') is not None:
            torch.backends.cuda.enable_flash_sdp(env_config['enable_flash_sdp'])
        if env_config.get('float32_matmul_precision'):
            torch.set_float32_matmul_precision(env_config['float32_matmul_precision'])
    
    async def _load_model(self):
        """Load the MedGemma 4B model and processor"""
        try:
            # Import here to avoid loading issues
            from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

            model_config = self.config.get('model', {})
            model_id = model_config.get('model_id', 'google/medgemma-4b-it')
            use_local = model_config.get('use_local', False)
            local_model_path = model_config.get('local_model_path', './models/medgemma-4b-it')

            # Determine model path
            if use_local and os.path.exists(local_model_path):
                model_path = local_model_path
                logger.info(f"Loading MedGemma 4B model from local path: {model_path}")
            else:
                model_path = model_id
                logger.info(f"Loading MedGemma 4B model from Hugging Face: {model_path}")

            logger.info(f"Model type: {settings.MODEL_TYPE}")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_path)
            
            # Configure quantization if requested
            if settings.MODEL_TYPE == "4bit":
                quant_config = model_config.get('quantization', {})
                compute_dtype = getattr(torch, quant_config.get('bnb_4bit_compute_dtype', 'bfloat16').replace('torch.', ''))
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=quant_config.get('load_in_4bit', True),
                    bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
                    bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
                    bnb_4bit_compute_dtype=compute_dtype
                )
            else:
                bnb_config = None
            
            # Load model with appropriate device mapping
            torch_settings = model_config.get('torch_settings', {})
            if torch.cuda.is_available():
                cuda_dtype = getattr(torch, torch_settings.get('cuda_dtype', 'bfloat16').replace('torch.', ''))
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    torch_dtype=cuda_dtype,
                    device_map=torch_settings.get('device_map', 'auto'),
                    attn_implementation=torch_settings.get('attn_implementation', 'eager'),
                )
            else:
                cpu_dtype = getattr(torch, torch_settings.get('cpu_dtype', 'float32').replace('torch.', ''))
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    torch_dtype=cpu_dtype,
                    device_map="cpu",
                    attn_implementation=torch_settings.get('attn_implementation', 'eager'),
                )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    async def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(settings.REPORTS_DIR, exist_ok=True)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Diagnosis service cleaned up")
    
    def check_cuda(self) -> bool:
        """Check if CUDA is available"""
        if torch.cuda.is_available():
            logger.info(f"CUDA available - GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            logger.warning("CUDA not available, using CPU")
            return False

    async def _load_image(self, image_path: str) -> Image.Image:
        """Load and prepare image"""
        return Image.open(image_path).convert('RGB')

    async def analyze_image(self, image_path: str, patient_info: Dict[str, Any], language: str = "en") -> Tuple[str, str]:
        """
        Analyze medical image and generate diagnosis report

        Args:
            image_path: Path to the medical image
            patient_info: Patient information dictionary
            language: Language for the analysis (en, ko, vi)

        Returns:
            Tuple of (concise_analysis, comprehensive_report)
        """
        if not self.is_initialized:
            raise RuntimeError("Diagnosis service not initialized")

        try:
            # Load and prepare the image
            image = await self._load_image(image_path)
            logger.info(f"Image loaded: {image_path}, size: {image.size}")

            # Generate concise analysis
            concise_analysis = await self._generate_concise_analysis(image, patient_info, language)

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Generate comprehensive diagnosis
            comprehensive_report = await self._generate_diagnosis_report(image, patient_info, language)

            return concise_analysis, comprehensive_report

        except Exception as e:
            logger.error(f"Error during image analysis: {e}")
            raise

    async def _generate_concise_analysis(self, image: Image.Image, patient_info: Dict[str, Any], language: str = "en") -> str:
        """Generate concise image analysis"""
        try:
            # Create concise analysis prompt using language manager
            prompt = self.language_manager.format_analysis_prompt(language, patient_info)

            # Get language-specific system message
            language_prompts = self.language_manager.get_prompts(language)
            system_message = language_prompts.get('system_message_analysis')

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]

            # Generate analysis
            result = await self._generate_response(messages, max_tokens=100)
            return result.strip()

        except Exception as e:
            logger.error(f"Error generating concise analysis: {e}")
            raise

    async def _generate_diagnosis_report(self, image: Image.Image, patient_info: Dict[str, Any], language: str = "en") -> str:
        """Generate comprehensive diagnosis report"""
        try:
            # Create diagnosis prompt using language manager
            prompt = self.language_manager.format_diagnosis_prompt(language, patient_info)

            # Get language-specific system message
            language_prompts = self.language_manager.get_prompts(language)
            system_message = language_prompts.get('system_message_diagnosis')

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]

            # Generate diagnosis
            result = await self._generate_response(messages, max_tokens=1200)
            return result.strip()

        except Exception as e:
            logger.error(f"Error generating diagnosis report: {e}")
            raise

    async def _generate_concise_analysis_stream(self, image: Image.Image, patient_info: Dict[str, Any], language: str = "en"):
        """Generate streaming concise image analysis"""
        try:
            # Create concise analysis prompt using language manager
            prompt = self.language_manager.format_analysis_prompt(language, patient_info)

            # Get language-specific system message
            language_prompts = self.language_manager.get_prompts(language)
            system_message = language_prompts.get('system_message_analysis')

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]

            # Generate streaming analysis
            async for token in self._generate_response_stream(messages, max_tokens=100):
                yield token

        except Exception as e:
            logger.error(f"Error generating streaming concise analysis: {e}")
            raise

    async def _generate_diagnosis_report_stream(self, image: Image.Image, patient_info: Dict[str, Any], language: str = "en"):
        """Generate streaming comprehensive diagnosis report"""
        try:
            # Create diagnosis prompt using language manager
            prompt = self.language_manager.format_diagnosis_prompt(language, patient_info)

            # Get language-specific system message
            language_prompts = self.language_manager.get_prompts(language)
            system_message = language_prompts.get('system_message_diagnosis')

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]

            # Generate streaming diagnosis
            async for token in self._generate_response_stream(messages, max_tokens=1200):
                yield token

        except Exception as e:
            logger.error(f"Error generating streaming diagnosis report: {e}")
            raise

    async def _generate_response(self, messages: list, max_tokens: int = 1200) -> str:
        """Generate response using the model"""
        try:
            # Process the input
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

            input_len = inputs["input_ids"].shape[-1]

            # Generation parameters
            generation_config = self.config.get('generation', {}).get('diagnosis', {})
            gen_kwargs = {
                'max_new_tokens': min(max_tokens, generation_config.get('max_new_tokens', max_tokens)),
                'do_sample': generation_config.get('do_sample', False),
                'use_cache': generation_config.get('use_cache', True),
                'pad_token_id': generation_config.get('pad_token_id') or self.processor.tokenizer.eos_token_id,
                'early_stopping': generation_config.get('early_stopping', False)
            }

            # Filter out None values
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            with torch.inference_mode():
                with torch.no_grad():
                    generation = self.model.generate(**inputs, **gen_kwargs)
                    generation = generation[0][input_len:]

            # Decode the response
            result = self.processor.decode(generation, skip_special_tokens=True)
            return result

        except Exception as e:
            logger.error(f"Error during model generation: {e}")
            raise

    async def _generate_response_stream(self, messages: list, max_tokens: int = 1200):
        """Generate streaming response using the model"""
        try:
            from transformers import TextIteratorStreamer
            import threading

            # Process the input
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

            # Generation parameters
            generation_config = self.config.get('generation', {}).get('diagnosis', {})
            gen_kwargs = {
                'max_new_tokens': min(max_tokens, generation_config.get('max_new_tokens', max_tokens)),
                'do_sample': generation_config.get('do_sample', False),
                'use_cache': False,  # Disable caching to avoid inference mode conflicts
                'pad_token_id': generation_config.get('pad_token_id') or self.processor.tokenizer.eos_token_id,
                'early_stopping': generation_config.get('early_stopping', False)
            }

            # Filter out None values
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            # Create streamer
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            gen_kwargs['streamer'] = streamer

            # Generation function to run in separate thread
            def generate_with_proper_context():
                try:
                    # Use no_grad instead of inference_mode for streaming to avoid cache conflicts
                    with torch.no_grad():
                        self.model.generate(**inputs, **gen_kwargs)
                except Exception as e:
                    logger.error(f"Error in generation thread: {e}")
                    # Signal the streamer to stop
                    streamer.end()

            # Start generation in a separate thread
            generation_thread = threading.Thread(target=generate_with_proper_context)
            generation_thread.start()

            # Yield tokens as they are generated
            for token in streamer:
                yield token

            # Wait for generation to complete
            generation_thread.join()

        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            raise


