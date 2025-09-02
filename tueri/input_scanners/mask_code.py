from __future__ import annotations

import re
import torch
# from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np

from tueri.util import calculate_risk_score, get_logger
from .base import Scanner

LOGGER = get_logger()


class CodeBertaClassifier:
    def __init__(self):
        # TODO: this model needs to be trained/fine-tuned for proper code classification
        # has to be able to identify code segments in the text so that it can be masked, instead of classifying the entire text as code or NL, so that the prompt can still go through
        # also needs to be able to identify the programming language
        self.model_name = "huggingface/CodeBERTa-small-v1"
        
        # self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        # self.model = RobertaForSequenceClassification.from_pretrained(
        #     self.model_name, 
        #     num_labels=2,
        #     torch_dtype=torch.float32
        # )
        # self.model = self.model.to('cpu')
        # self.model.eval()
    
    def _is_likely_code(self, text):
        # More precise code indicators with weighted scoring
        high_confidence_patterns = [
            (r'\bdef\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*:', 3),  # Python function definition
            (r'\bfunction\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*\{', 3),  # JS function
            (r'\bclass\s+[A-Z][a-zA-Z0-9_]*\s*[\({:]', 3),  # Class definition
            (r'(?:public|private|protected)\s+(?:static\s+)?(?:void|int|String|boolean)\s+\w+\s*\(', 3),  # Java methods
            (r'\b(?:int|float|double|char|bool|void|string)\s+[a-zA-Z_]\w*\s*[\(;=]', 3),  # C/C++ declarations
            (r'#include\s*[<"][\w./]+[>"]', 3),  # C/C++ includes
            (r'import\s+(?:[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*|\{[^}]+\})\s*(?:from\s+[\'"][^\'"]+[\'"])?', 3),  # Import statements
            (r'from\s+[a-zA-Z_][\w.]*\s+import\s+[a-zA-Z_][\w,\s*]*', 3),  # Python from import
            (r'(?:const|let|var)\s+[a-zA-Z_]\w*\s*=', 2),  # JS variable declarations
            (r'[a-zA-Z_]\w*\s*=\s*(?:new\s+\w+\(|[\[\{])', 2),  # Object/array assignments
        ]
        
        medium_confidence_patterns = [
            (r'(?:if|while|for)\s*\([^)]+\)\s*\{', 2),  # Control structures with braces
            (r'(?:else\s+if|elif)\s*\([^)]*\)', 2),  # Else if statements
            (r'try\s*\{|catch\s*\([^)]*\)\s*\{', 2),  # Exception handling
            (r'(?:console\.log|print|printf|println|echo)\s*\([^)]*\)', 2),  # Print statements
            (r'return\s+(?:[^;]+;|[^}]+})', 2),  # Return statements with values
            (r'[a-zA-Z_]\w*\.[a-zA-Z_]\w*\([^)]*\)', 2),  # Method calls
            (r'(?:SELECT|INSERT|UPDATE|DELETE)\s+.*(?:FROM|INTO|SET|WHERE)', 2),  # SQL
            (r'[a-zA-Z_]\w*\[[^\]]+\]\s*=', 2),  # Array assignments
        ]
        
        low_confidence_patterns = [
            (r'//[^\n]*', 1),  # Single line comments
            (r'/\*.*?\*/', 1),  # Multi-line comments  
            (r'#[^\n]*', 1),  # Shell/Python comments (but not hashtags)
            (r'[;}]\s*$', 1),  # Lines ending with semicolon or brace
            (r'^\s*[{}]\s*$', 1),  # Lines with only braces
            (r'\$[a-zA-Z_]\w*', 1),  # Shell variables
            (r'(?:&&|\|\||==|!=|<=|>=|\+=|-=|\*=|/=)', 1),  # Programming operators
        ]
        
        score = 0
        word_count = len(text.split())
        
        # Check high confidence patterns
        for pattern, weight in high_confidence_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            score += len(matches) * weight
            
        # Check medium confidence patterns  
        for pattern, weight in medium_confidence_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            score += len(matches) * weight
            
        # Check low confidence patterns
        for pattern, weight in low_confidence_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            score += len(matches) * weight
        
        # Avoid false positives on natural language
        if word_count > 20:  # Longer texts need higher scores
            threshold = 0.3
        else:
            threshold = 0.2
            
        normalized_score = score / max(word_count, 1)
        return normalized_score > threshold
    
    def _calculate_code_likelihood(self, text):
        features = {
            'has_semicolons': ';' in text,
            'has_brackets': any(char in text for char in '{}()[]'),
            'has_operators': any(op in text for op in ['==', '!=', '<=', '>=', '&&', '||', '->', '=>']),
            'has_keywords': any(keyword in text.lower() for keyword in [
                'function', 'def', 'class', 'import', 'return', 'if', 'else', 'for', 'while'
            ]),
            'line_structure': len([line for line in text.split('\n') if line.strip()]) > 1,
            'indentation_pattern': bool(re.search(r'^\s{2,}', text, re.MULTILINE)),
            'camelcase_vars': bool(re.search(r'\b[a-z][a-zA-Z0-9]*[A-Z]', text)),
            'snake_case_vars': bool(re.search(r'\b[a-z]+_[a-z_]+\b', text)),
        }
        code_score = sum(features.values()) / len(features)
        return code_score
    
    def classify_text(self, text):
        if not text.strip():
            return {"label": "natural_language", "confidence": 0.5}
        heuristic_score = self._calculate_code_likelihood(text)
        is_code_heuristic = self._is_likely_code(text)
        if heuristic_score > 0.3 or is_code_heuristic:
            confidence = min(0.8, 0.5 + heuristic_score)
            return {"label": "code", "confidence": confidence}
        else:
            confidence = min(0.8, 0.5 + (1 - heuristic_score))
            return {"label": "natural_language", "confidence": confidence}
    
    def classify_batch(self, texts):
        return [self.classify_text(text) for text in texts]
    
    def _extract_code_segments(self, text):
        code_segments = []
        
        # Improved patterns for better code block detection
        patterns = [
            (r'(?:^|\n)((?:[ \t]{2,}.+\n){2,})', re.MULTILINE), # Multi-line code blocks (indented)
            (r'(?:def|function|class)\s+[a-zA-Z_]\w*[^{]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|:[^\n]*(?:\n[ \t]+[^\n]+)*)', re.MULTILINE | re.DOTALL), # Function/class definitions with bodies
            (r'(?:if|for|while|try|catch)\s*\([^)]*\)[^{]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|:[^\n]*(?:\n[ \t]+[^\n]+)*)', re.MULTILINE | re.DOTALL), # Control structures with blocks
            (r'(?:import|from|#include).*(?:\n|$)', re.MULTILINE), # Import/include statements
            (r'(?:const|let|var|int|float|double|char|bool)\s+[a-zA-Z_]\w*.*(?:[;=]|$)', re.MULTILINE), # Variable declarations and assignments
            (r'[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*\([^)]*\)(?:\.[a-zA-Z_]\w*\([^)]*\))*', re.MULTILINE), # Method calls with chaining
            (r'(?:SELECT|INSERT|UPDATE|DELETE).*?(?:;|\n|$)', re.MULTILINE | re.DOTALL | re.IGNORECASE), # SQL queries
            (r'(?:^|\n)(?:\$\s*)?(?:[a-zA-Z_]\w*/)?\w+(?:\s+[-\w]+)*(?:\s+[|&]+\s*\w+)*(?:\s*[;&]|$)', re.MULTILINE), # Shell commands
            (r'[^.\n]*[;{}][^.\n]*(?:\n|$)', re.MULTILINE), # Code with semicolons or braces
        ]
        
        for pattern, flags in patterns:
            for match in re.finditer(pattern, text, flags):
                segment = match.group().strip()
                if len(segment) > 5 and self._is_likely_code(segment):  # Minimum length filter
                    code_segments.append((match.start(), match.end(), segment))
        
        # Remove overlapping segments, keeping the longest ones
        code_segments.sort(key=lambda x: (x[0], -(x[1] - x[0])))  # Sort by start, then by length desc
        filtered_segments = []
        
        for start, end, segment in code_segments:
            overlaps = False
            for i, (existing_start, existing_end, existing_segment) in enumerate(filtered_segments):
                if (start < existing_end and end > existing_start):
                    # If current segment is longer, replace the existing one
                    if (end - start) > (existing_end - existing_start):
                        filtered_segments[i] = (start, end, segment)
                    overlaps = True
                    break
            if not overlaps:
                filtered_segments.append((start, end, segment))
        
        # Sort by position in text
        filtered_segments.sort(key=lambda x: x[0])
        return filtered_segments
    
    def mask_code_in_text(self, text, mask_token="[CODE]"):
        code_segments = self._extract_code_segments(text)
        if not code_segments:
            return {
                "masked_text": text,
                "code_segments": [],
                "positions": []
            }
        code_segments.sort(key=lambda x: x[0], reverse=True)
        masked_text = text
        extracted_segments = []
        positions = []
        for start, end, segment in code_segments:
            masked_text = masked_text[:start] + mask_token + masked_text[end:]
            extracted_segments.append(segment)
            positions.append((start, start + len(mask_token)))
        extracted_segments.reverse()
        positions.reverse()
        return {
            "masked_text": masked_text,
            "code_segments": extracted_segments,
            "positions": positions
        }


class MaskCode(Scanner):
    """
    A scanner that uses regex to detect code in input and mask it.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        mask_token: str = "[CODE]",
        **kwargs
    ) -> None:
        """
        Initializes the MaskCode scanner.

        Parameters:
           threshold (float): The probability threshold for code detection. Default is 0.5.
           mask_token (str): Token to replace detected code with. Default is "[CODE]".
           **kwargs: Additional parameters (ignored for compatibility).
        """
        self._threshold = threshold
        self._mask_token = mask_token
        self._classifier = CodeBertaClassifier()
        
        # Log any unused parameters for debugging
        if kwargs:
            LOGGER.debug("MaskCode received unused parameters", unused_params=list(kwargs.keys()))

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        # Classify the entire text first
        classification = self._classifier.classify_text(prompt)
        
        # Calculate risk score based on classification confidence
        if classification["label"] == "code":
            score = classification["confidence"]
        else:
            score = 1 - classification["confidence"]
        
        risk_score = calculate_risk_score(score, self._threshold)
        
        # Always mask code segments if any are detected
        masking_result = self._classifier.mask_code_in_text(prompt, self._mask_token)
        
        if masking_result['code_segments']:
            LOGGER.warning(
                "Detected and masked code in the text",
                score=score,
                threshold=self._threshold,
                code_segments_count=len(masking_result['code_segments']),
                original_text=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                masked_text=masking_result['masked_text'][:100] + "..." if len(masking_result['masked_text']) > 100 else masking_result['masked_text']
            )
            
            # Return masked text but allow the request to continue (True)
            return masking_result['masked_text'], True, risk_score
        else:
            LOGGER.debug(
                "No code detected in the text",
                score=score,
                threshold=self._threshold,
                text=prompt[:100] + "..." if len(prompt) > 100 else prompt
            )
            
            return prompt, True, risk_score