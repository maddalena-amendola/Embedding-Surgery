import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["TRANSFORMERS_CACHE"] = cache_path+"models/"

class BGEReranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-gemma', max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path+"models/")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_path+"models/")
        self.model.eval()
        self.max_length = max_length
        self.yes_token_id = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        self.prompt = ("Given a query A and a passage B, determine whether the passage "
                       "contains an answer to the query by providing a prediction of either 'Yes' or 'No'.")
        self.sep = "\n"

    def _prepare_inputs(self, pairs):
        prompt_inputs = self.tokenizer(self.prompt, return_tensors=None, add_special_tokens=False)['input_ids']
        sep_inputs = self.tokenizer(self.sep, return_tensors=None, add_special_tokens=False)['input_ids']
        inputs = []

        for query, passage in pairs:
            query_inputs = self.tokenizer(f'A: {query}', return_tensors=None, add_special_tokens=False,
                                          max_length=self.max_length * 3 // 4, truncation=True)
            passage_inputs = self.tokenizer(f'B: {passage}', return_tensors=None, add_special_tokens=False,
                                            max_length=self.max_length, truncation=True)
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=self.max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            item['input_ids'] += sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)

        return self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=self.max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

    def rank(self, query, docs, doc_ids, return_scores=False):
        """
        Return the likelihood score of 'Yes' as the next token after each input.
        """
        pairs = [[query, doc] for doc in docs]
        inputs = self._prepare_inputs(pairs)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            next_token_logits = logits[:, -1, :]  # get logits for next token prediction
            yes_scores = next_token_logits[:, self.yes_token_id]
        
        scores = yes_scores.tolist()
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        if return_scores:
            return ranked
        return [doc for doc, _ in ranked]

class QwenReranker:
    def __init__(self, model_name="Qwen/Qwen3-Reranker-8B", max_length=8192, use_flash_attention=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_path+"models/")

        if use_flash_attention:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                cache_dir=cache_path+"models/"
            ).cuda().eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_path+"models/").eval()

        self.max_length = max_length
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ids in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ids + self.suffix_tokens

        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs):
        logits = self.model(**inputs).logits[:, -1, :]
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return log_probs[:, 1].exp().tolist()  # probability of "yes"

    def rank(self, query, docs, doc_ids, instruction=None, return_scores=False):
        pairs = [self.format_instruction(instruction, query, doc) for doc in docs]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        if return_scores:
            return ranked
        return [doc for doc, _ in ranked]
