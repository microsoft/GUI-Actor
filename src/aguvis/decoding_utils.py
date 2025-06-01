import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    LogitsProcessor,
    LogitsProcessorList,
    AutoModelForCausalLM,
    AutoTokenizer
)
from aguvis.constants import (
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
)

class ForceFollowTokensLogitsProcessor(LogitsProcessor):
    """
    Forces tokens B (pointer_pad_token) and C (pointer_end_token) to follow token A (pointer_start_token).
    Whenever token_a_id is generated, enqueue the forced_sequence (e.g. [B, C]).
    As long as forced tokens remain in the queue, force them in the output.
    """
    def __init__(self, token_a_id, forced_sequence=[DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN]):
        super().__init__()
        self.token_a_id = token_a_id
        self.forced_sequence = forced_sequence  # list of token IDs, e.g. [B_id, C_id]
        self.force_queue = []  # holds the tokens we still need to force

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Called at each decoding step to modify `scores`.
        
        Args:
            input_ids: shape (batch_size, seq_len). The already-decoded tokens.
            scores:    shape (batch_size, vocab_size). Model logits for the next token.
        """
        batch_size = input_ids.shape[0]
        if batch_size > 1:
            raise NotImplementedError("Batch size must be 1 for this logits processor.")
        
        # We assume batch_size=1 for simplicity; if you have multiple sequences,
        # you'll need to adapt the logic to handle each item in the batch.
        last_token_id = input_ids[0, -1].item()

        # If the last token was A, enqueue B and C
        if last_token_id == self.token_a_id:
            self.force_queue.extend(self.forced_sequence)
        
        # If we have forced tokens waiting in the queue, override the distribution
        if len(self.force_queue) > 0:
            forced_token = self.force_queue.pop(0)  # next token to force
            # Create a mask of -inf for all tokens except the forced one
            new_scores = torch.full_like(scores, float('-inf'))
            new_scores[0, forced_token] = 0.0  # log prob = 0 => prob = 1
            return new_scores
        
        # Otherwise, return scores unmodified
        return scores


if __name__ == "__main__":
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Suppose the "trigger" token is simply "A" (one token)
    # and the forced sequence is [B, C].
    # IMPORTANT: check how your tokenizer splits "A", "B", "C".
    # For GPT-2, "A", "B", "C" might appear as " A", " B", " C" (with leading space).
    # Adjust accordingly.
    token_a_id = tokenizer(" A", add_special_tokens=False)["input_ids"][0]
    token_b_id = tokenizer(" C", add_special_tokens=False)["input_ids"][0]
    token_c_id = tokenizer(" B", add_special_tokens=False)["input_ids"][0]

    # Build our processor
    processor = ForceFollowTokensLogitsProcessor(
        token_a_id=token_a_id,
        forced_sequence=[token_b_id, token_c_id]
    )

    prompt = "Hello, let's see what happens if we produce token A multiple times. A"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")
    # Apply the custom processor inside a LogitsProcessorList
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=50,
        logits_processor=LogitsProcessorList([processor]),
        do_sample=True,        # or False
        temperature=0.7        # adjust as you like
    )

    print(tokenizer.decode(output[0], skip_special_tokens=True))
