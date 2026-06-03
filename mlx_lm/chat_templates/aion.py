ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def apply_chat_template(
    messages,
    add_generation_prompt=True,
    continue_final_message=False,
    **_kwargs,
):
    parts = []
    for index, message in enumerate(messages):
        role = message["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"Unsupported Aion chat role: {role}")

        parts.append(ROLE_TOKENS[role])
        parts.append("\n")
        parts.append(message["content"])
        is_final_prefill = continue_final_message and index == len(messages) - 1
        if not is_final_prefill:
            parts.append("<|end|>")
        parts.append("\n")

    if add_generation_prompt:
        parts.append("<|assistant|>\n")

    return "".join(parts)