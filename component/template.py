from dataclasses import dataclass
from typing import Dict
from .prompt_template import (
    QA_OPENROAD_PROMPT,
    QA_XTOP_V2_PROMPT,
    QA_XTOP_V4_PROMPT,
    QA_XTOP_SELECT_REFERENCE_PROMPT,
    QA_XTOP_DPO_PROMPT,
    QA_SCORING_V1_PROMPT,
    QA_SCORING_V2_PROMPT,
)


@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str
    # stop_token_id: int


template_dict: Dict[str, Template] = dict()


def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
        # stop_token_id=stop_token_id
    )


# 注册template
register_template(
    template_name='default',
    system_format='System: {content}\n\n',
    user_format='User: {content}\nAssistant: ',
    assistant_format='{content} {stop_token}',
    system=None,
    stop_word=None
)

register_template(
    template_name="qa_openroad",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_OPENROAD_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_xtop_v2",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_XTOP_V2_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_xtop_v4",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_XTOP_V4_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_xtop_dpo",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_XTOP_DPO_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="select_reference_xtop",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_XTOP_SELECT_REFERENCE_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_scoring_v1",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_SCORING_V1_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_scoring_v2",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_SCORING_V2_PROMPT,
    stop_word='<|im_end|>'
)
