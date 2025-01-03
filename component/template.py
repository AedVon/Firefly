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
    QA_SCORING_V3_PROMPT,
    QA_SCORING_V5_PROMPT,
    QA_CLASSIFY_V5_PROMPT,
    QA_NAKED_V5_PROMPT,
    QA_SCORING_V6_PROMPT,
    QA_SCORING_V7_PROMPT,
    QA_SCORING_V8_PROMPT,
    QA_SCORING_V9_PROMPT,
    QA_SCORING_V10_PROMPT,
    QA_SCORING_V11_PROMPT,
    QA_SCORING_IMAGE_V11_PROMPT,
    QA_SCORING_V12_PROMPT,
    QA_SCORING_V13_PROMPT,
    QA_SCORING_V14_PROMPT,
    QA_SCORING_GPT_V14_PROMPT,
)


@dataclass
class Template:
    template_name: str
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
    system=QA_OPENROAD_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_openroad",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}\n\nReference related to the question:\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_OPENROAD_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_xtop_v2",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system=QA_XTOP_V2_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_xtop_v4",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system=QA_XTOP_V4_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_xtop_v4",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}\n\nReference related to the question:\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_XTOP_V4_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_xtop_dpo",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system=QA_XTOP_DPO_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="select_reference_xtop",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system=QA_XTOP_SELECT_REFERENCE_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_scoring_v1",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system=QA_SCORING_V1_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_scoring_v2",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system=QA_SCORING_V2_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_scoring_v2",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}\n\nReference related to the question:\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_SCORING_V2_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_scoring_v2",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>output:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system=QA_SCORING_V3_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_scoring_v3",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}\n\nReference related to the question:\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_SCORING_V3_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_scoring_v5",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}\n\nReference related to the question:\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_SCORING_V5_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_classify_v5",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_CLASSIFY_V5_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_naked_v5",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_NAKED_V5_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_scoring_v6",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}\n\nReference related to the question:\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_SCORING_V6_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_scoring_v7",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}\n\nReference:\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_SCORING_V7_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_scoring_v8",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}\n\nReference:\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_SCORING_V8_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_scoring_v9",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\nquestion:\n{query}\n\nReference:\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_SCORING_V9_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_scoring_v10",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n这是一条关于{tool}工具的问题：\n{query}\n\n参考资料：\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_SCORING_V10_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="yi_qa_scoring_v11",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n这是一条关于{tool}软件的问题：\n{query}\n\n参考资料：\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_SCORING_V11_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qwen_qa_scoring_image_v11",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n这是一条关于{tool}工具的问题：\n{query}\n\n参考资料：\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>\n',
    system=QA_SCORING_IMAGE_V11_PROMPT,
    stop_word='<|im_end|>'
)

# register_template(
#     template_name="qwen_qa_scoring_v12",
#     system_format='<|im_start|>system\n{content}<|im_end|>\n',
#     user_format='<|im_start|>user\n这是一条关于{tool}工具的问题：\n{query}<|im_end|>\n<|im_start|>assistant\n',
#     assistant_format='{content}<|im_end|>',
#     system = QA_SCORING_V12_PROMPT,
#     stop_word='<|im_end|>'
# )

register_template(
    template_name="qwen_qa_scoring_v12",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n这是一条关于{tool}工具的问题：\n{query}\n\n参考资料：\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>',
    system=QA_SCORING_V12_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qwen_qa_scoring_v13",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n这是一条关于{tool}工具的问题：\n{query}\n\n参考资料：\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>',
    system=QA_SCORING_V13_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qwen_qa_scoring_v14",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n这是一条关于{tool}工具的问题：\n{query}\n\n参考资料：\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>',
    system=QA_SCORING_V14_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qwen_qa_scoring_gpt_v14",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n这是一条关于{tool}工具的问题：\n{query}\n\n参考资料：\n{reference}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|im_end|>',
    system=QA_SCORING_GPT_V14_PROMPT,
    stop_word='<|im_end|>'
)
