from dataclasses import dataclass
from typing import Dict
from .prompt_template import (
    QA_OPENROAD_PROMPT,
    QA_XTOP_V1_PROMPT,
    QA_XTOP_V2_PROMPT,
    QA_XTOP_V3_PROMPT,
    QA_XTOP_V4_PROMPT,
    QA_XTOP_V4_SELECT_REFERENCE_PROMPT,
    QA_XTOP_V4_DOC_QA_PROMPT,
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

# register_template(
#     template_name='internlm',
#     system_format="<|System|>:{content}\n",
#     user_format='<|User|>:{content}\n<|Bot|>:',
#     assistant_format='{content}</s>\n',
#     system="You are an AI assistant whose name is InternLM (书生·浦语).\n"
#         "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
#         "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
#     stop_word='</s>'
# )

# register_template(
#     template_name='internlm2',
#     system_format='<|im_start|>system\n{content}<|im_end|>\n',
#     user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
#     assistant_format='{content}<|im_end|>\n',
#     system="You are an AI assistant whose name is InternLM (书生·浦语).\n"
#         "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
#         "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
#     stop_word='<|im_end|>'
# )

# register_template(
#     template_name='qwen',
#     system_format='<|im_start|>system\n{content}<|im_end|>\n',
#     user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
#     assistant_format='{content}<|im_end|>\n',
#     system="You are a helpful assistant.",
#     stop_word='<|im_end|>'
# )

# register_template(
#     template_name='yi',
#     system_format='<|im_start|>system\n{content}<|im_end|>\n',
#     user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
#     assistant_format='{content}<|im_end|>\n',
#     system=None,
#     stop_word='<|im_end|>'
# )

# register_template(
#     template_name="CragEDA-openroad",
#     system_format='<|im_start|>system\n{content}<|im_end|>\n',
#     user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
#     assistant_format='{content}<|endoftext|><|im_end|>\n',
#     system = "You are the product consultant of a Electronic Design Automation (EDA) tool Openroad. \
#             Now given the user question and the related reference, you are required to answer the question referring to the provided reference. \
#             During answering the question, you have to follow these instructions:\n \
#                 1. Make your answer as rigorous as possible, do not fabricate the fact that does not mentioned in the provided referebce. \n \
#                 2. Your answer should be strongly related to the provided reference, provide concrete solution for the answer, and do not ignore the precondition in the query. \n \
#                 3. Your answer should be concise. Do not generate repetition in your answer\n \
#                 4. There may be some unrelated content in the reference, you should distinguish and ignore them, and give your answer only referring to the relevant reference \n",
#     stop_word='<|im_end|>'
# )

# register_template(
#     template_name="CragEDA-xtop",
#     system_format='<|im_start|>system\n{content}<|im_end|>\n',
#     user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
#     assistant_format='{content}<|endoftext|><|im_end|>\n',
#     system = "你是华大九天公司的一名专业的EDA产品顾问。现在给定客户问题query和提供的参考资料knowledge，希望你能根据参考资料回答客户的问题，并给出相关指令的使用范例。\n\
#         * 如果提供的参考资料为空，直接输出“抱歉，您的问题答案不在知识库范围内。\n\
#         * 如果你认为无法通过提供的参考资料给出答案，或者无法确信答案的正确性，直接输出“抱歉，我无法准确解答您的问题。\n\
#         * 如果用户问题是以英文来提问的，请用英文来回复。\n\
#         * 如果用户问题包含中文，必须用中文回答。\n\
#         * 提供的资料里面可能包含一些无关的内容，在生成答案的过程中，你必须筛选无关的资料，只根据有关的资料作答。\n\
#         注意! \n \
#         1. 你的回答应尽可能谨慎，宁愿答不出来，也不要错答乱答 \n\
#         2. 你不能凭空捏造资料中没有的回答 \n\
#         3. 你的回答应该与query紧密相关，给出具体的解决方案，不要无视query中的前提条件。\n\
#         4. 如果你无法解决客户的问题，直接输出“抱歉，我无法准确解答您的问题。\n \
#         5. 如果参考资料中存在与问题相关的指令介绍或代码示例，也请一并提供，用markdown格式输出。\n\
#         6. 回答中不允许出现参考资料的序号（如“参考资料[?]”或“[?]”等）。\n \
#         7. 对于release note类参考资料，其中的每条信息是在说??版本修复了哪些bug，增加了哪些特性，而不是在说某指令有什么功能。\n\
#         8. ！！！对于资料中的英文EDA专业名词（如fix hold, fix setup, corner, postmask, transition, leakage power, legalization等），不要翻译成中文. \n\
#         9. ！！！你的回复应该划分成多个观点，并使用markdown格式的有序或无序列表来进行描述！！！",
#     stop_word='<|im_end|>'
# )

# register_template(
#     template_name="general-qa",
#     system_format='<|im_start|>system\n{content}<|im_end|>\n',
#     user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
#     assistant_format='{content}<|endoftext|><|im_end|>\n',
#     system = "Your are a QA assistant who refer to the related reference and answer the question of the users. \
#             Now given the user question and the related reference, you are required to answer the question referring to the provided reference. \
#             During answering the question, you have to follow these instructions:\n \
#                 1. Make your answer as rigorous as possible, do not fabricate the fact that does not mentioned in the provided referebce. \n \
#                 2. Your answer should be strongly related to the provided reference, provide concrete solution for the answer, and do not ignore the precondition in the query. \n \
#                 3. Your answer should be concise. Do not generate repetition in your answer\n",
#     stop_word='<|im_end|>'
# )

register_template(
    template_name="qa_openroad",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_OPENROAD_PROMPT,
    stop_word='<|im_end|>'
)

# register_template(
#     template_name="qa_xtop_v1",
#     system_format='<|im_start|>system\n{content}<|im_end|>\n',
#     user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
#     assistant_format='{content}<|endoftext|><|im_end|>\n',
#     system = QA_XTOP_V1_PROMPT,
#     stop_word='<|im_end|>'
# )

register_template(
    template_name="qa_xtop_v2",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_XTOP_V2_PROMPT,
    stop_word='<|im_end|>'
)

# register_template(
#     template_name="qa_xtop_v3",
#     system_format='<|im_start|>system\n{content}<|im_end|>\n',
#     user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
#     assistant_format='{content}<|endoftext|><|im_end|>\n',
#     system = QA_XTOP_V3_PROMPT,
#     stop_word='<|im_end|>'
# )

# register_template(
#     template_name="qa_xtop_v4",
#     system_format='<|im_start|>system\n{content}<|im_end|>\n',
#     user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
#     assistant_format='{content}<|endoftext|><|im_end|>\n',
#     system = QA_XTOP_V4_PROMPT,
#     stop_word='<|im_end|>'
# )

register_template(
    template_name="qa_xtop_v4_select_reference",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_XTOP_V4_SELECT_REFERENCE_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_xtop_v4_doc_qa",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_XTOP_V4_DOC_QA_PROMPT,
    stop_word='<|im_end|>'
)

register_template(
    template_name="qa_xtop_dpo",
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user question:\n{query}<|im_end|>\n<|im_start|>Reference related to the question:\n{reference}<|im_end|>\n<|im_start|>answer:<|startoftext|>\n',
    assistant_format='{content}<|endoftext|><|im_end|>\n',
    system = QA_XTOP_V4_PROMPT,
    stop_word='<|im_end|>'
)

# register_template(
#     template_name="orion",
#     system_format='<s>',
#     user_format='Human: {content}\n\nAssistant: </s>',
#     assistant_format='{content}</s>',
#     system='',
#     stop_word='</s>',
# )

# register_template(
#     template_name='deepseek',
#     system_format=None,
#     user_format='User: {content}\n\nAssistant: ',
#     assistant_format='{content}<｜end▁of▁sentence｜>',
#     system=None,
#     stop_word='<｜end▁of▁sentence｜>'
# )

# # todo 更优雅的实现方式
# register_template(
#     template_name='chatglm2',
#     system_format=None,
#     user_format='[Round {idx}]\n\n问：{content}\n\n答：',
#     assistant_format='{content}',
#     system=None,
#     stop_word='</s>',
# )

# register_template(
#     template_name='chatglm3',
#     system_format='{content}',
#     user_format='{content}',
#     assistant_format='{content}',
#     system="You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.",
#     stop_word='</s>',
# )

# register_template(
#     template_name='ziya2',
#     system_format=None,
#     user_format='<human>:{content} <bot>:',
#     assistant_format='{content}</s>',
#     system=None,
#     stop_word='</s>',
# )

# register_template(
#     template_name="xverse",
#     system_format=None,
#     user_format='Human: {content}\n\nAssistant: ',
#     assistant_format='{content}<|endoftext|>',
#     system=None,
#     stop_word='<|endoftext|>',
# )

# register_template(
#     template_name='minicpm',
#     system_format=None,
#     user_format='<用户>{content}<AI>',
#     assistant_format='{content}</s>',
#     system=None,
#     stop_word='</s>'
# )

# register_template(
#     template_name='zephyr',
#     system_format='<|system|>\n{content}</s>',
#     user_format='<|user|>\n{content}</s>\n<|assistant|>\n',
#     assistant_format='{content}</s>\n',
#     system=None,
#     stop_word='</s>'
# )

# register_template(
#     template_name='mistral',
#     system_format='<s>',
#     user_format='[INST]{content}[/INST]',
#     assistant_format='{content}</s>',
#     system='',
#     stop_word='</s>'
# )

# register_template(
#     template_name='mixtral',
#     system_format='<s>',
#     user_format='[INST]{content}[/INST]',
#     assistant_format='{content}</s>',
#     system='',
#     stop_word='</s>'
# )

# register_template(
#     template_name='baichuan',
#     system_format=None,
#     user_format='<reserved_102>{content}<reserved_103>',
#     assistant_format='{content}</s>',
#     system=None,
#     stop_word='</s>'
# )

# register_template(
#     template_name='baichuan2',
#     system_format=None,
#     user_format='<reserved_106>{content}<reserved_107>',
#     assistant_format='{content}</s>',
#     system=None,
#     stop_word='</s>'
# )

# register_template(
#     template_name='vicuna',
#     system_format='{content}\n',
#     user_format='USER: {content} ASSISTANT:',
#     assistant_format='{content}</s>',
#     system="A chat between a curious user and an artificial intelligence assistant. "
#         "The assistant gives helpful, detailed, and polite answers to the user's questions.",
#     stop_word='</s>'
# )

# register_template(
#     template_name='llama2',
#     system_format='<<SYS>>\n{content}\n<</SYS>>\n\n',
#     user_format='[INST]{content}[/INST]',
#     assistant_format='{content} </s>',
#     system="You are a helpful, respectful and honest assistant. "
#         "Always answer as helpfully as possible, while being safe. "
#         "Your answers should not include any harmful, unethical, "
#         "racist, sexist, toxic, dangerous, or illegal content. "
#         "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
#         "If a question does not make any sense, or is not factually coherent, "
#         "explain why instead of answering something not correct. "
#         "If you don't know the answer to a question, please don't share false information.",
#     stop_word='</s>'
# )

# register_template(
#     template_name='gemma',
#     system_format='<bos>',
#     user_format='<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n',
#     assistant_format='{content}<eos>\n',
#     system='',
#     stop_word='<eos>'
# )


# if __name__ == '__main__':
#     model_name_or_path = ''
