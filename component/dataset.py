import json
import random
from re import A
from typing import Any, Dict, List, Optional
from loguru import logger
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from component.template import Template
from component.retriever import build_tfidf_retriever


class UnifiedSFTDataset(Dataset):
    """
    统一的数据处理dataset
    """
    def __init__(self, file, tokenizer, max_seq_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据拼接格式为: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        # setting system information
        if self.system_format is not None:
            system = data['system'].strip() if 'system' in data.keys() else self.system
            # system信息不为空
            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human = self.user_format.format(content=human, stop_token=self.tokenizer.eos_token)
            assistant = self.assistant_format.format(content=assistant, stop_token=self.tokenizer.eos_token)

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)


            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class CragEDASFTDataset(Dataset):

    def __init__(self, file, tokenizer, max_seq_length, template,template_general_qa):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.system_format_general_qa = template_general_qa.system_format
        self.user_format_general_qa = template_general_qa.user_format
        self.assistant_format_general_qa = template_general_qa.assistant_format
        self.system_general_qa = template_general_qa.system

        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据拼接格式为: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        if data["category"]=="Brainstorming":
            # setting system information
            if self.system_format is not None:
                system = data['system'].strip() if 'system' in data.keys() else self.system
                # system信息不为空
                if system is not None:
                    system_text = self.system_format.format(content=system)
                    input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                    target_mask = [0] * len(input_ids)

            conversations = data['conversation']
            # 拼接多轮对话
            for i, conv in enumerate(conversations):
                question = conv["question"].strip()
                reference = conv["reference_content"].strip()
                answer = conv["answer"].strip()


                human = self.user_format.format(query=question, reference = reference, stop_token=self.tokenizer.eos_token)
                assistant = self.assistant_format.format(content=answer, stop_token=self.tokenizer.eos_token)
                # print(human)
                # print(assistant)
                # exit()
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)


                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)
        elif data["category"]=="general_qa":
            if self.system_format_general_qa is not None:
                system = data['system'].strip() if 'system' in data.keys() else self.system_general_qa
                # system信息不为空
                if system is not None:
                    system_text = self.system_format_general_qa.format(content=system)
                    input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                    target_mask = [0] * len(input_ids)

            conversations = data['conversation']
            # 拼接多轮对话
            for i, conv in enumerate(conversations):
                question = conv["question"].strip()
                reference = conv["reference_content"].strip()
                answer = conv["answer"].strip()


                human = self.user_format_general_qa.format(query=question, reference = reference, stop_token=self.tokenizer.eos_token)
                assistant = self.assistant_format_general_qa.format(content=answer, stop_token=self.tokenizer.eos_token)
                # print(human)
                # print(assistant)
                # exit()
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)


                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)
        else:
            exit()

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class CustomEDASFTDataset(Dataset):

    def __init__(
        self,
        file: str,
        knowledge_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        template_map: Dict[str, Template],
    ) -> None:
        self.tokenizer = tokenizer
        self.template_map = template_map
        self.retriever = build_tfidf_retriever(
            file_path=knowledge_path,
            split_text=False,
            k=30,
        )

        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list
        random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def _parse_reference(self, refs: List[Dict[str, str]]) -> str:
        parsed_refs = [f'[{idx}]\n{elem["content"]}\n(doc_id: {elem["doc_id"]})' for idx, elem in enumerate(refs, 1)]
        return "\n\n".join(parsed_refs)

    def _data_aug(self, data: Dict[str, Any], data_aug_prob: Optional[List[float]] = None):
        def _random_add_negative(_data):
            # save positive
            _data["positive_reference_doc_id"] = _data["reference_doc_id"]
            _data["positive_reference"] = _data["reference"]
            # add negative
            relevant_documents = self.retriever.get_relevant_documents(_data["question"])
            neg_documents = [doc for doc in relevant_documents
                             if doc.metadata["doc_id"] not in _data["reference_doc_id"]]
            neg_documents = random.sample(neg_documents, random.randint(1, 3))
            _data["reference_doc_id"] += [doc.metadata["doc_id"] for doc in neg_documents]
            _data["reference"] += [{"doc_id": doc.metadata["doc_id"], "content": doc.page_content} for doc in neg_documents]
            return _data

        def _random_shuffle_reference(_data):
            reference_doc_id = _data["reference_doc_id"]
            reference = _data["reference"]
            zip_reference = [(doc_id, ref) for doc_id, ref in zip(reference_doc_id, reference)]

            random.shuffle(zip_reference)
            _data["reference_doc_id"] = [elem[0] for elem in zip_reference]
            _data["reference"] = [elem[1] for elem in zip_reference]
            return _data

        def _remove_reference(_data):
            _data["reference_doc_id"] = []
            _data["reference"] = []
            _data["answer"] = "抱歉，您的问题答案不在知识库范围内。" if not _data["answer"].encode("utf-8").isalpha() else \
                              "Sorry, the answer is not in our knowledge base."
            return _data

        def _remove_positive(_data):
            if "selected_reference_doc_id" in _data:
                _reference_doc_id, _reference = [], []
                for _doc_id, _doc in zip(_data["reference_doc_id"], _data["reference"]):
                    if _doc_id not in _data["selected_reference_doc_id"]:
                        _reference_doc_id.append(_doc_id)
                        _reference.append(_doc)
                _data["reference_doc_id"], _data["reference"] = _reference_doc_id, _reference
                _data["analysis"] = "这些参考资料都与客户问题无关。" if not _data["answer"].encode("utf-8").isalpha() else \
                              "None of these references are related to query."
                _data["answer"] = "抱歉，我无法准确解答您的问题。" if not _data["answer"].encode("utf-8").isalpha() else \
                                  "Sorry, I can't answer your question accurately."
            elif "positive_reference_doc_id" in _data:
                _reference_doc_id, _reference = [], []
                for _doc_id, _doc in zip(_data["reference_doc_id"], _data["reference"]):
                    if _doc_id not in _data["positive_reference_doc_id"]:
                        _reference_doc_id.append(_doc_id)
                        _reference.append(_doc)
                _data["reference_doc_id"], _data["reference"] = _reference_doc_id, _reference
                _data["answer"] = "抱歉，我无法准确解答您的问题。" if not _data["answer"].encode("utf-8").isalpha() else \
                                  "Sorry, I can't answer your question accurately."
            else:
                _data["reference_doc_id"] = []
                _data["reference"] = []
                _data["answer"] = "抱歉，您的问题答案不在知识库范围内。" if not _data["answer"].encode("utf-8").isalpha() else \
                                  "Sorry, the answer is not in our knowledge base."
            return _data

        data_aug_func = [_random_add_negative, _random_shuffle_reference, _remove_reference, _remove_positive]
        data_aug_prob = [0.8, 0.5, 0.05, 0.] if data_aug_prob is None else data_aug_prob
        assert len(data_aug_func) == len(data_aug_prob)

        for func, prob in zip(data_aug_func, data_aug_prob):
            if random.random() < prob:
                data = func(data)

        return data

    def __getitem__(self, index):
        # 每条数据拼接格式为: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        data = self.data_list[index]
        data = json.loads(data, strict=False)
        category = data["category"]
        conversations = data['conversation']

        input_ids, target_mask = [], []

        if category in ["qa_openroad_v1", "qa_openroad_v2"]:
            template = self.template_map[category]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                # 数据增强
                conv = self._data_aug(conv, data_aug_prob=[0., 0., 0., 0.])

                question = conv["question"]
                reference = self._parse_reference(conv["reference"])
                answer = conv["answer"]

                # format and encode
                human = template.user_format.format(query=question, reference=reference, stop_token=self.tokenizer.eos_token)
                assistant = template.assistant_format.format(content=answer, stop_token=self.tokenizer.eos_token)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_xtop_v1", "qa_xtop_v2", "qa_xtop_v3"]:
            template = self.template_map[category]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                # 数据增强
                conv = self._data_aug(conv, data_aug_prob=[0.8, 1.0, 0., 0.])

                question = conv["question"]
                reference = self._parse_reference(conv["reference"])
                answer = conv["answer"]

                # format and encode
                human = template.user_format.format(query=question, reference=reference, stop_token=self.tokenizer.eos_token)
                assistant = template.assistant_format.format(content=answer, stop_token=self.tokenizer.eos_token)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_xtop_v4"]:
            task = random.choice(["select_reference", "doc_qa", "all"])
            template = self.template_map[f"{category}_{task}"]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                conv["reference_doc_id"] = conv["relevant_reference_doc_id"]
                conv["reference"] = conv["relevant_reference"]
                # data augment
                if len(conv["relevant_reference_doc_id"]) == len(conv["selected_reference_doc_id"]):
                    conv = self._data_aug(conv, data_aug_prob=[1.0, 1.0, 0., 0.])
                else:
                    conv = self._data_aug(conv, data_aug_prob=[0., 1.0, 0., 0.])

                # data process
                reference_doc_id, reference = conv["reference_doc_id"], conv["reference"]
                if len(reference) > 5 and not len(conv["selected_reference"]) > 5:
                    neg_reference_doc_id, neg_reference = [], []
                    for _doc_id, _doc in zip(reference_doc_id, reference):
                        if _doc_id not in conv["selected_reference_doc_id"]:
                            neg_reference_doc_id.append(_doc_id)
                            neg_reference.append(_doc)
                    reference_doc_id = conv["selected_reference_doc_id"] + random.sample(neg_reference_doc_id, 5 - len(conv["selected_reference_doc_id"]))
                    reference = conv["selected_reference"] + random.sample(neg_reference, 5 - len(conv["selected_reference"]))

                selected_ids = [reference_doc_id.index(_doc_id) + 1 for _doc_id in conv["selected_reference_doc_id"] if _doc_id in reference_doc_id]

                question = conv["question"]
                reference = self._parse_reference(reference)
                if task == "all":
                    answer = {
                        "thought": conv["analysis"],
                        "selected_ids": selected_ids,
                        "answer": conv["answer"],
                    }
                elif task == "select_reference":
                    answer = {
                        "thought": conv["analysis"],
                        "selected_ids": selected_ids,
                    }
                elif task == "doc_qa":
                    answer = {
                        "thought": conv["analysis"],
                        "answer": conv["answer"],
                    }
                else:
                    raise NotImplementedError
                answer = json.dumps(answer, ensure_ascii=False)

                # format and encode
                human = template.user_format.format(query=question, reference=reference, stop_token=self.tokenizer.eos_token)
                assistant = template.assistant_format.format(content=answer, stop_token=self.tokenizer.eos_token)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        else:
            exit()

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class ChatGLM2SFTDataset(UnifiedSFTDataset):

    def __getitem__(self, index):
        # 每条数据格式为: [gMASK]sop [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        data = self.data_list[index]
        data = json.loads(data)

        input_ids = self.tokenizer.get_prefix_tokens()
        target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            human = self.user_format.format(content=human, idx=i + 1)
            assistant = self.assistant_format.format(content=assistant)

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class ChatGLM3SFTDataset(UnifiedSFTDataset):

    def __getitem__(self, index):
        # [gMASK]sop <|system|>xxx<|user|>xxx<|assistant|>xxx<eos>
        data = self.data_list[index]
        data = json.loads(data)
        system = data['system'].strip() if 'system' in data.keys() else self.system
        input_ids = self.tokenizer.get_prefix_tokens() + \
                    [self.tokenizer.get_command(f"<|system|>")] + \
                    self.tokenizer.encode(system, add_special_tokens=False)
        target_mask = [0] * len(input_ids)

        conversations = data['conversation']
        # 拼接多轮对话
        for i, conv in enumerate(conversations):
            human = conv['human'].strip()
            assistant = conv['assistant'].strip()

            input_tokens = [self.tokenizer.get_command(f"<|user|>")] + \
                           self.tokenizer.encode(human, add_special_tokens=False) + \
                           [self.tokenizer.get_command(f"<|assistant|>")]
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class UnifiedDPODataset(Dataset):
    """
    统一的DPO数据集
    """
    def __init__(self, file, tokenizer, max_seq_length, max_prompt_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def build_prompt_input_ids(self, system, history):
        """
        chatglm2: [gMASK]sop [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        chatglm3: [gMASK]sop <|system|>xxx<|user|>xxx<|assistant|>xxx<eos>
        others: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        """
        # chatglm模型具有特殊的起始token
        if self.template_name in ['chatglm2', 'chatglm3']:
            prompt_input_ids = self.tokenizer.get_prefix_tokens()
        else:
            prompt_input_ids = []

        # collect system information
        if self.system_format is not None:
            system = system if system is not None else self.system
            # system信息不为空
            if system is not None:
                if self.template_name == 'chatglm3':
                    prompt_input_ids += [self.tokenizer.get_command(f"<|system|>")] + self.tokenizer.encode(system, add_special_tokens=False)
                else:
                    system_text = self.system_format.format(content=system)
                    prompt_input_ids += self.tokenizer.encode(system_text, add_special_tokens=False)

        # collect history
        for i, conv in enumerate(history):
            role = conv['role'].strip()
            content = conv['content'].strip()

            assert role != 'system', 'there should not be more than one system information'
            if role == 'user':
                if self.template_name == 'chatglm2':
                    human = self.user_format.format(content=content, idx=i//2 + 1)
                    input_ids = self.tokenizer.encode(human, add_special_tokens=False)
                elif self.template_name == 'chatglm3':
                    input_ids = [self.tokenizer.get_command(f"<|user|>")] + \
                                self.tokenizer.encode(content, add_special_tokens=False) + \
                                [self.tokenizer.get_command(f"<|assistant|>")]
                else:
                    human = self.user_format.format(content=content, stop_token=self.tokenizer.eos_token)
                    input_ids = self.tokenizer.encode(human, add_special_tokens=False)
            elif role == 'assistant':
                if self.template_name in ['chatglm2', 'chatglm3']:
                    input_ids = self.tokenizer.encode(content, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                else:
                    assistant = self.assistant_format.format(content=content, stop_token=self.tokenizer.eos_token)
                    input_ids = self.tokenizer.encode(assistant, add_special_tokens=False)
            else:
                raise Exception('role error')
            prompt_input_ids += input_ids

        return prompt_input_ids

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        chosen = data['chosen']
        rejected = data['rejected']
        assert len(chosen) == len(rejected)

        # 判断第0个是否为system
        if chosen[0]['role'] == 'system':
            system = chosen[0]['content'].strip()
            history = chosen[1:-1]  # 对话上文
            chosen, rejected = chosen[-1], rejected[-1]
        else:
            system = None
            history = chosen[:-1]  # 对话上文
            chosen, rejected = chosen[-1], rejected[-1]

        # build prompt
        prompt_input_ids = self.build_prompt_input_ids(system, history)

        # build response
        if self.template_name in ['chatglm2', 'chatglm3']:
            chosen_input_ids = self.tokenizer.encode(chosen['content'], add_special_tokens=False) + [self.tokenizer.eos_token_id]
            rejected_input_ids = self.tokenizer.encode(rejected['content'], add_special_tokens=False) + [self.tokenizer.eos_token_id]
        else:
            chosen = self.assistant_format.format(content=chosen['content'], stop_token=self.tokenizer.eos_token)
            rejected = self.assistant_format.format(content=rejected['content'], stop_token=self.tokenizer.eos_token)

            chosen_input_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
            rejected_input_ids = self.tokenizer.encode(rejected, add_special_tokens=False)

        # truncate by max_seq_length
        longer_response_length = max(len(chosen_input_ids), len(rejected_input_ids))
        # if combined sequence is too long, truncate the prompt
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            max_prompt_length = max(self.max_prompt_length, self.max_seq_length - longer_response_length)
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        # if that's still too long, truncate the response
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            chosen_input_ids = chosen_input_ids[: self.max_seq_length - len(prompt_input_ids)]
            rejected_input_ids = rejected_input_ids[: self.max_seq_length - len(prompt_input_ids)]

        chosen_labels = [-100] * len(prompt_input_ids) + chosen_input_ids
        chosen_input_ids = prompt_input_ids + chosen_input_ids
        rejected_labels = [-100] * len(prompt_input_ids) + rejected_input_ids
        rejected_input_ids = prompt_input_ids + rejected_input_ids
        assert len(chosen_labels) == len(chosen_input_ids)
        assert len(rejected_labels) == len(rejected_input_ids)

        inputs = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=[1]*len(prompt_input_ids),
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=[1]*len(chosen_input_ids),
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=[1]*len(rejected_input_ids),
            rejected_labels=rejected_labels,
        )
        return inputs

    # 为了适配DPOTrainer的接口
    def map(self, func, **kwargs):
        return self


class CustomDPODataset(Dataset):
    """
    自定义的DPO数据集
    """
    def __init__(self, file, tokenizer, max_seq_length, max_prompt_length, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def build_prompt_input_ids(self, system, history):
        """
        {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        """
        prompt_input_ids = []

        # collect system information
        if self.system_format is not None:
            system = system if system is not None else self.system
            # system信息不为空
            if system is not None:
                system_text = self.system_format.format(content=system)
                prompt_input_ids += self.tokenizer.encode(system_text, add_special_tokens=False)

        # collect history
        for i, conv in enumerate(history):
            role = conv['role'].strip()

            assert role != 'system', 'there should not be more than one system information'
            if role == 'user':
                query = conv['query'].strip()
                reference = conv['reference'].strip()
                human = self.user_format.format(query=query, reference=reference, stop_token=self.tokenizer.eos_token)
                input_ids = self.tokenizer.encode(human, add_special_tokens=False)
            elif role == 'assistant':
                content = conv['content'].strip()
                assistant = self.assistant_format.format(content=content, stop_token=self.tokenizer.eos_token)
                input_ids = self.tokenizer.encode(assistant, add_special_tokens=False)
            else:
                raise Exception('role error')
            prompt_input_ids += input_ids

        return prompt_input_ids

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        chosen = data['chosen']
        rejected = data['rejected']
        assert len(chosen) == len(rejected)

        # 判断第0个是否为system
        if chosen[0]['role'] == 'system':
            system = chosen[0]['content'].strip()
            history = chosen[1:-1]  # 对话上文
            chosen, rejected = chosen[-1], rejected[-1]
        else:
            system = None
            history = chosen[:-1]  # 对话上文
            chosen, rejected = chosen[-1], rejected[-1]

        # build prompt
        prompt_input_ids = self.build_prompt_input_ids(system, history)

        # build response
        chosen = self.assistant_format.format(content=chosen['content'], stop_token=self.tokenizer.eos_token)
        rejected = self.assistant_format.format(content=rejected['content'], stop_token=self.tokenizer.eos_token)

        chosen_input_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        rejected_input_ids = self.tokenizer.encode(rejected, add_special_tokens=False)

        # truncate by max_seq_length
        longer_response_length = max(len(chosen_input_ids), len(rejected_input_ids))
        # if combined sequence is too long, truncate the prompt
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            max_prompt_length = max(self.max_prompt_length, self.max_seq_length - longer_response_length)
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        # if that's still too long, truncate the response
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            chosen_input_ids = chosen_input_ids[: self.max_seq_length - len(prompt_input_ids)]
            rejected_input_ids = rejected_input_ids[: self.max_seq_length - len(prompt_input_ids)]

        chosen_labels = [-100] * len(prompt_input_ids) + chosen_input_ids
        chosen_input_ids = prompt_input_ids + chosen_input_ids
        rejected_labels = [-100] * len(prompt_input_ids) + rejected_input_ids
        rejected_input_ids = prompt_input_ids + rejected_input_ids
        assert len(chosen_labels) == len(chosen_input_ids)
        assert len(rejected_labels) == len(rejected_input_ids)

        inputs = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=[1]*len(prompt_input_ids),
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=[1]*len(chosen_input_ids),
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=[1]*len(rejected_input_ids),
            rejected_labels=rejected_labels,
        )
        return inputs

    # 为了适配DPOTrainer的接口
    def map(self, func, **kwargs):
        return self


if __name__ == "__main__":
    from component.template import template_dict

    # template_map={
    #     "qa_openroad_v1": "qa_openroad",
    #     "qa_openroad_v2": "qa_openroad",
    #     "qa_xtop_v1": "qa_xtop_v1",
    #     "qa_xtop_v2": "qa_xtop_v2",
    #     "qa_xtop_v4": "qa_xtop_v4",
    # }
    # template_map = {
    #     k: template_dict[v]
    # for k, v in template_map.items() if v in template_dict}

    # dataset = CustomEDASFTDataset(
    #     file="/home/ubuntu/tairu/code/huada-docqa-data-generation/data/qa_data/qa_xtop_openroad_v2.jsonl",
    #     knowledge_path="data/raw_data/ref_xtop.json",
    #     tokenizer=None,
    #     max_seq_length=8192,
    #     template_map=template_map,
    # )
    # data = dataset[-1]
    # print(data)

    dataset = CustomDPODataset(
        file="data/raw_data/qa_xtop_dpo_merged.jsonl",
        tokenizer=None,
        max_seq_length=8192,
        max_prompt_length=8192,
        template=template_dict["qa_xtop_dpo"],
    )
    data = dataset[-1]
    print(data)
