import json
import random
import re
from typing import Any, Dict, List
from loguru import logger
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from component.template import Template


class CustomEDASFTDataset(Dataset):

    def __init__(
        self,
        file: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        template_map: Dict[str, Template],
    ) -> None:
        self.tokenizer = tokenizer
        self.template_map = template_map

        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("There are {} data in dataset".format(len(data_list)))

        random.shuffle(data_list)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def _parse_reference(self, refs: List[Dict[str, str]], **kwargs) -> str:
        tool = kwargs.get("tool")
        llm = kwargs.get("llm", "yi")

        if llm == "yi":
            if tool is None:
                ref_format = "参考资料[{idx}]\ndoc_id: {doc_id}\n{content}"
                parsed_refs = [ref_format.format(idx=idx, doc_id=elem["doc_id"], content=elem["content"]) for idx, elem in enumerate(refs, 1)]
            else:
                ref_format = "{tool}软件的参考资料[{idx}]\ndoc_id: {doc_id}\n{content}"
                parsed_refs = [ref_format.format(tool=tool, idx=idx, doc_id=elem["doc_id"], content=elem["content"]) for idx, elem in enumerate(refs, 1)]
        elif llm == "qwen":
            ref_format = "**[{idx}] 来自 {tool}软件文档 {doc_id} 的内容：**\n\n```\n{content}\n```"
            parsed_refs = [ref_format.format(tool=tool, idx=idx, doc_id=elem["doc_id"], content=elem["content"]) for idx, elem in enumerate(refs, 1)]
        else:
            raise NotImplementedError

        return "\n\n".join(parsed_refs)

    def _parse_answer(self, response: str) -> Dict[str, Any]:
        assert "<Analysis>" in response and "</Analysis>" in response, f"<Analysis> not found in\n{response}"
        analysis = response.split("<Analysis>")[1].split("</Analysis>")[0].strip()

        assert "<Score>" in response and "</Score>" in response, f"<Score> not found in\n{response}"
        score = response.split("<Score>")[1].split("</Score>")[0].strip()

        assert "<Answer>" in response and "</Answer>" in response, f"<Answer> not found in\n{response}"
        answer = response.split("<Answer>")[1].split("</Answer>")[0].strip()

        return {
            "analysis": analysis,
            "score": score,
            "answer": answer,
        }

    def __getitem__(self, index):
        # 每条数据拼接格式为: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        data = self.data_list[index]
        data = json.loads(data, strict=False)
        category = data["category"]
        conversations = data['conversation']

        input_ids, target_mask = [], []

        if category in ["qa_openroad_v1", "qa_openroad_v2", "qa_xtop_v2"]:
            template = self.template_map[category]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
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

        elif category in ["qa_xtop_v4", "qa_xtop_v5", "qa_xtop_v7", "qa_xtop_v8"]:
            template = self.template_map[category]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                reference = self._parse_reference(conv["selected_reference"])
                answer = {
                    "thought": conv["thought"],
                    "answer": conv["answer"],
                }
                answer = json.dumps(answer, ensure_ascii=False)

                # format and encode
                human = template.user_format.format(query=question, reference=reference)
                assistant = template.assistant_format.format(content=answer)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_scoring_v1"]:
            template = self.template_map[category]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                reference_thought = [elem for elem in zip(conv["reference"], conv["thought"].split("\n"))]

                # data augment
                random.shuffle(reference_thought)
                reference = [elem[0] for elem in reference_thought]
                thought = '\n'.join([elem[1] for elem in reference_thought])

                reference = self._parse_reference(reference)
                answer = f"<Analysis>\n{thought}\n</Analysis>\n<Answer>\n{conv['answer']}\n</Answer>"

                # format and encode
                human = template.user_format.format(query=question, reference=reference)
                assistant = template.assistant_format.format(content=answer)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_scoring_v2", "qa_scoring_v3", "qa_scoring_v5"]:
            template = self.template_map[category]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                answer = conv["answer"]
                parsed_answer = self._parse_answer(answer)

                # data augment
                zip_ref_score = [elem for elem in zip(conv["reference"], parsed_answer["score"].split("\n"))]
                # zip_ref_score = random.sample(zip_ref_score, random.randint(2, min(len(zip_ref_score), 5)))
                random.shuffle(zip_ref_score)
                reference = [elem[0] for elem in zip_ref_score]
                score_analysis = '\n'.join([elem[1] for elem in zip_ref_score])

                reference = self._parse_reference(reference)
                answer = f"<Analysis>\n{parsed_answer['analysis']}\n</Analysis>\n<Score>\n{score_analysis}\n</Score>\n<Answer>\n{parsed_answer['answer']}\n</Answer>"

                # format and encode
                human = template.user_format.format(query=question, reference=reference)
                assistant = template.assistant_format.format(content=answer)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False, max_length=8192)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False, max_length=8192)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_scoring_v6", "qa_scoring_v7", "qa_scoring_v8", "qa_scoring_v9"]:
            template = self.template_map[category]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                answer = conv["answer"]
                parsed_answer = self._parse_answer(answer)
                reference = conv["reference"]

                reference = self._parse_reference(reference)
                answer = f"<Analysis>\n{parsed_answer['analysis']}\n</Analysis>\n<Score>\n{parsed_answer['score']}\n</Score>\n<Answer>\n{parsed_answer['answer']}\n</Answer>"

                # format and encode
                human = template.user_format.format(query=question, reference=reference)
                assistant = template.assistant_format.format(content=answer)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False, max_length=8192)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False, max_length=8192)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_scoring_v10", "qa_scoring_v11"]:
            template = self.template_map[category]
            tool = conversations[0]["tool"]
            system_text = template.system_format.format(content=template.system).replace("{tool}", tool)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                answer = conv["answer"]
                hint = conv["hint"]
                reference = conv["reference"]

                question = f"{question}\n你可以参考这些背景知识提示来回答：\n{hint}" if len(hint) > 0 else question
                reference = self._parse_reference(reference, tool=tool)

                # format and encode
                human = template.user_format.format(query=question, reference=reference, tool=tool)
                assistant = template.assistant_format.format(content=answer)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False, max_length=8192)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False, max_length=8192)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_scoring_image_v11"]:
            template = self.template_map[category]
            tool = conversations[0]["tool"]
            system_text = template.system_format.format(content=template.system).replace("{tool}", tool)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                answer = conv["answer"]
                hint = conv["hint"]
                reference = conv["reference"]

                question = f"{question}\n你可以参考这些背景知识提示来回答：\n{hint}" if len(hint) > 0 else question
                question = f"{question}\n如果相关参考资料中包含示意图，请在答案中以“![图片描述](图片路径)”的格式插入示意图来补充说明。"
                reference = self._parse_reference(reference, tool=tool, llm="qwen")

                # format and encode
                human = template.user_format.format(query=question, reference=reference, tool=tool)
                assistant = template.assistant_format.format(content=answer)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        # elif category in ["qa_scoring_v12"]:
        #     template = self.template_map[category]
        #     tool = conversations[0]["tool"]
        #     reference = conversations[0]["reference"]
        #     reference = self._parse_reference(reference, tool=tool, llm="qwen")

        #     system_text = template.system_format.format(content=template.system).replace("{reference}", reference)
        #     input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
        #     target_mask = [0] * len(input_ids)

        #     # 拼接多轮对话
        #     for conv in conversations:
        #         question = conv["question"]
        #         answer = conv["answer"]
        #         hint = conv["hint"]

        #         question = f"{question}\n你还可以参考以下背景知识提示来回答：\n{hint}" if len(hint) > 0 else question

        #         # format and encode
        #         human = template.user_format.format(query=question, tool=tool)
        #         assistant = template.assistant_format.format(content=answer)
        #         input_tokens = self.tokenizer.encode(human, add_special_tokens=False, max_length=8192)
        #         output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False, max_length=8192)

        #         input_ids += input_tokens + output_tokens
        #         target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_scoring_v12", "qa_scoring_v13", "qa_scoring_v14"]:
            template = self.template_map[category]
            tool = conversations[0]["tool"]
            reference = conversations[0]["reference"]
            reference = self._parse_reference(reference, tool=tool, llm="qwen")

            system_text = template.system_format.format(content=template.system).replace("{tool}", tool)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                answer = conv["answer"]
                question = f"{question} 可以详细说明一下吗？"

                # format and encode
                human = template.user_format.format(query=question, tool=tool, reference=reference)
                assistant = template.assistant_format.format(content=answer)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_scoring_v14_gpt"]:
            template = self.template_map[category]
            tool = conversations[0]["tool"]
            reference = conversations[0]["reference"]
            reference = self._parse_reference(reference, tool=tool, llm="qwen")

            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                answer = conv["answer"]
                question = f"{question} 可以详细说明一下吗？"

                # format and encode
                human = template.user_format.format(query=question, tool=tool, reference=reference)
                assistant = template.assistant_format.format(content=answer)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_classify_v5"]:
            template = self.template_map[category]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                classify = conv["classify"]

                # format and encode
                human = template.user_format.format(query=question)
                assistant = template.assistant_format.format(content=classify)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        elif category in ["qa_naked_v5", "qa_naked_v7"]:
            template = self.template_map[category]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                naked_answer = conv["naked_answer"]

                # format and encode
                human = template.user_format.format(query=question)
                assistant = template.assistant_format.format(content=naked_answer)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False, max_length=8192)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False, max_length=8192)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        else:
            exit()

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        if len(input_ids) > self.max_seq_length:
            # print("Too long input_ids: ", len(input_ids))
            inputs = self.__getitem__(random.randint(0, len(self.data_list) - 1))
        else:
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


class SelectReferenceEDASFTDataset(Dataset):

    def __init__(
        self,
        file: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        template_map: Dict[str, Template],
    ) -> None:
        self.tokenizer = tokenizer
        self.template_map = template_map

        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("There are {} data in dataset".format(len(data_list)))

        random.shuffle(data_list)
        self.data_list = data_list


    def __len__(self):
        return len(self.data_list)

    def _parse_reference(self, refs: List[Dict[str, str]]) -> str:
        parsed_refs = [f'[{idx}]\n{elem["content"]}\n(doc_id: {elem["doc_id"]})' for idx, elem in enumerate(refs, 1)]
        return "\n\n".join(parsed_refs)

    def __getitem__(self, index):
        # 每条数据拼接格式为: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        data = self.data_list[index]
        data = json.loads(data, strict=False)
        category = data["category"]
        conversations = data['conversation']

        input_ids, target_mask = [], []

        if category in ["select_reference_xtop_v1", "select_reference_xtop_v2"]:
            template = self.template_map[category]
            system_text = template.system_format.format(content=template.system)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

            # 拼接多轮对话
            for conv in conversations:
                question = conv["question"]
                reference = self._parse_reference(conv["relevant_reference"])
                selected_ids = [conv["relevant_reference_doc_id"].index(_doc_id) + 1 for _doc_id in conv["selected_reference_doc_id"]]
                answer = {
                    "thought": conv["select_refer_thought"],
                    "selected_ids": selected_ids,
                    "confidence": conv["select_refer_confidence"],
                }
                answer = json.dumps(answer, ensure_ascii=False)

                # format and encode
                human = template.user_format.format(query=question, reference=reference)
                assistant = template.assistant_format.format(content=answer)
                input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        else:
            exit()

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        if len(input_ids) > self.max_seq_length:
            print("Too long input_ids: ", len(input_ids))
            inputs = self.__getitem__(random.randint(0, len(self.data_list) - 1))
        else:
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

        random.shuffle(data_list)
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
        data = json.loads(data, strict=False)
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

        # parse chosen rejected content
        chosen_content = json.loads(chosen['content'].strip(), strict=False)['answer']
        rejected_content = json.loads(rejected['content'].strip(), strict=False)['answer']

        # build response
        chosen = self.assistant_format.format(content=chosen_content, stop_token=self.tokenizer.eos_token)
        rejected = self.assistant_format.format(content=rejected_content, stop_token=self.tokenizer.eos_token)

        chosen_input_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        rejected_input_ids = self.tokenizer.encode(rejected, add_special_tokens=False)

        longer_response_length = max(len(chosen_input_ids), len(rejected_input_ids))
        if len(prompt_input_ids) + longer_response_length > self.max_seq_length:
            print("Too long input_ids: ", len(prompt_input_ids) + longer_response_length)
            inputs = self.__getitem__(random.randint(0, len(self.data_list) - 1))
        else:
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
    from transformers import AutoTokenizer, AutoConfig
    from component.template import template_dict

    def load_tokenizer(model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        # 加载tokenzier
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            # llama不支持fast
            use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
        )

        if tokenizer.__class__.__name__ == 'QWenTokenizer':
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token_id = tokenizer.eod_id
            tokenizer.eos_token_id = tokenizer.eod_id
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
        assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
        logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
        return tokenizer

    template_map={
        "qa_scoring_v12": "qwen_qa_scoring_v12",
    }
    template_map = {
        k: template_dict[v]
    for k, v in template_map.items() if v in template_dict}

    dataset = CustomEDASFTDataset(
        file="data/raw_data/qa_scoring_xtop_qualib_openroad_v11_03.jsonl",
        tokenizer=load_tokenizer("/nvme_disk1/public/weights/Qwen2.5-32B-Instruct"),
        max_seq_length=4096,
        template_map=template_map,
    )
    data = dataset[-1]

    # dataset = CustomDPODataset(
    #     # file="data/raw_data/qa_xtop_dpo_merged_v2.jsonl",
    #     file="data/raw_data/qa_xtop_dpo_merged_v3.jsonl",
    #     tokenizer=load_tokenizer("/nvme_disk1/public/weights/Qwen1.5-14B-Chat"),
    #     max_seq_length=2048,
    #     max_prompt_length=300,
    #     template=template_dict["qa_xtop_dpo"],
    # )
    # data = dataset[-1]
    # print(data)

    # dataset = SelectReferenceEDASFTDataset(
    #     file="data/raw_data/select_reference_xtop_merged.jsonl",
    #     tokenizer=load_tokenizer("/nvme_disk1/public/weights/Qwen1.5-14B-Chat"),
    #     max_seq_length=4096,
    #     template_map=template_map,
    # )
    # data = dataset[-1]
    # print(data)
