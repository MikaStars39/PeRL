import openai
import pandas as pd
from tqdm import tqdm
import argparse
import concurrent.futures
import logging
import json
import time
import subprocess
import requests
import signal
import os
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class InferenceProcessor:

    def __init__(self, args: argparse.Namespace):
        """
        初始化处理器。

        Args:
            args (argparse.Namespace): 从命令行解析的参数。
        """
        self.args = args
        self.client: Optional[openai.OpenAI] = None
        self.server_process: Optional[subprocess.Popen] = None

        self.data_to_process: List[Dict[str, Any]] = []

    def _start_server(self) -> bool:
        """
        启动 vLLM OpenAI 兼容服务器并等待其准备就绪。

        Returns:
            bool: 如果服务器成功启动则返回 True，否则返回 False。
        """
        logging.info(f"正在启动 vLLM 服务器，模型: {self.args.model_name}...")
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.args.model_name,
            "--max-model-len",
            str(self.args.max_model_len),
            "--port",
            str(self.args.port),
            "--tensor-parallel-size",
            str(self.args.tensor_parallel_size),
        ]

        try:
            logging.info(f"服务器启动命令: {' '.join(command)}")
            self.server_process = subprocess.Popen(command)
            logging.info(f"服务器进程已启动，PID: {self.server_process.pid}")

            if self._wait_for_server():
                self.client = openai.OpenAI(base_url=self.args.base_url,
                                            api_key=self.args.api_key,
                                            timeout=None)
                return True
            else:
                logging.error("vLLM 服务器启动失败。")
                self._stop_server()
                return False
        except Exception as e:
            logging.error(f"启动 vLLM 服务器时发生异常: {e}")
            return False

    def _wait_for_server(self, timeout: int = 600) -> bool:

        start_time = time.time()
        health_check_url = f"http://127.0.0.1:{self.args.port}/health"
        logging.info(f"等待服务器在 {health_check_url} 上线...")

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_check_url, timeout=5)
                if response.status_code == 200:
                    logging.info("服务器已成功启动并准备就绪！")
                    return True
            except requests.exceptions.RequestException:
                pass

            # 检查子进程是否意外退出
            if self.server_process and self.server_process.poll() is not None:
                logging.error("服务器进程意外终止。")
                return False
            time.sleep(5)

        logging.error(f"服务器在 {timeout} 秒内未能启动。")
        return False

    def _stop_server(self):
        if self.server_process:
            logging.info(f"正在关闭 vLLM 服务器 (PID: {self.server_process.pid})...")
            # 使用 os.kill 发送信号在 Linux/macOS 上更可靠
            try:
                os.kill(self.server_process.pid, signal.SIGINT)
            except (ProcessLookupError, AttributeError):
                # 如果进程已不存在或在Windows上 os.kill 不可用
                self.server_process.send_signal(signal.SIGINT)

            try:
                self.server_process.wait(timeout=30)
                logging.info("服务器已优雅关闭。")
            except subprocess.TimeoutExpired:
                logging.warning("服务器未能优雅关闭，将强制终止。")
                self.server_process.kill()
                logging.info("服务器已被强制终止。")

            self.server_process = None

    def _load_data(self):
        logging.info(f"正在从 {self.args.input_file} 读取数据...")
        try:
            df = pd.read_parquet(self.args.input_file)
            self.data_to_process = df.to_dict('records')

        except FileNotFoundError as e:
            logging.error(f"文件未找到: {e}")
            raise
        except Exception as e:
            logging.error(f"加载数据时出错: {e}")
            raise

    def _request_api(self, prompt: str) -> List[str]:

        if not self.client:
            return []
        try:
            response = self.client.chat.completions.create(
                model=self.args.model_name,
                messages=[
                    # {
                    # "role":
                    # "system",
                    # "content":
                    # "Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: \"<think>\n {thoughts} </think>\n\". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After \"</think>\n,\" in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions.",
                    # }, 
                    {
                    "role": "user",
                    "content": prompt + "Please reason step by step, and put your final answer within \\boxed{{}}."
                }],
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                n=self.args.n_samples,
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            logging.error(f"调用 API 时出错: {e}")
            return []

    def _fetch_responses_for_item(self, item: Dict[str,
                                                   Any]) -> Dict[str, Any]:
        prompt_raw = item.get('prompt')
        # 支持 prompt 为消息列表或纯字符串（包括 numpy array）
        if hasattr(prompt_raw, 'tolist'):
            prompt_raw = prompt_raw.tolist()

        prompt_text = ""
        if isinstance(prompt_raw, (list, tuple)):
            # 优先取列表中第二个且 role 为 user 的 content
            if (len(prompt_raw) > 1 and isinstance(prompt_raw[1], dict)
                    and prompt_raw[1].get('role') == 'user'
                    and 'content' in prompt_raw[1]):
                prompt_text = prompt_raw[1]['content']
            else:
                # 退化为查找首个 user 消息
                for msg in prompt_raw:
                    if (isinstance(msg, dict)
                            and msg.get('role') == 'user'
                            and 'content' in msg):
                        prompt_text = msg['content']
                        break
        elif isinstance(prompt_raw, dict) and 'content' in prompt_raw:
            prompt_text = prompt_raw['content']
        elif isinstance(prompt_raw, str):
            prompt_text = prompt_raw
        else:
            prompt_text = str(prompt_raw)

        responses = self._request_api(prompt_text)
        item['responses'] = responses
        return item

    def run_inference(self) -> List[Dict[str, Any]]:
        processed_data = []
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.args.max_workers) as executor:
            future_to_item = {
                executor.submit(self._fetch_responses_for_item, item): item
                for item in self.data_to_process
            }

            logging.info(f"已提交 {len(self.data_to_process)} 个任务，等待响应...")

            progress_bar = tqdm(
                concurrent.futures.as_completed(future_to_item),
                total=len(self.data_to_process),
                desc="处理进度")

            for future in progress_bar:
                try:
                    result_item = future.result()
                    processed_data.append(result_item)
                except Exception as e:
                    item = future_to_item[future]
                    logging.error(
                        f"处理问题 '{item.get('question', 'N/A')}' 时发生错误: {e}")

        return processed_data

    def _save_results(self, data: List[Dict[str, Any]]):
        if data:
            logging.info(f"处理完成，共获得 {len(data)} 条结果。")
            try:
                pd.DataFrame(data).to_parquet(self.args.output_file)
                logging.info(f"结果已保存到 {self.args.output_file}")
            except Exception as e:
                logging.error(f"保存结果到 {self.args.output_file} 时出错: {e}")
        else:
            logging.warning("没有生成任何有效结果。")

    def run(self):
        try:
            if not self._start_server():
                return
            self._load_data()
            results = self.run_inference()
            self._save_results(results)
        except Exception as e:
            logging.error(f"在执行过程中发生致命错误: {e}")
        finally:
            self._stop_server()


def main():
    parser = argparse.ArgumentParser(description="使用 vLLM 并发执行推理任务并保存结果。")

    parser.add_argument("--input_file",
                        type=str,
                        default="./data/algebrarium-warmup/warmup.parquet")
    parser.add_argument("--output_file",
                        type=str,
                        default="gsm8k.llama3.2-1b.parquet")

    parser.add_argument("--model_name",
                        type=str,
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="要使用的模型路径或名称。")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len",
                        type=int,
                        default=32768,
                        help="模型的最大长度。")

    parser.add_argument("--port", type=int, default=9999, help="vLLM 服务监听的端口。")
    parser.add_argument("--api_key",
                        type=str,
                        default="EMPTY",
                        help="OpenAI API 的密钥（对于 vLLM 通常为 'EMPTY'）。")

    parser.add_argument("--n_samples",
                        type=int,
                        default=16,
                        help="每个提示生成的样本数。")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048)

    parser.add_argument("--max_workers",
                        type=int,
                        default=128,
                        help="并发执行的最大工作线程数。")

    args = parser.parse_args()
    args.base_url = f"http://127.0.0.1:{args.port}/v1"

    processor = InferenceProcessor(args)
    processor.run()


if __name__ == "__main__":
    main()
