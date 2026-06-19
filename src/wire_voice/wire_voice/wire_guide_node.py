import os
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
import subprocess
import tempfile
import sys

# rokey_web의 db_logger 사용
sys.path.insert(0, os.path.expanduser('~/rokey_web'))
try:
    import db_logger
except ImportError:
    pass

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv

PACKAGE_NAME = "voice_processing"
try:
    PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
    RESOURCE_PATH = os.path.join(PACKAGE_PATH, "resource")
    ENV_PATH = os.path.join(RESOURCE_PATH, ".env")
    load_dotenv(dotenv_path=ENV_PATH)
except Exception as e:
    print(f"Failed to load .env from {PACKAGE_NAME}: {e}")

class WireGuideNode(Node):
    def __init__(self):
        super().__init__('wire_guide_node')

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            self.get_logger().error("OPENAI_API_KEY environment variable is not set!")
            
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.8, 
            openai_api_key=openai_api_key
        )
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Prompt to generate diverse, natural, and sarcastic Korean feedback
        self.prompt_template = PromptTemplate(
            input_variables=["current_order", "target_order"],
            template="""
당신은 작업자의 선 조립을 지켜보는 장난기 많고 얄미운 찐친(절친) 인공지능입니다.
반말을 사용해서 아주 친근하고 킹받게(장난스럽게) 말하세요.
현재 선의 왼쪽부터 오른쪽 배열 상태와, 목표 배열 상태가 주어집니다.

만약 현재 상태와 목표 상태가 완벽히 일치한다면, 약간 무시하는 듯하면서도 은근히 칭찬하는 말투로 대답하세요.
(예: "오~ 웬일로 맞췄냐? 니가 이걸 해낼 줄은 몰랐네 ㅋㅋ", "뭐야, 어떻게 한 거야? 기계인 줄 알았잖아 ㄷㄷ")

만약 일치하지 않는다면, 친구를 신나게 놀리고 갈구면서 어떻게 선을 움직여야 하는지 알려주세요.
(예: "아니 눈 뒀다 뭐하냐? 초록색이랑 파란색 위치를 싹 바꿔야지 으이구!", "하.. 진짜 답답하네. 빨간선을 맨 오른쪽으로 치워보라고!")

매번 똑같은 패턴의 문장 대신, 다양하고 창의적인 드립과 어휘를 사용하고, 예시와 너무 똑같이 말하지는 마세요.
최대 1~2문장으로 핵심(어떤 선을 어떻게 옮겨야 할지)은 반드시 포함해서 아주 짧고 굵게 대답하세요.

[Data]
Current arrangement (left-to-right): {current_order}
Target arrangement (left-to-right): {target_order}
"""
        )
        
        self.chain = self.prompt_template | self.llm
        
        self.publisher_ = self.create_publisher(String, '/arrangement_guide_response', 10)
        self.subscription = self.create_subscription(
            String,
            '/arrangement_guide_request',
            self.guide_request_callback,
            10
        )
        self.get_logger().info("Wire Guide Node started. Listening on /arrangement_guide_request.")

    def guide_request_callback(self, msg):
        try:
            req_data = json.loads(msg.data)
            current_order = req_data.get("current_order", "")
            target_order = req_data.get("target_order", "")
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse incoming request JSON.")
            return
            
        self.get_logger().info(f"Received request - Current: {current_order}, Target: {target_order}")
        
        try:
            try:
                db_logger.update_state(voice_status='피드백 생성 중...')
            except Exception:
                pass

            result = self.chain.invoke({
                "current_order": current_order,
                "target_order": target_order
            })
            guide_text = result.content.strip()
            self.get_logger().info(f"Generated Guide: {guide_text}")
            
            try:
                db_logger.update_state(
                    voice_status='🗣️ 피드백 재생 중...',
                    last_voice_text=f"AI: {guide_text}"
                )
            except Exception:
                pass

            # OpenAI TTS
            try:
                tts_response = self.openai_client.audio.speech.create(
                    model="tts-1",
                    voice="echo",
                    input=guide_text
                )
                
                # Save to a temporary file
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts_response.stream_to_file(tmp_file.name)
                
                # If correct, play the pre-recorded mp3 first
                if current_order == target_order:
                    try:
                        pkg_path = get_package_share_directory("wire_voice")
                        correct_mp3_path = os.path.join(pkg_path, "resource", "정답입니다.mp3")
                        if os.path.exists(correct_mp3_path):
                            self.get_logger().info("Playing 정답입니다.mp3...")
                            subprocess.run(
                                ["ffplay", "-nodisp", "-autoexit", correct_mp3_path],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                        else:
                            self.get_logger().warn(f"정답입니다.mp3 not found at {correct_mp3_path}")
                    except Exception as e:
                        self.get_logger().error(f"Failed to play 정답입니다.mp3: {e}")

                # Play LLM audio using ffplay
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", tmp_file.name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self.get_logger().info("Finished playing TTS audio.")
            except Exception as tts_err:
                self.get_logger().error(f"TTS Error: {tts_err}")

            try:
                db_logger.update_state(voice_status='')
            except Exception:
                pass

            resp_msg = String()
            resp_msg.data = json.dumps({"success": True, "guide_message": guide_text})
            self.publisher_.publish(resp_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error generating guide: {e}")
            try:
                db_logger.update_state(voice_status='')
                db_logger.set_error(f"피드백 생성 오류: {e}", source="wire_guide")
            except Exception:
                pass
            resp_msg = String()
            resp_msg.data = json.dumps({"success": False, "guide_message": "Error: Could not generate guide."})
            self.publisher_.publish(resp_msg)

def main(args=None):
    rclpy.init(args=args)
    node = WireGuideNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
